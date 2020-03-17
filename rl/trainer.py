import os
from time import time
from collections import defaultdict, OrderedDict
import gzip
import pickle
import h5py
import copy

import torch
import wandb
import numpy as np
import moviepy.editor as mpy
from tqdm import tqdm, trange
import env
import gym
from gym import spaces
from sklearn.externals import joblib

from rl.policies import get_actor_critic_by_name
from rl.meta_ppo_agent import MetaPPOAgent
from rl.meta_sac_agent import MetaSACAgent
from rl.rollouts import RolloutRunner
from rl.dataset import HERSampler
from util.logger import logger
from util.pytorch import get_ckpt_path, count_parameters, to_tensor
from util.mpi import mpi_sum
from util.gym import observation_size, action_size



def get_agent_by_name(algo, use_ae=False):
    if algo == "sac":
        if use_ae:
            from rl.sac_ae_agent import SACAEAgent
            return SACAEAgent
        else:
            from rl.sac_agent import SACAgent
            return SACAgent
    elif algo == "ppo":
        from rl.ppo_agent import PPOAgent
        return PPOAgent
    elif algo == 'ddpg':
        from rl.ddpg_agent import DDPGAgent
        return DDPGAgent
    elif algo == 'td3':
        from rl.td3_agent import TD3Agent
        return TD3Agent


class Trainer(object):
    def __init__(self, config):
        self._config = config
        self._is_chef = config.is_chef

        # create a new environment
        self._env = gym.make(config.env, **config.__dict__)
        self._env_eval = gym.make(config.env, **copy.copy(config).__dict__) if self._is_chef else None
        self._config._xml_path = self._env.xml_path
        config.nq = self._env.model.nq

        ob_space = self._env.observation_space
        ac_space = self._env.action_space
        joint_space = self._env.joint_space

        # get actor and critic networks
        actor, critic = get_actor_critic_by_name(config.policy, config.use_ae)

        # build up networks
        non_limited_idx = np.where(self._env.model.jnt_limited[:action_size(self._env.action_space)]==0)[0]

        if config.subgoal_type == 'joint':
            meta_ac_space = joint_space
        else:
            meta_ac_space = spaces.Dict({'default': spaces.Box(shape=(2,), low=-0.5, high=0.5)})

        sampler = None
        if config.her:
            def reward_func(ag, g, info):
                return self._env.her_compute_reward(ag, g, info)
            sampler = HERSampler(config.replay_strategy,
                                 config.replay_k,
                                 reward_func)

        if config.meta_algo == 'ppo':
            self._meta_agent = MetaPPOAgent(config, ob_space, meta_ac_space, sampler=sampler)
        elif config.meta_algo == 'sac':
            self._meta_agent = MetaSACAgent(config, ob_space, meta_ac_space, sampler=sampler)
        else:
            raise NotImplementedError

        ll_ob_space = ob_space

        if config.ll_type == 'mp':
            config.primitive_skills = ['mp']

        if config.hrl:
            mp = None
            from rl.low_level_agent import LowLevelAgent
            if config.ll_type == 'mix' or config.ll_type == 'mp':
                from rl.mp_agent import MpAgent
                mp = MpAgent(config, ac_space, non_limited_idx)
            self._agent = LowLevelAgent(
                config, ll_ob_space, ac_space, actor, critic, mp
            )
        else:
            self._agent = get_agent_by_name(config.algo, config.use_ae)(
                config, ob_space, ac_space, actor, critic
            )


        # build rollout runner
        self._runner = RolloutRunner(
            config, self._env, self._env_eval, self._meta_agent, self._agent
        )

        # setup wandb
        if self._is_chef and self._config.is_train:
            exclude = ["device"]
            if config.debug:
                os.environ["WANDB_MODE"] = "dryrun"

            tags = [config.env, config.hl_type, config.ll_type, config.policy, config.algo]
            if config.hrl:
                tags.append('hrl')

            wandb.init(
                resume=config.run_name,
                project="hrl-planner",
                config={k: v for k, v in config.__dict__.items() if k not in exclude},
                dir=config.log_dir,
                entity="clvr",
                notes=config.notes,
                tags=tags,
                group=config.group
            )

    def _save_ckpt(self, ckpt_num, update_iter):
        ckpt_path = os.path.join(self._config.log_dir, "ckpt_%08d.pt" % ckpt_num)
        state_dict = {"step": ckpt_num, "update_iter": update_iter}
        state_dict["meta_agent"] = self._meta_agent.state_dict()
        state_dict["agent"] = self._agent.state_dict()
        torch.save(state_dict, ckpt_path)
        logger.warn("Save checkpoint: %s", ckpt_path)

        #if self._config.policy == 'mlp' or not self._config.use_ae:
        replay_path = os.path.join(self._config.log_dir, "replay_%08d.pkl" % ckpt_num)
        with gzip.open(replay_path, "wb") as f:
            if self._config.hrl:
                if self._config.hrl_network_to_update == "HL":
                    replay_buffers = {"replay": self._meta_agent.replay_buffer()}
                elif self._config.hrl_network_to_update == "LL":
                    replay_buffers = {"replay": self._agent.replay_buffer()}
                else: # both
                    replay_buffers = {"hl_replay": self._meta_agent.replay_buffer(),
                                      "ll_replay": self._agent.replay_buffer()}
            else:
                replay_buffers = {"replay": self._agent.replay_buffer()}
            if self._config.policy == 'cnn' or self._config.use_ae:
                joblib.dump(replay_buffers, f)
            else:
                pickle.dump(replay_buffers, f)
            #joblib.dump(replay_buffers, f)

    def _load_ckpt(self, ckpt_num=None):
        ckpt_path, ckpt_num = get_ckpt_path(self._config.log_dir, ckpt_num)

        if ckpt_path is not None:
            logger.warn("Load checkpoint %s", ckpt_path)
            ckpt = torch.load(ckpt_path)
            self._meta_agent.load_state_dict(ckpt["meta_agent"])
            self._agent.load_state_dict(ckpt["agent"])

            if self._config.is_train:
                #if self._config.policy == 'mlp' or not self._config.use_ae:
                replay_path = os.path.join(self._config.log_dir, "replay_%08d.pkl" % ckpt_num)
                logger.warn("Load replay_buffer %s", replay_path)
                with gzip.open(replay_path, "rb") as f:
                    replay_buffers = pickle.load(f)
                    #replay_buffers = joblib.load(f)
                    if self._config.hrl:
                        if self._config.hrl_network_to_update == "HL":
                            self._meta_agent.load_replay_buffer(replay_buffers["replay"])
                        elif self._config.hrl_network_to_update == "LL":
                            self._agent.load_replay_buffer(replay_buffers["replay"])
                        else: # both
                            self._meta_agent.load_replay_buffer(replay_buffers["hl_replay"])
                            self._agent.load_replay_buffer(replay_buffers["ll_replay"])
                    else:
                        self._agent.load_replay_buffer(replay_buffers["replay"])

            return ckpt["step"], ckpt["update_iter"]
        else:
            logger.warn("Randomly initialize models")
            return 0, 0

    def _log_train(self, step, train_info, ep_info, prefix=""):
        for k, v in train_info.items():
            if np.isscalar(v) or (hasattr(v, "shape") and np.prod(v.shape) == 1):
                wandb.log({"train_rl/%s" % k: v}, step=step)
            elif isinstance(v, np.ndarray) or isinstance(v, list):
                wandb.log({"train_rl/%s" % k: wandb.Histogram(v)}, step=step)
            else:
                wandb.log({"train_rl/%s" % k: [wandb.Image(v)]}, step=step)

        for k, v in ep_info.items():
            wandb.log({prefix+"train_ep/%s" % k: np.mean(v)}, step=step)
            wandb.log({prefix+"train_ep_max/%s" % k: np.max(v)}, step=step)

    def _log_test(self, step, ep_info, vids=None, obs=None):
        if self._config.is_train:
            for k, v in ep_info.items():
                wandb.log({"test_ep/%s" % k: np.mean(v)}, step=step)
            if vids is not None:
                self.log_videos(vids.transpose((0, 1, 4, 2, 3)), 'test_ep/video', step=step)
            if obs is not None:
                self.log_obs(obs, 'test_ep/obs', step=step)

    def _log_mp_test(self, step, ep_info, vids=None, obs=None):
        if self._config.is_train:
            for k, v in ep_info.items():
                wandb.log({"test_mp_ep/%s" % k: np.mean(v)}, step=step)
            if vids is not None:
                self.log_videos(vids.transpose((0, 1, 4, 2, 3)), 'test_ep/mp_video', step=step)
            if obs is not None:
                self.log_obs(obs, 'test_ep/mp_obs', step=step)

    def train(self):
        config = self._config
        num_batches = config.num_batches

        # load checkpoint
        step, update_iter = self._load_ckpt()

        # sync the networks across the cpus
        self._agent.sync_networks()

        logger.info("Start training at step=%d", step)
        if self._is_chef:
            pbar = tqdm(initial=step, total=config.max_global_step, desc=config.run_name)
            ep_info = defaultdict(list)

        # dummy run for preventing weird
        runner = None
        random_runner = None
        if config.hrl:
            if config.hrl_network_to_update == "LL":
                runner = self._runner.run(every_steps=1)
                random_runner = self._runner.run(every_steps=1, random_exploration=True)
            elif config.hrl_network_to_update == 'HL':
                if config.ll_type == 'mp' or config.ll_type == 'mix': # would required to modify this later
                    if config.meta_algo == 'sac':
                        runner = self._runner.run_with_mp(every_steps=1)
                        random_runner = self._runner.run_with_mp(every_steps=1, random_exploration=True)
                    else:
                        self._runner.run_episode_with_mp(every_steps=self._config.rollout_length)
                elif config.ll_type == 'rl':
                    runner = self._runner.run(every_steps=1)
            else:
                raise NotImplementedError
        else:
            if config.algo == 'sac':
                runner = self._runner.run(every_steps=1)
                random_runner = self._runner.run(every_steps=1, random_exploration=True)
            elif config.algp == 'ppo':
                runner = self._runner.run(every_steps=self._config.rollout_length)

        st_time = time()
        st_step = step
        global_run_ep = 0

        init_step = 0
        init_ep = 0

        # If it does not previously learned data and use SAC, then we firstly fill the experieince replay with the specified number of samples
        if step == 0:
            if random_runner is not None:
                while init_step < self._config.start_steps:
                    rollout, meta_rollout, info = next(random_runner)
                    step_per_batch = mpi_sum(len(rollout['ac']))
                    init_step += step_per_batch

                    if (config.hrl_network_to_update == "HL" or \
                        config.hrl_network_to_update == "both"):
                        self._meta_agent.store_episode(meta_rollout)
                    if (config.hrl_network_to_update == "LL" or \
                        config.hrl_network_to_update == "both"):
                        self._agent.store_episode(rollout)

        while step < config.max_global_step:
            # collect rollouts
            rollout, meta_rollout, info = next(runner)
            if config.hrl:
                if (config.hrl_network_to_update == "HL" or \
                    config.hrl_network_to_update == "both"):
                    self._meta_agent.store_episode(meta_rollout)
                if (config.hrl_network_to_update == "LL" or \
                    config.hrl_network_to_update == "both"):
                    self._agent.store_episode(rollout)
            else:
                self._agent.store_episode(rollout)

            step_per_batch = mpi_sum(len(rollout['ac']))

            # train an agent
            logger.info("Update networks %d", update_iter)
            if config.hrl:
                if (config.hrl_network_to_update == "HL" or \
                    config.hrl_network_to_update == "both"):
                    train_info = self._meta_agent.train()
                    hl_train_info = train_info
                else:
                    hl_train_info = None
                if (config.hrl_network_to_update == "LL" or \
                    config.hrl_network_to_update == "both"):
                    train_info = self._agent.train()
                    ll_train_info = train_info
                else:
                    ll_train_info = None
                if config.hrl_network_to_update == "both":
                    train_info = {}
                    train_info.update({k + "_hl": v for k, v in hl_train_info.items()})
                    train_info.update({k + "_ll": v for k, v in ll_train_info.items()})
            else:
                train_info = self._agent.train()

            logger.info("Update networks done")

            # if step < config.max_ob_norm_step and self._config.policy != 'cnn':
            #     self._update_normalizer(rollout, meta_rollout)

            step += step_per_batch
            update_iter += 1

            if self._is_chef:
                pbar.update(step_per_batch)

                if update_iter % config.log_interval == 0:
                    for k, v in info.items():
                        if isinstance(v, list):
                            ep_info[k].extend(v)
                        else:
                            ep_info[k].append(v)
                    train_info.update({
                        "sec": (time() - st_time) / config.log_interval,
                        "steps_per_sec": (step - st_step) / (time() - st_time),
                        "update_iter": update_iter
                    })
                    st_time = time()
                    st_step = step
                    self._log_train(step, train_info, ep_info)
                    ep_info = defaultdict(list)

                ## Evaluate both MP and RL
                if update_iter % config.evaluate_interval == 1:
                    logger.info("Evaluate at %d", update_iter)
                    rollout, info, vids = self._evaluate(step=step, record=config.record)
                    obs = None
                    if self._config.policy == 'cnn':
                        if self._config.is_rgb:
                            obs = rollout['ob'][0]['default'].transpose((1, 2, 0))
                        else:
                            obs = rollout['ob'][0]['default'][0]

                        if self._config.use_ae:
                            _to_tensor = lambda x: to_tensor(x, self._config.device)
                            h = self._agent._critic_encoder(_to_tensor(rollout['ob'][0]['default']).unsqueeze(0))
                            recon = self._agent._decoder(h)[0].detach().cpu().numpy().transpose((1, 2, 0))
                            self.log_obs(recon, 'test_ep/reconstructed', step=step)
                    self._log_test(step, info, vids, obs)

                    # Evaluate mp
                    if self._config.ll_type == 'mp' or self._config.ll_type == 'mix':
                        mp_rollout, mp_info, mp_vids = self._mp_evaluate(step=step, record=config.record)
                        self._log_mp_test(step, mp_info, mp_vids)


                if update_iter % config.ckpt_interval == 0:
                    self._save_ckpt(step, update_iter)

        logger.info("Reached %s steps. worker %d stopped.", step, config.rank)

    def _update_normalizer(self, rollout, meta_rollout):
        if self._config.ob_norm:
            self._meta_agent.update_normalizer(meta_rollout["ob"])
            self._agent.update_normalizer(rollout["ob"])

    def _save_success_qpos(self, info, prefix=""):
        if self._config.save_qpos and info["episode_success"]:
            path = os.path.join(self._config.record_dir, prefix+"qpos.p")
            with h5py.File(path, "a") as f:
                key_id = len(f.keys())
                num_qpos = len(info["saved_qpos"])
                for qpos_to_save in info["saved_qpos"]:
                    f["{}".format(key_id)] = qpos_to_save
                    key_id += 1
        if self._config.save_success_qpos and info["episode_success"]:
            path = os.path.join(self._config.record_dir, prefix+"success_qpos.p")
            with h5py.File(path, "a") as f:
                key_id = len(f.keys())
                num_qpos = len(info["saved_qpos"])
                for qpos_to_save in info["saved_qpos"][int(num_qpos / 2):]:
                    f["{}".format(key_id)] = qpos_to_save
                    key_id += 1

    def _evaluate(self, step=None, record=False, idx=None):
        """ Run one rollout if in eval mode
            Run num_record_samples rollouts if in train mode
        """
        vids = []
        for i in range(self._config.num_record_samples):
            rollout, meta_rollout, info, frames = \
                self._runner.run_episode(is_train=False, record=record)

            if record:
                ep_rew = info["rew"]
                ep_success = "s" if info["episode_success"] else "f"
                fname = "{}_step_{:011d}_{}_r_{}_{}.mp4".format(
                    self._config.env, step, idx if idx is not None else i,
                    ep_rew, ep_success)
                self._save_video(fname, frames)
                vids.append(frames)

            if idx is not None:
                break

        logger.info("rollout: %s", {k: v for k, v in info.items() if not "qpos" in k})
        # self._save_success_qpos(info)
        return rollout, info, np.array(vids)

    def _mp_evaluate(self, step=None, record=False, idx=None):
        """ Run one rollout if in eval mode
            Run num_record_samples rollouts if in train mode
            Evaluation for Motion Planner
        """
        vids = []
        for i in range(self._config.num_record_samples):
            rollout, meta_rollout, info, frames = \
                self._runner.run_episode_with_mp(is_train=False, record=record)

            if record:
                ep_rew = info["rew"]
                ep_success = "s" if info["episode_success"] else "f"
                fname = "{}_step_{:011d}_{}_r_{}_{}_mp.mp4".format(
                    self._config.env, step, idx if idx is not None else i,
                    ep_rew, ep_success)
                self._save_video(fname, frames)
                vids.append(frames)

            if idx is not None:
                break

        logger.info("mp rollout: %s", {k: v for k, v in info.items() if not "qpos" in k})
        # self._save_success_qpos(info, prefix="mp_")
        return rollout, info, np.array(vids)


    def evaluate(self):
        step, update_iter = self._load_ckpt(ckpt_num=self._config.ckpt_num)

        logger.info("Run %d evaluations at step=%d, update_iter=%d",
                    self._config.num_eval, step, update_iter)
        info_history = defaultdict(list)
        rollouts = []
        for i in trange(self._config.num_eval):
            logger.warn("Evalute run %d", i+1)
            rollout, info, vids = \
                self._evaluate(step=step, record=self._config.record, idx=i)
            for k, v in info.items():
                info_history[k].append(v)
            if self._config.save_rollout:
                rollouts.append(rollout)

        keys = ["episode_success", "reward_goal_dist"]
        os.makedirs("result", exist_ok=True)
        with h5py.File("result/{}.hdf5".format(self._config.run_name), "w") as hf:
            for k in keys:
                hf.create_dataset(k, data=info_history[k])

            result = "{:.02f} $\\pm$ {:.02f}".format(
                    np.mean(info_history["episode_success"]),
                    np.std(info_history["episode_success"])
            )
            logger.warn(result)

        if self._config.save_rollout:
            os.makedirs("saved_rollouts", exist_ok=True)
            with open("saved_rollouts/{}.p".format(self._config.run_name), "wb") as f:
                pickle.dump(rollouts, f)

    def _save_video(self, fname, frames, fps=8.):
        path = os.path.join(self._config.record_dir, fname)

        def f(t):
            frame_length = len(frames)
            new_fps = 1./(1./fps + 1./frame_length)
            idx = min(int(t*new_fps), frame_length-1)
            return frames[idx]

        video = mpy.VideoClip(f, duration=len(frames)/fps+2)

        video.write_videofile(path, fps, verbose=False, logger=None)
        logger.warn("[*] Video saved: {}".format(path))


    def log_videos(self, vids, name, fps=15, step=None):
        """Logs videos to WandB in mp4 format.
        Assumes list of numpy arrays as input with [time, channels, height, width]."""
        assert len(vids[0].shape) == 4 and vids[0].shape[1] == 3
        assert isinstance(vids[0], np.ndarray)
        log_dict = {name: [wandb.Video(vid, fps=fps, format="mp4") for vid in vids]}
        wandb.log(log_dict) if step is None else wandb.log(log_dict, step=step)

    def log_obs(self, obs, name, step=None):
        if self._config.is_rgb:
            log_dict = {name: [wandb.Image(obs)]}
        else:
            log_dict = {name: [wandb.Image(obs, mode='L')]}
        wandb.log(log_dict) if step is None else wandb.log(log_dict, step=step)

