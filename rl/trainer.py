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
import matplotlib.pyplot as plt

from rl.policies import get_actor_critic_by_name
from rl.rollouts import RolloutRunner
from rl.subgoal_rollouts import SubgoalRolloutRunner
from rl.subgoal_ppo_rollouts import SubgoalPPORolloutRunner
from rl.planner_rollouts import PlannerRolloutRunner
from rl.dataset import HERSampler
from util.logger import logger
from util.pytorch import get_ckpt_path, count_parameters, to_tensor
from util.mpi import mpi_sum
from util.gym import observation_size, action_size
from util.misc import make_ordered_pair



def get_agent_by_name(algo):
    if algo == "sac":
        from rl.sac_agent import SACAgent
        return SACAgent
    elif algo == "ppo":
        from rl.ppo_agent import PPOAgent
        return PPOAgent

def get_meta_agent_by_name(algo):
    if algo == 'ppo':
        from rl.meta_ppo_agent import MetaPPOAgent
        return MetaPPOAgent
    elif algo == 'sac':
        from rl.meta_sac_agent import MetaSACAgent
        return MetaSACAgent
    else:
        raise NotImplementedError

class Trainer(object):
    def __init__(self, config):
        self._config = config
        self._is_chef = config.is_chef

        # create a new environment
        self._env = gym.make(config.env, **config.__dict__)
        self._env_eval = gym.make(config.env, **copy.copy(config).__dict__) if self._is_chef else None
        self._config._xml_path = self._env.xml_path
        config.nq = self._env.sim.model.nq

        ob_space = self._env.observation_space
        ac_space = self._env.action_space
        # if config.planner_integration:
        #     if config.relative_goal:
        #         ac_space['default'].high = np.ones_like(ac_space['default'].high) * config.action_range
        #         ac_space['default'].low = -np.ones_like(ac_space['default'].low) * config.action_range
        #     else:
        #         ac_space['default'].high = self._env._jnt_maximum[self._env.ref_joint_pos_indexes]
        #         ac_space['default'].low = self._env._jnt_minimum[self._env.ref_joint_pos_indexes]
        joint_space = self._env.joint_space

        allowed_collsion_pairs = []
        geom_ids = self._env.agent_geom_ids + self._env.static_geom_ids
        if config.allow_self_collision:
            from itertools import combinations
            comb = combinations(geom_ids, 2)
            for pair in list(comb):
                allowed_collsion_pairs.append(make_ordered_pair(pair[0], pair[1]))

        if config.allow_manipulation_collision:
            for manipulation_geom_id in self._env.manipulation_geom_ids:
                for geom_id in geom_ids:
                    allowed_collsion_pairs.append(make_ordered_pair(manipulation_geom_id, geom_id))

        if config.ignored_contact_geoms is not None:
            ids = []
            for i, geom in enumerate(config.ignored_contact_geoms):
                ids.append([])
                geom = geom.split("/")
                geom_pairs = []
                for g in geom:
                    pair = g.split(",")
                    pair_id = []
                    for p in pair:
                        if p != 'None':
                            pair_id.append(self._env.sim.model.geom_name2id(p))
                    # geom_pairs.append(pair_id)
                    if len(pair_id) != 0:
                        ids[i].append(make_ordered_pair(pair_id[0], pair_id[1]))
                    if len(allowed_collsion_pairs) != 0:
                        ids[i].extend(allowed_collsion_pairs)
            config.ignored_contact_geom_ids = ids

        passive_joint_idx = list(range(len(self._env.sim.data.qpos)))
        [passive_joint_idx.remove(idx) for idx in self._env.ref_joint_pos_indexes]
        config.passive_joint_idx = passive_joint_idx


        # get actor and critic networks
        actor, critic = get_actor_critic_by_name(config.policy)

        # build up networks
        non_limited_idx = np.where(self._env.sim.model.jnt_limited[:action_size(self._env.action_space)]==0)[0]
        meta_ac_space = joint_space

        sampler = None
        if config.her:
            def reward_func(ag, g, info):
                return self._env.her_compute_reward(ag, g, info)
            sampler = HERSampler(config.replay_strategy,
                                 config.replay_k,
                                 reward_func)

        self._meta_agent = get_meta_agent_by_name(config.meta_algo)(config, ob_space, meta_ac_space, sampler=sampler)

        ll_ob_space = ob_space
        if config.planner_integration and config.extended_action:
            ac_space.spaces['ac_type'] = spaces.Discrete(2)
        if config.hrl:
            if config.use_subgoal_space:
                if config.relative_goal:
                    subgoal_space = self._env.subgoal_space
                else:
                    subgoal_space = spaces.Dict({'default': spaces.Box(low=self._env._jnt_minimum[self._env.ref_joint_pos_indexes],
                                               high=self._env._jnt_maximum[self._env.ref_joint_pos_indexes])})
            else:
                subgoal_space = ac_space

            if config.termination:
                subgoal_space.spaces['term'] = spaces.Discrete(2)
                ac_space.spaces['term'] = spaces.Discrete(2)
            if config.algo == 'sac':
                from rl.low_level_agent import LowLevelAgent
                self._agent = LowLevelAgent(
                    config, ll_ob_space, ac_space, actor, critic,
                    non_limited_idx, subgoal_space,
                )
            else:
                from rl.low_level_ppo_agent import LowLevelPPOAgent
                self._agent = LowLevelPPOAgent(
                    config, ll_ob_space, ac_space, subgoal_space, actor, critic, non_limited_idx
                )

        else:
            self._agent = get_agent_by_name(config.algo)(
                config, ob_space, ac_space, actor, critic, non_limited_idx, self._env.ref_joint_pos_indexes, self._env.joint_space, self._env._is_jnt_limited
            )

        self._runner = None
        if config.planner_integration:
            self._runner = PlannerRolloutRunner(
                config, self._env, self._env_eval, self._meta_agent, self._agent
            )
        elif config.hrl:
            if config.subgoal_predictor:
                if config.algo == 'sac':
                    self._runner = SubgoalRolloutRunner(
                        config, self._env, self._env_eval, self._meta_agent, self._agent
                    )
                else:
                    self._runner = SubgoalPPORolloutRunner(
                        config, self._env, self._env_eval, self._meta_agent, self._agent
                    )
            else:
                if config.ll_type == 'rl':
                    # build rollout runner
                    self._runner = RolloutRunner(
                        config, self._env, self._env_eval, self._meta_agent, self._agent
                    )
        else:
            self._runner = RolloutRunner(
                config, self._env, self._env_eval, self._meta_agent, self._agent
            )


        # setup wandb
        if self._is_chef and self._config.is_train:
            exclude = ["device"]
            if config.debug:
                os.environ["WANDB_MODE"] = "dryrun"

            tags = [config.env, config.hl_type, config.ll_type, config.policy, config.algo, config.reward_type]
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

        replay_path = os.path.join(self._config.log_dir, "replay_%08d.pkl" % ckpt_num)
        with gzip.open(replay_path, "wb") as f:
            if self._config.hrl:
                if self._config.meta_update_target == "HL":
                    replay_buffers = {"replay": self._meta_agent.replay_buffer()}
                elif self._config.meta_update_target == "LL":
                    replay_buffers = {"replay": self._agent.replay_buffer()}
                else: # both
                    if not self._config.meta_oracle:
                        replay_buffers = {"hl_replay": self._meta_agent.replay_buffer(),
                                          "ll_replay": self._agent.replay_buffer()}
                    else:
                        replay_buffers = {"replay": self._agent.replay_buffer()}

            else:
                replay_buffers = {"replay": self._agent.replay_buffer()}
            if self._config.policy == 'cnn':
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
                replay_path = os.path.join(self._config.log_dir, "replay_%08d.pkl" % ckpt_num)
                logger.warn("Load replay_buffer %s", replay_path)
                with gzip.open(replay_path, "rb") as f:
                    replay_buffers = pickle.load(f)
                    #replay_buffers = joblib.load(f)
                    if self._config.hrl:
                        if self._config.meta_update_target == "HL":
                            self._meta_agent.load_replay_buffer(replay_buffers["replay"])
                        elif self._config.meta_update_target == "LL":
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

    def _log_train(self, step, train_info, ep_info, prefix="", env_step=None):
        for k, v in train_info.items():
            if np.isscalar(v) or (hasattr(v, "shape") and np.prod(v.shape) == 1):
                wandb.log({"train_rl/%s" % k: v}, step=step)
                if env_step is not None and self._config.planner_integration:
                    wandb.log({"train_rl/env_step/%s" % k: v}, step=env_step)
            elif isinstance(v, np.ndarray) or isinstance(v, list):
                wandb.log({"train_rl/%s" % k: wandb.Histogram(v)}, step=step)
                if env_step is not None and self._config.planner_integration:
                    wandb.log({"train_rl/env_step/%s" % k: wandb.Histogram(v)}, step=env_step)
            else:
                wandb.log({"train_rl/%s" % k: [wandb.Image(v)]}, step=step)
                if env_step is not None and self._config.planner_integration:
                    wandb.log({"train_rl/env_step/%s" % k: [wandb.Image(v)]}, step=env_step)

        for k, v in ep_info.items():
            wandb.log({prefix+"train_ep/%s" % k: np.mean(v)}, step=step)
            wandb.log({prefix+"train_ep_max/%s" % k: np.max(v)}, step=step)
            if env_step is not None and self._config.planner_integration:
                wandb.log({prefix+"train_ep/env_step/%s" % k: np.mean(v)}, step=env_step)
                wandb.log({prefix+"train_ep_max/env_step/%s" % k: np.max(v)}, step=env_step)
        if self._config.vis_replay:
            if step % self._config.vis_replay_interval == 0:
                self._vis_replay_buffer(step)

    def _log_test(self, step, ep_info, vids=None, obs=None, env_step=None):
        if self._config.is_train:
            for k, v in ep_info.items():
                wandb.log({"test_ep/%s" % k: np.mean(v)}, step=step)
                if env_step is not None and self._config.planner_integration:
                    wandb.log({"test_ep/env_step/%s" % k: np.mean(v)}, step=env_step)
            if vids is not None:
                self.log_videos(vids.transpose((0, 1, 4, 2, 3)), 'test_ep/video', step=step)
            if obs is not None:
                self.log_obs(obs, 'test_ep/obs', step=step)

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
            if config.meta_update_target == 'HL':
                if config.meta_algo == 'sac':
                    runner = self._runner.run(every_steps=1)
                    random_runner = self._runner.run(every_steps=1, random_exploration=True)
                else:
                    runner = self._runner.run_episode(every_steps=self._config.rollout_length)
            else:
                if config.algo == 'sac':
                    runner = self._runner.run(every_steps=1)
                    random_runner = self._runner.run(every_steps=1, random_exploration=True)
                else:
                    runner = self._runner.run(every_steps=self._config.rollout_length)

        else:
            if config.algo == 'sac':
                runner = self._runner.run(every_steps=1)
                random_runner = self._runner.run(every_steps=1, random_exploration=True)
            elif config.algo == 'ppo':
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
                    if config.is_mpi:
                        step_per_batch = mpi_sum(len(rollout['ac']))
                    else:
                        step_per_batch = len(rollout['ac'])
                    init_step += step_per_batch

                    if config.hrl:
                        if (config.meta_update_target == "HL" or \
                            config.meta_update_target == "both") and not config.meta_oracle:
                            if len(meta_rollout['ob']) != 0:
                                self._meta_agent.store_episode(meta_rollout)
                        if (config.meta_update_target == "LL" or \
                            config.meta_update_target == "both"):
                            if len(rollout['ob']) != 0:
                                self._agent.store_episode(rollout)
                    else:
                        self._agent.store_episode(rollout)

        env_step = 0
        while step < config.max_global_step:
            # collect rollouts
            env_step_per_batch = None
            rollout, meta_rollout, info = next(runner)
            if config.hrl:
                if (config.meta_update_target == "HL" or \
                    config.meta_update_target == "both") and not config.meta_oracle:
                    if len(meta_rollout['ob']) != 0:
                        self._meta_agent.store_episode(meta_rollout)
                if (config.meta_update_target == "LL" or \
                    config.meta_update_target == "both"):
                    if len(rollout['ob']) != 0:
                        self._agent.store_episode(rollout)
            else:
                self._agent.store_episode(rollout)

            if config.is_mpi:
                step_per_batch = mpi_sum(len(rollout['ac']))
            else:
                step_per_batch = len(rollout['ac'])

            if self._config.planner_integration and 'env_step' in info.keys():
                env_step_per_batch = len(info['env_step'])

            # train an agent
            if step % config.log_freq == 0:
                logger.info("Update networks %d", update_iter)
            if config.hrl:
                if (config.meta_update_target == "HL" or \
                    config.meta_update_target == "both") and not config.meta_oracle:
                    train_info = self._meta_agent.train()
                    hl_train_info = train_info
                else:
                    hl_train_info = None
                if (config.meta_update_target == "LL" or \
                    config.meta_update_target == "both"):
                    train_info = self._agent.train()
                    ll_train_info = train_info
                else:
                    ll_train_info = None
                if config.meta_update_target == "both":
                    train_info = {}
                    train_info.update({k + "_hl": v for k, v in hl_train_info.items()})
                    train_info.update({k + "_ll": v for k, v in ll_train_info.items()})
            else:
                train_info = self._agent.train()

            if step % config.log_freq == 0:
                logger.info("Update networks done")

            # if step < config.max_ob_norm_step and self._config.policy != 'cnn':
            #     self._update_normalizer(rollout, meta_rollout)
            step += step_per_batch

            if env_step_per_batch is not None:
                env_step += env_step_per_batch
            update_iter += 1

            if self._is_chef:
                pbar.update(step_per_batch)
                if update_iter % config.log_interval == 0 or len(info) != 0:
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
                    self._log_train(step, train_info, ep_info, env_step=env_step)
                    ep_info = defaultdict(list)

                ## Evaluate both MP and RL
                if update_iter % config.evaluate_interval == 0:
                    logger.info("Evaluate at %d", update_iter)
                    obs = None
                    rollout, info, vids = self._evaluate(step=step, record=config.record)
                    if self._config.policy == 'cnn':
                        if self._config.is_rgb:
                            obs = rollout['ob'][0]['default'].transpose((1, 2, 0))
                        else:
                            obs = rollout['ob'][0]['default'][0]

                    self._log_test(step, info, vids, obs, env_step=env_step)

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

    def _vis_replay_buffer(self, step):
        if step > self._agent._buffer._size:
            return # visualization does not work if ealier samples were overriden

        size = self._agent._buffer._current_size
        states = np.array([ob[1]['fingertip'] for ob in self._agent._buffer.state_dict()['ob']])
        fig = plt.figure()
        # plt.scatter(states[:, 0], states[:, 1], s=5, c=np.arange(size)[:1000000], cmap='Blues')
        plt.scatter(states[:, 0], states[:, 1], s=5, c=np.arange(500000), cmap='Blues')
        plt.axis("equal")
        wandb.log({'replay_vis': wandb.Image(fig)}, step=step)
        plt.close(fig)



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

