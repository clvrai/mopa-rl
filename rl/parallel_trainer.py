import os
from time import time
from collections import defaultdict, OrderedDict
import gzip
import pickle
import h5py

import torch
import torch.multiprocessing as mp
import wandb
import numpy as np
import moviepy.editor as mpy
from tqdm import tqdm, trange
import env
import gym
from gym import spaces

from rl.policies import get_actor_critic_by_name
from rl.meta_ppo_agent import MetaPPOAgent
from rl.rollouts import RolloutRunner
from util.logger import logger
from util.pytorch import get_ckpt_path, count_parameters, to_tensor
from util.mpi import mpi_sum
from util.gym import observation_size


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


class ParallelController(object):
    def __init__(self, config):
        self._config = config
        self._is_chef = config.is_chef

        # create a new environment
        self._env = gym.make(config.env, **config.__dict__)
        self._config._xml_path = self._env.xml_path
        config.nq = self._env.model.nq

        ob_space = self._env.observation_space
        ac_space = self._env.action_space
        joint_space = self._env.joint_sapce

        # get actor and critic networks
        actor, critic = get_actor_critic_by_name(config.policy, config.use_ae)

        # build up networks
        self._meta_agent = MetaPPOAgent(config, ob_space, ac_space)
        self._mp = None

        if config.hl_type == 'subgoal':
            # use subgoal
            if config.policy == 'cnn':
                ll_ob_space = spaces.Dict({'default': ob_space['default'], 'subgoal': ac_space['default']})
            elif config.policy == 'mlp':
                ll_ob_space = spaces.Dict({'default': ob_space['default'],
                                           'subgoal': ac_space['default']})
            else:
                raise NotImplementedError
        else:
            # no subgoal, only choose which low-level controler we use
            ll_ob_space = spaces.Dict({'default': ob_space['default']})


        if config.hrl:
            if config.ll_type == 'rl':
                from rl.low_level_agent import LowLevelAgent
                self._agent = LowLevelAgent(
                    config, ll_ob_space, ac_space, actor, critic
                )
            else:
                from rl.low_level_mp_agent import LowLevelMpAgent
                from rl.low_level_agent import LowLevelAgent
                self._agent = LowLevelAgent(
                    config, ll_ob_space, ac_space, actor, critic
                )
                self._mp = LowLevelMpAgent(config, ll_ob_space, ac_space)
        else:
            self._agent = get_agent_by_name(config.algo, config.use_ae)(
                config, ob_space, ac_space, actor, critic
            )

        # build rollout runner
        self._runner = RolloutRunner(
            config, self._env, self._meta_agent, self._agent, self._mp
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
                tags=tags
            )

        self._worker_nums = config.worker_nums
        self.manager = mp.Manager()
        self.start_worker()


    def start_workder(self):
        self._workders = []
        self._shared_que = self.manager.Queue(self._worker_nums)
        self.start_barrier = mp.Barrier(self._workder_nums+1)

        for i in range(self._worker_nums):
            p = mp.Process(
                target=self.__class__.train_worker_process,
                args=(self.__class__)
            )

