import sys
import signal
import os
import json

import numpy as np
import torch
from six.moves import shlex_quote
from mpi4py import MPI
from logging import CRITICAL

from config import argparser
from config.motion_planner import add_arguments as mp_add_arguments
from rl.trainer import Trainer
from util.logger import logger
from datetime import datetime

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

def run(config):
    rank = MPI.COMM_WORLD.Get_rank()
    config.rank = rank
    config.is_chef = rank == 0
    config.seed = config.seed + rank
    config.num_workers = MPI.COMM_WORLD.Get_size()
    config.is_mpi = False if config.num_workers == 1 else True

    if torch.get_num_threads() != 1:
        fair_num_threads = max(int(torch.get_num_threads() / MPI.COMM_WORLD.Get_size()), 1)
        torch.set_num_threads(fair_num_threads)

    if config.is_chef:
        logger.warning('Running a base worker.')
        make_log_files(config)
    else:
        logger.warning('Running worker %d and disabling logger', config.rank)
        logger.setLevel(CRITICAL)


        if config.date is None:
            now = datetime.now()
            date = now.strftime("%m.%d")
        else:
            date = config.date
        config.run_name = 'rl.{}.{}.{}.{}'.format(config.env, date, config.prefix, config.seed-rank)
        if config.group is None:
            config.group = 'rl.{}.{}.{}'.format(config.env, date, config.prefix)

        config.log_dir = os.path.join(config.log_root_dir, config.run_name)
        if config.is_train:
            config.record_dir = os.path.join(config.log_dir, 'video')
        else:
            config.record_dir = os.path.join(config.log_dir, 'eval_video')

    def shutdown(signal, frame):
        logger.warning('Received signal %s: exiting', signal)
        sys.exit(128+signal)


    signal.signal(signal.SIGHUP, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # set global seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    os.environ["DISPLAY"] = ":1"

    if config.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(config.gpu)
        assert torch.cuda.is_available()
        config.device = torch.device("cuda")
    else:
        config.device = torch.device("cpu")

    # build a trainer
    trainer = Trainer(config)
    if config.is_train:
        trainer.train()
        logger.info("Finish training")
    else:
        if config.ll_type == 'mix' and config.subgoal_predictor:
            trainer.mp_evaluate()
            logger.info('Finish evaluating')
        else:
            trainer.evaluate()
            logger.info("Finish evaluating")


def make_log_files(config):
    if config.date is None:
        now = datetime.now()
        date = now.strftime("%m.%d")
    else:
        date = config.date
    # date = '07.25'
    config.run_name = 'rl.{}.{}.{}.{}'.format(config.env, date, config.prefix, config.seed)
    if config.group is None:
        config.group = 'rl.{}.{}.{}'.format(config.env, date, config.prefix)

    config.log_dir = os.path.join(config.log_root_dir, config.run_name)
    logger.info('Create log directory: %s', config.log_dir)
    os.makedirs(config.log_dir, exist_ok=True)

    if config.is_train:
        config.record_dir = os.path.join(config.log_dir, 'video')
    else:
        config.record_dir = os.path.join(config.log_dir, 'eval_video')
    logger.info('Create video directory: %s', config.record_dir)
    os.makedirs(config.record_dir, exist_ok=True)

    if config.primitive_dir is None:
        config.primitive_dir = config.log_root_dir

    if config.is_train:
        # log git diff
        cmds = [
            "echo `git rev-parse HEAD` >> {}/git.txt".format(config.log_dir),
            "git diff >> {}/git.txt".format(config.log_dir),
            "echo 'python -m rl.main {}' >> {}/cmd.sh".format(
                ' '.join([shlex_quote(arg) for arg in sys.argv[1:]]),
                config.log_dir),
        ]
        os.system("\n".join(cmds))

        # log config
        param_path = os.path.join(config.log_dir, 'params.json')
        logger.info('Store parameters in %s', param_path)
        with open(param_path, 'w') as fp:
            json.dump(config.__dict__, fp, indent=4, sort_keys=True)


if __name__ == '__main__':
    parser = argparser()
    args, unparsed = parser.parse_known_args()

    if 'pusher' in args.env:
        from config.pusher import add_arguments
    elif 'sawyer' in args.env:
        from config.sawyer import add_arguments
    else:
        raise ValueError('args.env (%s) is not supported' % args.env)



    add_arguments(parser)
    mp_add_arguments(parser)
    args, unparsed = parser.parse_known_args()


    if args.debug:
        args.rollout_length = 150
        args.start_steps = 100

    if len(unparsed):
        logger.error('Unparsed argument is detected:\n%s', unparsed)
    else:
        run(args)

