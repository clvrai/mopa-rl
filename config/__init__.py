import argparse

from util import str2bool, str2list


def argparser():
    parser = argparse.ArgumentParser(
        "MoPA-RL", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--date", type=str, default=None, help="experiment date")
    # environment
    parser.add_argument(
        "--env", type=str, default="reacher-obstacle-v0", help="environment name"
    )

    # training algorithm
    parser.add_argument("--algo", type=str, default="sac", choices=["sac", "td3"], help="RL algorithm")
    parser.add_argument("--policy", type=str, default="mlp", choices=["mlp"], help="policy type")
    parser.add_argument("--mopa", type=str2bool, default=False, help="enable MoPA")
    parser.add_argument(
        "--ac_space_type",
        type=str,
        default="piecewise",
        choices=["normal", "piecewise"],
        help="Action space type for MoPA",
    )
    parser.add_argument(
        "--use_ik_target",
        type=str2bool,
        default=False,
        help="Enable cartasian action space for inverse kienmatics",
    )
    parser.add_argument(
        "--ik_target",
        type=str,
        default="fingertip",
        help="reference location for inverse kinematics",
    )
    parser.add_argument(
        "--expand_ac_space",
        type=str2bool,
        default=False,
        help="enable larger action space for SAC",
    )

    # vanilla rl
    parser.add_argument("--rl_hid_size", type=int, default=256, help="hidden unit size")
    parser.add_argument(
        "--rl_activation", type=str, default="relu", choices=["relu", "elu", "tanh"],
        help="activation function"
    )
    parser.add_argument("--tanh_policy", type=str2bool, default=True, help="enable tanh policy")
    parser.add_argument("--actor_num_hid_layers", type=int, default=2, help="number of hidden layer in an actor")

    # motion planning
    parser.add_argument(
        "--invalid_target_handling",
        type=str2bool,
        default=False,
        help="enable invalid target handling",
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=100,
        help="number of trials for invalid target handling",
    )
    parser.add_argument("--interpolation", type=str2bool, default=True, help="enable interpolation for motion planner")

    # MoPA
    parser.add_argument(
        "--omega",
        type=float,
        default=1.0,
        help="threshold of direct action execution and motion planner",
    )
    parser.add_argument(
        "--reuse_data", type=str2bool, default=False, help="enable reuse of data"
    )
    parser.add_argument(
        "--max_reuse_data",
        type=int,
        default=30,
        help="maximum number of reused samples",
    )
    parser.add_argument(
        "--action_range", type=float, default=1.0, help="range of radian"
    )
    parser.add_argument(
        "--discrete_action",
        type=str2bool,
        default=False,
        help="enable discrete action to choose motion planner or direct action execution",
    )
    parser.add_argument(
        "--stochastic_eval",
        type=str2bool,
        default=False,
        help="eval an agent with a stochastic policy",
    )

    # off-policy rl
    parser.add_argument(
        "--buffer_size", type=int, default=int(1e6), help="the size of the buffer"
    )
    parser.add_argument(
        "--discount_factor", type=float, default=0.99, help="the discount factor"
    )
    parser.add_argument(
        "--lr_actor", type=float, default=3e-4, help="the learning rate of the actor"
    )
    parser.add_argument(
        "--lr_critic", type=float, default=3e-4, help="the learning rate of the critic"
    )
    parser.add_argument(
        "--lr_alpha", type=float, default=3e-4, help="the learning rate of the alpha"
    )
    parser.add_argument(
        "--polyak", type=float, default=0.995, help="the average coefficient"
    )
    parser.add_argument("--actor_update_freq", type=int, default=1, help="frequency of actor update")
    parser.add_argument("--critic_target_update_freq", type=int, default=1, help="frequency of critic target update")

    # training
    parser.add_argument("--is_train", type=str2bool, default=True, help="enable training")
    parser.add_argument(
        "--num_batches",
        type=int,
        default=1,
        help="the times to update the network per epoch",
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="the sample batch size"
    )
    parser.add_argument("--max_global_step", type=int, default=int(3e6), help="maximum number of time steps")
    parser.add_argument("--gpu", type=int, default=None, help="gpu id")

    # sac
    parser.add_argument(
        "--start_steps",
        type=int,
        default=1e4,
        help="number of samples collected before training",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Temperature for Gumbel Softmax"
    )
    parser.add_argument("--alpha", type=float, default=1.0, help="inverse of reward scale")
    parser.add_argument("--reward_scale", type=float, default=1.0, help="reward scale for SAC")
    parser.add_argument(
        "--use_smdp_update",
        type=str2bool,
        default=False,
        help="update a policy under semi-markov decision process",
    )

    # td3
    parser.add_argument("--target_noise", type=float, default=0.2, help="target noise for TD3")
    parser.add_argument("--action_noise", type=float, default=0.1), help="action noise for TD3")
    parser.add_argument("--noise_clip", type=float, default=0.5, help="noise clip for TD3")

    # log
    parser.add_argument("--log_interval", type=int, default=1000, help="interval for logging")
    parser.add_argument("--vis_replay_interval", type=int, default=10000, help="interval for visualization of replay buffer")
    parser.add_argument("--evaluate_interval", type=int, default=10000, help="interval for evaluation")
    parser.add_argument("--ckpt_interval", type=int, default=200000, help="interval for saving a checkpoint file")
    parser.add_argument("--log_root_dir", type=str, default="log", help="root directory for logging")
    parser.add_argument(
        "--wandb",
        type=str2bool,
        default=False,
        help="set it True if you want to use wandb",
    )
    parser.add_argument(
        "--entity", type=str, default="clvr", help="Set an entity name for wandb"
    )
    parser.add_argument(
        "--project", type=str, default="hrl-planner", help="set a project name for wandb"
    )
    parser.add_argument("--group", type=str, default=None, help="group for wandb")
    parser.add_argument("--vis_replay", type=str2bool, default=True, help="enable visualization of replay buffer")
    parser.add_argument("--plot_type", type=str, default="2d", choices=["2d", "3d"], help="plot type for replay buffer visualization")
    parser.add_argument("--vis_info", type=str2bool, default=True, help="enable visualization information of rollout in a video")

    # evaluation
    parser.add_argument("--ckpt_num", type=int, default=None, help="checkpoint nubmer")
    parser.add_argument("--num_eval", type=int, default=10, help="number of evaluations")
    parser.add_argument(
        "--save_rollout",
        type=str2bool,
        default=False,
        help="save rollout information during evaluation",
    )
    parser.add_argument(
        "--record", type=str2bool, default=True, help="enable video recording"
    )
    parser.add_argument("--record_caption", type=str2bool, default=True)
    parser.add_argument(
        "--num_record_samples",
        type=int,
        default=1,
        help="number of trajectories to collect during eval",
    )

    # misc
    parser.add_argument("--prefix", type=str, default="test", help="prefix for wandb")
    parser.add_argument("--notes", type=str, default="", help="notes")
    parser.add_argument("--seed", type=int, default=1234, help="random seed")
    parser.add_argument("--debug", type=str2bool, default=False, help="enable debugging model")

    return parser
