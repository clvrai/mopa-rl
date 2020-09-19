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
    parser.add_argument("--env_args", type=str, default=None, help="arguments for environment")
    parser.add_argument("--terminal", type=str2bool, default=True)

    # training algorithm
    parser.add_argument("--algo", type=str, default="sac", choices=["sac", "td3"], help="algorithm")
    parser.add_argument("--policy", type=str, default="mlp", choices=["mlp"])
    parser.add_argument("--mopa", type=str2bool, default=False)
    parser.add_argument("--use_discount_meta", type=str2bool, default=True)
    parser.add_argument(
        "--ac_space_type",
        type=str,
        default="piecewise",
        choices=["normal", "piecewise"],
    )
    parser.add_argument("--use_ik_target", type=str2bool, default=False)
    parser.add_argument("--ik_target", type=str, default="fingertip")
    parser.add_argument("--expand_ac_space", type=str2bool, default=False)

    # vanilla rl
    parser.add_argument("--rl_hid_size", type=int, default=256)
    parser.add_argument(
        "--rl_activation", type=str, default="relu", choices=["relu", "elu", "tanh"]
    )
    parser.add_argument("--tanh_policy", type=str2bool, default=True)
    parser.add_argument("--activation", type=str, default="tanh")
    parser.add_argument("--actor_num_hid_layers", type=int, default=2)

    # motion planning
    parser.add_argument("--ignored_contact_geoms", nargs="+", default=None)
    parser.add_argument("--allow_manipulation_collision", type=str2bool, default=False)
    parser.add_argument("--allow_approximate", type=str2bool, default=False)
    parser.add_argument("--allow_invalid", type=str2bool, default=False)
    parser.add_argument("--find_collision_free", type=str2bool, default=False)
    parser.add_argument("--use_double_planner", type=str2bool, default=False)
    parser.add_argument("--num_trials", type=int, default=100)
    parser.add_argument("--use_interpolation", type=str2bool, default=True)
    parser.add_argument(
        "--interpolate_type", type=str, default="simple", choices=["planner", "simple"]
    )

    # single policy
    parser.add_argument("--omega", type=float, default=1.0)
    parser.add_argument("--reuse_data", type=str2bool, default=False)
    parser.add_argument("--max_reuse_data", type=int, default=30)
    parser.add_argument("--min_reuse_span", type=int, default=1)
    parser.add_argument("--action_range", type=float, default=1.0)
    parser.add_argument("--discrete_action", type=str2bool, default=False)
    parser.add_argument("--stochastic_eval", type=str2bool, default=False)

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
    parser.add_argument("--actor_update_freq", type=int, default=1)
    parser.add_argument("--critic_target_update_freq", type=int, default=1)

    # training
    parser.add_argument("--is_train", type=str2bool, default=True)
    parser.add_argument(
        "--num_batches",
        type=int,
        default=1,
        help="the times to update the network per epoch",
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="the sample batch size"
    )
    parser.add_argument("--max_global_step", type=int, default=int(3e6))
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--invalid_planner_rew", type=float, default=0.0)

    # sac
    parser.add_argument("--start_steps", type=int, default=1e4)
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Temperature for Gumbel Softmax"
    )
    parser.add_argument("--use_automatic_entropy_tuning", type=str2bool, default=True)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--reward_scale", type=float, default=1.0)
    parser.add_argument("--use_smdp_update", type=str2bool, default=False)
    parser.add_argument("--actor_bias", type=float, default=None)
    parser.add_argument("--discrete_ent_coef", type=float, default=1.0)

    # td3
    parser.add_argument("--target_noise", type=float, default=0.2)
    parser.add_argument("--action_noise", type=float, default=0.1)
    parser.add_argument("--noise_clip", type=float, default=0.5)

    # log
    parser.add_argument("--log_interval", type=int, default=1000)
    parser.add_argument("--vis_replay_interval", type=int, default=10000)
    parser.add_argument("--evaluate_interval", type=int, default=10000)
    parser.add_argument("--ckpt_interval", type=int, default=200000)
    parser.add_argument("--log_root_dir", type=str, default="log")
    parser.add_argument(
        "--wandb",
        type=str2bool,
        default=False,
        help="set it True if you want to use wandb",
    )
    parser.add_argument("--group", type=str, default=None)
    parser.add_argument("--vis_replay", type=str2bool, default=True)
    parser.add_argument("--plot_type", type=str, default="2d")
    parser.add_argument("--vis_info", type=str2bool, default=True)

    # evaluation
    parser.add_argument("--ckpt_num", type=int, default=None)
    parser.add_argument("--num_eval", type=int, default=10)
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
    parser.add_argument("--prefix", type=str, default="test")
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument("--seed", type=int, default=1234, help="random seed")
    parser.add_argument("--debug", type=str2bool, default=False)

    return parser
