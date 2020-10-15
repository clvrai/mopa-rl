from util import str2bool


def add_arguments(parser):
    """
    Adds a list of arguments to argparser for the pusher environment.
    """
    # pusher
    parser.add_argument(
        "--reward_type",
        type=str,
        default="dense",
        choices=["sparse"],
        help="reward type",
    )
    parser.add_argument("--distance_threshold", type=float, default=0.05, help="distance threshold for termination")
    parser.add_argument("--max_episode_steps", type=int, default=150, help="maximum timesteps in an episode")

    # observations
    parser.add_argument(
        "--screen_width", type=int, default=500, help="width of camera image"
    )
    parser.add_argument(
        "--screen_height", type=int, default=500, help="height of camera image"
    )
    parser.add_argument(
        "--frame_skip", type=int, default=1, help="Numer of skip frames"
    )
    parser.add_argument("--kp", type=float, default=150.0, help="p term for a PID controller")
    parser.add_argument("--kd", type=float, default=20.0, help="d term for a PID controller")
    parser.add_argument("--ki", type=float, default=0.1, help="i term for a PID controller")
    parser.add_argument("--frame_dt", type=float, default=1.0, help="dt between each frame")
    parser.add_argument("--ctrl_reward_coef", type=float, default=0, help="control reward coefficient")
    parser.add_argument("--success_reward", type=float, default=150.0, help="completion reward")
    parser.add_argument("--camera_name", type=str, default="cam0", help="camera name in an environment")
    parser.add_argument("--range", type=float, default=0.2, help="range of motion planner")
    parser.add_argument("--simple_planner_range", type=float, default=0.1, help="range of simple motion planner")
    parser.add_argument("--timelimit", type=float, default=1.0, help="timelimit for planning")
    parser.add_argument("--simple_planner_timelimit", type=float, default=0.02, help="timelimit for planning by simple motion planner")
    parser.add_argument("--contact_threshold", type=float, default=-0.0015, help='depth threshold for contact')
    parser.add_argument("--joint_margin", type=float, default=0.0, help="margin of each joint")
    parser.add_argument("--step_size", type=float, default=0.04, help="step size for invalid target handling")


def get_default_config():
    """
    Gets default configurations for the pusher environment.
    """
    import argparse

    parser = argparse.ArgumentParser("Default Configuration for 2D Pusher Environment")
    add_argument(parser)

    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--debug", type=str2bool, default=False)

    config = parser.parse_args([])
    return config
