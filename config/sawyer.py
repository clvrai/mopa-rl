from util import str2bool


def add_arguments(parser):
    """
    Adds a list of arguments to argparser for the sawyer environment.
    """
    # sawyer
    parser.add_argument(
        "--reward_type", type=str, default="dense", choices=["dense", "sparse"],
        help="reward type"
    )
    parser.add_argument("--distance_threshold", type=float, default=0.06, help="distance threshold for termination")
    parser.add_argument("--max_episode_steps", type=int, default=250, help="maximum timesteps in an episode")
    parser.add_argument(
        "--screen_width", type=int, default=500, help="width of camera image"
    )
    parser.add_argument(
        "--screen_height", type=int, default=500, help="height of camera image"
    )
    parser.add_argument("--camera_name", type=str, default="topview", help="camera name in an environment")

    # observations
    parser.add_argument(
        "--frame_skip", type=int, default=1, help="Numer of skip frames"
    )
    parser.add_argument("--action_repeat", type=int, default=5, help="number of action repeats")
    parser.add_argument("--ctrl_reward_coef", type=float, default=0, help="control reward coefficient")

    parser.add_argument("--kp", type=float, default=40.0, help="p term for a PID controller")  # 150.)
    parser.add_argument("--kd", type=float, default=8.0, help="d term for a PID controller")  # 20.)
    parser.add_argument("--ki", type=float, default=0.0, help="i term for a PID controller")
    parser.add_argument("--frame_dt", type=float, default=0.15, help="dt between each frame")  # 0.1)
    parser.add_argument("--use_robot_indicator", type=str2bool, default=True, help="enable visualization of robot indicator for motion planner")
    parser.add_argument("--use_target_robot_indicator", type=str2bool, default=True, help="enable visualization of robot indicator for target position of motion planner")
    parser.add_argument("--success_reward", type=float, default=150.0, help="completion reward")
    parser.add_argument("--range", type=float, default=0.1, help="range of motion planner")
    parser.add_argument("--simple_planner_range", type=float, default=0.05, help="range of simple motion planner")
    parser.add_argument("--timelimit", type=float, default=1.0, help="timelimit for motion planner")
    parser.add_argument("--simple_planner_timelimit", type=float, default=0.05, help="timelimit for simple motion planner")
    parser.add_argument("--contact_threshold", type=float, default=-0.002, help="depth thredhold for contact")
    parser.add_argument("--joint_margin", type=float, default=0.001, help="marin of each joint")
    parser.add_argument("--step_size", type=float, default=0.02, help="step size for invalid target handling")


def get_default_config():
    """
    Gets default configurations for the Sawyer environment.
    """
    import argparse

    parser = argparse.ArgumentParser("Default Configuration for Sawyer Environment")
    add_argument(parser)

    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--debug", type=str2bool, default=False)

    config = parser.parse_args([])
    return config
