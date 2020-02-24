from util import str2bool


def add_arguments(parser):
    """
    Adds a list of arguments to argparser for the pusher environment.
    """
    # pusher
    parser.add_argument("--reward_type", type=str, default="dense",
                        choices=["dense", "sparse", "dist_diff", "composition"])
    parser.add_argument("--distance_threshold", type=float, default=0.06)
    parser.add_argument("--max_episode_steps", type=int, default=150)

    # observations
    parser.add_argument("--robot_ob", type=str2bool, default=True,
                        help="includes agent state in observation")
    parser.add_argument("--object_ob", type=str2bool, default=True,
                        help="includes object pose in observation")
    parser.add_argument("--visual_ob", type=str2bool, default=False,
                        help="includes camera image in observation")
    parser.add_argument("--screen_width", type=int, default=500,
                        help="width of camera image")
    parser.add_argument("--screen_height", type=int, default=500,
                        help="height of camera image")
    parser.add_argument("--frame_skip", type=int, default=1,
                        help="Numer of skip frames")
    parser.add_argument("--img_height", type=int, default=64,
                        help="Image observation height")
    parser.add_argument("--img_width", type=int, default=64,
                        help="Image observation width")
    parser.add_argument("--is_rgb", type=str2bool, default=False)
    parser.add_argument("--action_min", type=float, default=-1)
    parser.add_argument("--action_max", type=float, default=1)
    parser.add_argument("--kp", type=float, default=150.)
    parser.add_argument("--kd", type=float, default=20.)
    parser.add_argument("--ki", type=float, default=0.1)
    parser.add_argument("--frame_dt", type=float, default=1.)
    parser.add_argument("--reward_coef", type=float, default=10.)
    parser.add_argument("--ctrl_reward_coef", type=float, default=1.)
    parser.add_argument("--box_to_target_coef", type=float, default=1.)
    parser.add_argument("--end_effector_to_box_coef", type=float, default=1.)


def get_default_config():
    """
    Gets default configurations for the pusher environment.
    """
    import argparse
    from util import str2bool

    parser = argparse.ArgumentParser("Default Configuration for pusher Environment")
    add_argument(parser)

    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--debug", type=str2bool, default=False)

    config = parser.parse_args([])
    return config

