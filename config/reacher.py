from util import str2bool


def add_arguments(parser):
    """
    Adds a list of arguments to argparser for the reacher environment.
    """
    # reacher
    parser.add_argument("--reward_type", type=str, default="dense",
                        choices=["dense", "sparse"])
    parser.add_argument("--distance_threshold", type=float, default=0.01)

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


def get_default_config():
    """
    Gets default configurations for the reacher environment.
    """
    import argparse
    from util import str2bool

    parser = argparse.ArgumentParser("Default Configuration for Reacher Environment")
    add_argument(parser)

    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--debug", type=str2bool, default=False)

    config = parser.parse_args([])
    return config

