from util import str2bool


def add_arguments(parser):
    """
    Adds a list of arguments to argparser for the reacher environment.
    """
    # reacher
    parser.add_argument("--planner_type", type=str, default="rrt",
                        choices=["sst", "rrt", "rrt_connect", "prm_star"])
    parser.add_argument("--planner_objective", type=str, default="",
                        choices=["maximize_min_clearance", "path_length", "state_const_integral", "constraint"])
    parser.add_argument("--sst_selection_radius", type=float, default=0.01)
    parser.add_argument("--sst_pruning_radius", type=float, default=0.01)


def get_default_config():
    """
    Gets default configurations for the reacher environment.
    """
    import argparse
    from util import str2bool

    parser = argparse.ArgumentParser("Default Configuration for Motion Planner")
    add_argument(parser)

    config = parser.parse_args([])
    return config

