from util import str2bool


def add_arguments(parser):
    """
    Adds a list of arguments to argparser for the reacher environment.
    """
    # reacher
    parser.add_argument(
        "--planner_type",
        type=str,
        default="rrt_connect",
        choices=[
            "sst",
            "rrt",
            "rrt_connect",
            "prm_star",
            "kpiece",
            "spars",
            "lazy_prm_star",
            "rrt_sharp",
        ],
    )
    parser.add_argument(
        "--simple_planner_type",
        type=str,
        default="rrt_connect",
        choices=[
            "sst",
            "rrt",
            "rrt_connect",
            "prm_star",
            "kpiece",
            "spars",
            "lazy_prm_star",
        ],
    )
    parser.add_argument(
        "--planner_objective",
        type=str,
        default="path_length",
        choices=[
            "maximize_min_clearance",
            "path_length",
            "state_cost_integral",
            "constraint",
        ],
    )
    parser.add_argument("--sst_selection_radius", type=float, default=0.01)
    parser.add_argument("--sst_pruning_radius", type=float, default=0.01)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--construct_time", type=float, default=200.0)
    parser.add_argument("--is_simplified", type=str2bool, default=False)
    parser.add_argument("--simplified_duration", type=float, default=0.01)
    parser.add_argument("--simple_planner_simplified", type=str2bool, default=False)
    parser.add_argument(
        "--simple_planner_simplified_duration", type=float, default=0.01
    )


def get_default_config():
    """
    Gets default configurations for the reacher environment.
    """
    import argparse

    parser = argparse.ArgumentParser("Default Configuration for Motion Planner")
    add_argument(parser)

    config = parser.parse_args([])
    return config
