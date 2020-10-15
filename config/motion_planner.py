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
            "rrt",
            "rrt_connect",
        ],
        help="planner type"
    )
    parser.add_argument(
        "--simple_planner_type",
        type=str,
        default="rrt_connect",
        choices=[
            "rrt",
            "rrt_connect",
        ],
        help="planner type for simple planner"
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
        help="planner objective function"
    )
    parser.add_argument("--threshold", type=float, default=0.0, help="threshold for optimization objective")
    parser.add_argument("--is_simplified", type=str2bool, default=False, help="enable simplification of planned trajectory')
    parser.add_argument("--simplified_duration", type=float, default=0.01, help="duration of simplification of planned trajectory")
    parser.add_argument("--simple_planner_simplified", type=str2bool, default=False, help="enable simplification of planned trajectory for simple planner")
    parser.add_argument(
        "--simple_planner_simplified_duration", type=float, default=0.01,
        'duration of simplification of planned trajectory for simple planner'
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
