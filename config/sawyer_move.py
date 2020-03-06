from util import str2bool


def add_arguments(parser):
    """
    Adds a list of arguments to argparser for the sawyer environment.
    """
    # sawyer
    parser.add_argument("--single_object_mode", type=int, default=0)
    parser.add_argument("--object_type", type=str, default=None)
    parser.add_argument("--table_full_size", nargs=3, default=(0.39, 0.49, 0.82))
    parser.add_argument("--table_friction", nargs=3, default=(1, 0.005, 0.0001))
    parser.add_argument("--gripper_visualization", type=str2bool, default=False)
    parser.add_argument("--use_object_obs", type=str2bool, default=True)
    parser.add_argument("--reward_shaping", type=str2bool, default=False)
    parser.add_argument("--gripper_type", type=str, default="TwoFingerGripper")
    parser.add_argument("--use_indicator_object", type=str2bool, default=False)
    parser.add_argument("--control_freq", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--has_renderer", type=str2bool, default=False)
    parser.add_argument("--has_offscreen_renderer", type=str2bool, default=True)
    parser.add_argument("--render_collision_mesh", type=str2bool, default=True)
    parser.add_argument("--render_visual_mesh", type=str2bool, default=True)
    parser.add_argument("--use_camera_obs", type=str2bool, default=True)
    parser.add_argument("--camera_name", type=str, default='frontview')
    parser.add_argument("--placement_initializer", type=str, default=None)
    #parser.add_argument("--distance_threshold", type=float, default=0.01)

    parser.add_argument("--reward_type", type=str, default="dense",
                        choices=["dense", "sparse"])
    parser.add_argument("--distance_threshold", type=float, default=0.03)
    parser.add_argument("--max_episode_steps", type=int, default=1000)
    parser.add_argument("--camera_width", type=int, default=500,
                        help="width of camera image")
    parser.add_argument("--camera_height", type=int, default=500,
                        help="height of camera image")
    parser.add_argument("--camera_depth", type=str2bool, default=False)

    # observations
    parser.add_argument("--robot_ob", type=str2bool, default=True,
                        help="includes agent state in observation")
    parser.add_argument("--object_ob", type=str2bool, default=True,
                        help="includes object pose in observation")
    parser.add_argument("--visual_ob", type=str2bool, default=False,
                        help="includes camera image in observation")
    parser.add_argument("--frame_skip", type=int, default=5,
                        help="Numer of skip frames")
    parser.add_argument("--action_repeat", type=int, default=5)
    parser.add_argument("--img_height", type=int, default=84,
                        help="Image observation height")
    parser.add_argument("--img_width", type=int, default=84,
                        help="Image observation width")


def get_default_config():
    """
    Gets default configurations for the reacher environment.
    """
    import argparse
    from util import str2bool

    parser = argparse.ArgumentParser("Default Configuration for Reacher Environment")
    add_argument(parser)

    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--debug", type=str2bool, default=False)

    config = parser.parse_args([])
    return config

