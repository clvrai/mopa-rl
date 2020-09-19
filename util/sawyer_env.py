import os
from collections import OrderedDict
from numbers import Number

import numpy as np
from gym.spaces import Box
import mujoco_py

ENV_ASSET_DIR = os.path.join(os.path.dirname(__file__), "../env/assets")


def create_stats_ordered_dict(
    name,
    data,
    stat_prefix=None,
    always_show_all_stats=True,
    exclude_max_min=False,
):
    if stat_prefix is not None:
        name = "{} {}".format(stat_prefix, name)
    if isinstance(data, Number):
        return OrderedDict({name: data})

    if len(data) == 0:
        return OrderedDict()

    if isinstance(data, tuple):
        ordered_dict = OrderedDict()
        for number, d in enumerate(data):
            sub_dict = create_stats_ordered_dict(
                "{0}_{1}".format(name, number),
                d,
            )
            ordered_dict.update(sub_dict)
        return ordered_dict

    if isinstance(data, list):
        try:
            iter(data[0])
        except TypeError:
            pass
        else:
            data = np.concatenate(data)

    if isinstance(data, np.ndarray) and data.size == 1 and not always_show_all_stats:
        return OrderedDict({name: float(data)})

    stats = OrderedDict(
        [
            (name + " Mean", np.mean(data)),
            (name + " Std", np.std(data)),
        ]
    )
    if not exclude_max_min:
        stats[name + " Max"] = np.max(data)
        stats[name + " Min"] = np.min(data)
    return stats


def get_generic_path_information(paths, stat_prefix=""):
    """
    Get an OrderedDict with a bunch of statistic names and values.
    """
    statistics = OrderedDict()
    returns = [sum(path["rewards"]) for path in paths]

    rewards = np.vstack([path["rewards"] for path in paths])
    statistics.update(
        create_stats_ordered_dict("Rewards", rewards, stat_prefix=stat_prefix)
    )
    statistics.update(
        create_stats_ordered_dict("Returns", returns, stat_prefix=stat_prefix)
    )
    actions = [path["actions"] for path in paths]
    if len(actions[0].shape) == 1:
        actions = np.hstack([path["actions"] for path in paths])
    else:
        actions = np.vstack([path["actions"] for path in paths])
    statistics.update(
        create_stats_ordered_dict("Actions", actions, stat_prefix=stat_prefix)
    )
    statistics["Num Paths"] = len(paths)

    return statistics


def get_average_returns(paths):
    returns = [sum(path["rewards"]) for path in paths]
    return np.mean(returns)


def get_path_lengths(paths):
    return [len(path["observations"]) for path in paths]


def get_stat_in_paths(paths, dict_name, scalar_name):
    if len(paths) == 0:
        return np.array([[]])

    if type(paths[0][dict_name]) == dict:
        # Support rllab interface
        return [path[dict_name][scalar_name] for path in paths]

    return [[info[scalar_name] for info in path[dict_name]] for path in paths]


def get_asset_full_path(file_name):
    return os.path.join(ENV_ASSET_DIR, file_name)


def concatenate_box_spaces(*spaces):
    """
    Assumes dtypes of all spaces are the of the same type
    """
    low = np.concatenate([space.low for space in spaces])
    high = np.concatenate([space.high for space in spaces])
    return Box(low=low, high=high, dtype=np.float32)


def quat2axisangle(quat):
    theta = 0
    axis = np.array([0, 0, 1])
    sin_theta = np.linalg.norm(quat[1:])

    if sin_theta > 0.0001:
        theta = 2 * np.arcsin(sin_theta)
        theta *= 1 if quat[0] >= 0 else -1
        axis = quat[1:] / sin_theta

    return axis, theta


def quat_to_zangle(quat):
    q = quat_mul(quat_inv(quat_create(np.array([0, 1.0, 0]), np.pi / 2)), quat)
    ax, angle = quat2axisangle(q)
    return angle


def zangle_to_quat(zangle):
    """
    :param zangle in rad
    :return: quaternion
    """
    return quat_mul(
        quat_create(np.array([0, 1.0, 0]), np.pi / 2),
        quat_create(np.array([-1.0, 0, 0]), zangle),
    )


def quat_create(axis, angle):
    """
    Create a quaternion from an axis and angle.
    :param axis The three dimensional axis
    :param angle The angle in radians
    :return: A 4-d array containing the components of a quaternion.
    """
    quat = np.zeros([4], dtype="float")
    mujoco_py.functions.mju_axisAngle2Quat(quat, axis, angle)
    return quat


def quat_inv(quat):
    """
    Invert a quaternion, represented by a 4d array.
    :param A quaternion (4-d array). Must not be the zero quaternion (all elements equal to zero)
    :return: A 4-d array containing the components of a quaternion.
    """
    d = 1.0 / np.sum(quat ** 2)
    return d * np.array([1.0, -1.0, -1.0, -1.0]) * quat


def quat_mul(quat1, quat2):
    """
    Multiply two quaternions, both represented as 4-d arrays.
    """
    prod_quat = np.zeros([4], dtype="float")
    mujoco_py.functions.mju_mulQuat(prod_quat, quat1, quat2)
    return prod_quat
