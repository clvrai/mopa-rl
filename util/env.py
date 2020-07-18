import os, sys
import numpy as np
from collections import OrderedDict
from numbers import Number

import numpy as np
from gym.spaces import Box
import mujoco_py


ENV_ASSET_DIR = os.path.join(os.path.dirname(__file__), 'assets')

def joint_convert(angle):
    if angle > 0:
        if (angle // 3.14) % 2 == 0:
            return angle % 3.14
        else:
            return angle % 3.14 - 3.14
    else:
        if (angle // -3.14) % 2 == 0:
            return angle % -3.14
        else:
            return angle % -3.14 + 3.14




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

    if (isinstance(data, np.ndarray) and data.size == 1
            and not always_show_all_stats):
        return OrderedDict({name: float(data)})

    stats = OrderedDict([
        (name + ' Mean', np.mean(data)),
        (name + ' Std', np.std(data)),
    ])
    if not exclude_max_min:
        stats[name + ' Max'] = np.max(data)
        stats[name + ' Min'] = np.min(data)
    return stats


def get_generic_path_information(paths, stat_prefix=''):
    """
    Get an OrderedDict with a bunch of statistic names and values.
    """
    statistics = OrderedDict()
    returns = [sum(path["rewards"]) for path in paths]

    rewards = np.vstack([path["rewards"] for path in paths])
    statistics.update(create_stats_ordered_dict('Rewards', rewards,
                                                stat_prefix=stat_prefix))
    statistics.update(create_stats_ordered_dict('Returns', returns,
                                                stat_prefix=stat_prefix))
    actions = [path["actions"] for path in paths]
    if len(actions[0].shape) == 1:
        actions = np.hstack([path["actions"] for path in paths])
    else:
        actions = np.vstack([path["actions"] for path in paths])
    statistics.update(create_stats_ordered_dict(
        'Actions', actions, stat_prefix=stat_prefix
    ))
    statistics['Num Paths'] = len(paths)

    return statistics


def get_average_returns(paths):
    returns = [sum(path["rewards"]) for path in paths]
    return np.mean(returns)


def get_path_lengths(paths):
    return [len(path['observations']) for path in paths]


def get_stat_in_paths(paths, dict_name, scalar_name):
    if len(paths) == 0:
        return np.array([[]])

    if type(paths[0][dict_name]) == dict:
        # Support rllab interface
        return [path[dict_name][scalar_name] for path in paths]

    return [
        [info[scalar_name] for info in path[dict_name]]
        for path in paths
    ]


def get_asset_full_path(file_name):
    return os.path.join(ENV_ASSET_DIR, file_name)

def concatenate_box_spaces(*spaces):
    """
    Assumes dtypes of all spaces are the of the same type
    """
    low = np.concatenate([space.low for space in spaces])
    high = np.concatenate([space.high for space in spaces])
    return Box(low=low, high=high, dtype=np.float32)

def quat_to_zangle(quat):
    q = quat_mul(quat_inv(quat_create(np.array([0,1.,0]), np.pi/2)), quat)
    ax,angle = mjrot.quat2axisangle(q)
    return angle

def zangle_to_quat(zangle):
    """
    :param zangle in rad
    :return: quaternion
    """
    return quat_mul(quat_create(np.array([0, 1., 0]), np.pi/2),
                    quat_create(np.array([-1., 0, 0]), zangle))

def quat_create(axis, angle):
    """
        Create a quaternion from an axis and angle.
        :param axis The three dimensional axis
        :param angle The angle in radians
        :return: A 4-d array containing the components of a quaternion.
    """
    quat = np.zeros([4], dtype='float')
    mujoco_py.functions.mju_axisAngle2Quat(quat, axis, angle)
    return quat

def quat_inv(quat):
    """
        Invert a quaternion, represented by a 4d array.
        :param A quaternion (4-d array). Must not be the zero quaternion (all elements equal to zero)
        :return: A 4-d array containing the components of a quaternion.
    """
    d = 1. / np.sum(quat**2)
    return d * np.array([1., -1., -1., -1.]) * quat

def quat_mul(quat1, quat2):
    """
        Multiply two quaternions, both represented as 4-d arrays.
    """
    prod_quat = np.zeros([4], dtype='float')
    mujoco_py.functions.mju_mulQuat(prod_quat, quat1, quat2)
    return prod_quat


def mat2quat(rmat, precise=False):
    """
    Converts given rotation matrix to quaternion.
    Args:
        rmat: 3x3 rotation matrix
        precise: If isprecise is True, the input matrix is assumed to be a precise
             rotation matrix and a faster algorithm is used.
    Returns:
        vec4 float quaternion angles
    """
    M = np.array(rmat, dtype=np.float32, copy=False)[:3, :3]
    if precise:
        q = np.empty((4,))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 0, 1, 2
            if M[1, 1] > M[0, 0]:
                i, j, k = 1, 2, 0
            if M[2, 2] > M[i, i]:
                i, j, k = 2, 0, 1
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
            q = q[[3, 0, 1, 2]]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = np.array(
            [
                [m00 - m11 - m22, 0.0, 0.0, 0.0],
                [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
                [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
                [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
            ]
        )
        K /= 3.0
        # quaternion is Eigen vector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[[3, 0, 1, 2], np.argmax(w)]
    if q[0] < 0.0:
        np.negative(q, q)
    return q[[1, 2, 3, 0]]
