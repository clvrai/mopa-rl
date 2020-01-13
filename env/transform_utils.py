import math
import numpy as np
from pyquaternion import Quaternion


PI = np.pi
EPS = np.finfo(float).eps * 4.


def sample_quat(low=0, high=2*np.pi):
    """Samples quaternions of random rotations along the z-axis."""
    rot_angle = np.random.uniform(high=high, low=low)
    return [np.cos(rot_angle / 2), 0, 0, np.sin(rot_angle / 2)]


# code from stanford robosuite
def convert_quat(q, to="xyzw"):
    """
    Converts quaternion from one convention to another.
    The convention to convert TO is specified as an optional argument.
    If to == 'xyzw', then the input is in 'wxyz' format, and vice-versa.

    Args:
        q: a 4-dim numpy array corresponding to a quaternion
        to: a string, either 'xyzw' or 'wxyz', determining
            which convention to convert to.
    """
    if to == "xyzw":
        return q[[1, 2, 3, 0]]
    if to == "wxyz":
        return q[[3, 0, 1, 2]]
    raise Exception("convert_quat: choose a valid `to` argument (xyzw or wxyz)")


def norm(x):
    return x / np.linalg.norm(x)


def lookat_to_quat(forward, up):
    vector = norm(forward)
    vector2 = norm(np.cross(norm(up), vector))
    vector3 = np.cross(vector, vector2)
    m00 = vector2[0]
    m01 = vector2[1]
    m02 = vector2[2]
    m10 = vector3[0]
    m11 = vector3[1]
    m12 = vector3[2]
    m20 = vector[0]
    m21 = vector[1]
    m22 = vector[2]

    num8 = (m00 + m11) + m22
    quaternion = np.zeros(4)
    if num8 > 0:
        num = np.sqrt(num8 + 1)
        quaternion[3] = num * 0.5
        num = 0.5 / num
        quaternion[0] = (m12 - m21) * num
        quaternion[1] = (m20 - m02) * num
        quaternion[2] = (m01 - m10) * num
        return quaternion

    if ((m00 >= m11) and (m00 >= m22)):
        num7 = np.sqrt(((1 + m00) - m11) - m22)
        num4 = 0.5 / num7
        quaternion[0] = 0.5 * num7
        quaternion[1] = (m01 + m10) * num4
        quaternion[2] = (m02 + m20) * num4
        quaternion[3] = (m12 - m21) * num4
        return quaternion

    if m11 > m22:
        num6 = np.sqrt(((1 + m11) - m00) - m22)
        num3 = 0.5 / num6
        quaternion[0] = (m10+ m01) * num3
        quaternion[1] = 0.5 * num6
        quaternion[2] = (m21 + m12) * num3
        quaternion[3] = (m20 - m02) * num3
        return quaternion

    num5 = np.sqrt(((1 + m22) - m00) - m11)
    num2 = 0.5 / num5
    quaternion[0] = (m20 + m02) * num2
    quaternion[1] = (m21 + m12) * num2
    quaternion[2] = 0.5 * num5
    quaternion[3] = (m01 - m10) * num2
    return quaternion


# https://www.gamedev.net/forums/topic/56471-extracting-direction-vectors-from-quaternion/
def forward_vector_from_quat(quat):
    qw = quat[0]
    qx = quat[1]
    qy = quat[2]
    qz = quat[3]

    x = 2 * (qx * qy + qw * qz)#
    y = 1 - 2 * (qx * qx + qz * qz)
    z = 2 * (qy * qz - qw * qx)#
    return np.array([x, y, z])


def up_vector_from_quat(quat):
    qw = quat[0]
    qx = quat[1]
    qy = quat[2]
    qz = quat[3]

    x = 2 * (qx * qz - qw * qy)
    y = 2 * (qy * qz + qw * qx)
    z = 1 - 2 * (qx * qx + qy * qy)
    return np.array([x, y, z])


def right_vector_from_quat(quat):
    qw = quat[0]
    qx = quat[1]
    qy = quat[2]
    qz = quat[3]

    x = 1 - 2 * (qy * qy + qz * qz)
    y = 2 * (qx * qy - qw * qz)
    z = 2 * (qx * qz + qw * qy)#
    return np.array([x, y, z])


def quat_dist(quat1, quat2):
    q1 = Quaternion(axis=quat1[:-1], angle=quat1[-1])
    q2 = Quaternion(axis=quat2[:-1], angle=quat2[-1])
    return Quaternion.sym_distance(q1, q2)


def l2_dist(a, b):
    return np.linalg.norm(a - b)


def cos_dist(a, b):
    return np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b)

