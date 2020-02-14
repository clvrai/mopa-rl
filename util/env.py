import os, sys

import numpy as np

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

