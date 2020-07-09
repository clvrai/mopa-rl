import numpy as np
from gym import spaces
import cv2


def observation_size(observation_space):
    if isinstance(observation_space, spaces.Dict):
        return sum([observation_size(value) for key, value in observation_space.spaces.items()])
    elif isinstance(observation_space, spaces.Box):
        return np.product(observation_space.shape)

def render_frame(env, step, info={}):
    color = (200, 200, 200)
    text = "Step: {}".format(step)
    frame = env.render('rgb_array') * 255.0
    fheight, fwidth = frame.shape[:2]
    frame = np.concatenate([frame, np.zeros((fheight, fwidth, 3))], 0)

    font_size = 0.4
    thickness = 1
    offset = 12
    x, y = 5, fheight+10
    cv2.putText(frame, text,
                (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                font_size, (255, 255, 0), thickness, cv2.LINE_AA)

    for i, k in enumerate(info.keys()):
        v = info[k]
        key_text = '{}: '.format(k)
        (key_width, _), _ = cv2.getTextSize(key_text, cv2.FONT_HERSHEY_SIMPLEX,
                                          font_size, thickness)
        cv2.putText(frame, key_text,
                    (x, y+offset*(i+2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_size, (66, 133, 244), thickness, cv2.LINE_AA)
        cv2.putText(frame, str(v),
                    (x + key_width, y+offset*(i+2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_size, (255, 255, 255), thickness, cv2.LINE_AA)
    return frame

def action_size(action_space):
    if isinstance(action_space, spaces.Dict):
        return sum([action_size(value) for key, value in action_space.spaces.items()])
    elif isinstance(action_space, spaces.Box):
        return np.product(action_space.shape)
    elif isinstance(action_space, spaces.Discrete):
        return action_space.n
    elif isinstance(action_space, spaces.MultiDiscrete):
        return np.product(action_space.nvec)
    elif isinstance(action_space, spaces.MultiBinary):
        return action_space.n


