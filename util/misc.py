import numpy as np
import moviepy.editor as mpy
import cv2


def save_video(fpath, frames, fps=8.0):
    def f(t):
        frame_length = len(frames)
        new_fps = 1.0 / (1.0 / fps + 1.0 / frame_length)
        idx = min(int(t * new_fps), frame_length - 1)
        return frames[idx]

    video = mpy.VideoClip(f, duration=len(frames) / fps + 2)
    video.write_videofile(fpath, fps, verbose=False)
    print("[*] Video saved: {}".format(fpath))


def make_ordered_pair(id1, id2):
    return (min(id1, id2), max(id1, id2))


def render_frame(env, step, info={}):
    color = (200, 200, 200)
    text = "Step: {}".format(step)
    frame = env.render("rgb_array") * 255.0
    fheight, fwidth = frame.shape[:2]
    frame = np.concatenate([frame, np.zeros((fheight, fwidth, 3))], 0)

    font_size = 0.4
    thickness = 1
    offset = 12
    x, y = 5, fheight + 10
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_size,
        (255, 255, 0),
        thickness,
        cv2.LINE_AA,
    )

    for i, k in enumerate(info.keys()):
        v = info[k]
        key_text = "{}: ".format(k)
        (key_width, _), _ = cv2.getTextSize(
            key_text, cv2.FONT_HERSHEY_SIMPLEX, font_size, thickness
        )
        cv2.putText(
            frame,
            key_text,
            (x, y + offset * (i + 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_size,
            (66, 133, 244),
            thickness,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            str(v),
            (x + key_width, y + offset * (i + 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_size,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
    return frame


# workaround for mujoco py issue #390
def mujocopy_render_hack():
    render_hack = false  # set to true for bugfix on bad openGL context
    if render_hack:
        print("Setting an offscreen GlfwContext. See mujoco-py issue #390")
        from mujoco_py import GlfwContext

        GlfwContext(offscreen=True)  # Create a window to init GLFW.
    return
