import os, sys

import numpy as np
import moviepy.editor as mpy


def save_video(fpath, frames, fps=8.):
    def f(t):
        frame_length = len(frames)
        new_fps = 1./(1./fps + 1./frame_length)
        idx = min(int(t*new_fps), frame_length-1)
        return frames[idx]

    video = mpy.VideoClip(f, duration=len(frames)/fps+2)
    video.write_videofile(fpath, fps, verbose=False)
    print("[*] Video saved: {}".format(fpath))

def make_ordered_pair(id1, id2):
    return (min(id1, id2), max(id1, id2))
