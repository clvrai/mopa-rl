import sys, os
sys.path.append('./rai-python/rai/rai/ry')

import libry as ry
import numpy as np

class MotionPlanner:
    def __init__(self, env_file, vis=False):
        self.vis = vis
        self.K = ry.Config()
        self.K.addFile(env_file)

    def plan_motion(self, x_target, q_0=None, T=20):
        if q_0 is not None:
            self.K.setJointState(q_0)
        self.K.setFrameState(np.array([x_target[0], x_target[1], 0.01, 0, 0, 1, 0]), ['goal'])
        komo = self.K.komo_path(1., T, 10., True)
        komo.addObjective(time=[.9, 1.], type=ry.OT.eq, feature=ry.FS.positionDiff, frames=['endeff', 'goal'])
        komo.optimize(True)
        traj = self.get_trajectory(komo)

        if self.vis:
            komo.displayTrajectory()
        return traj

    def get_trajectory(self, komo):
        traj = []
        for i in range(komo.getT()):
            self.K.setFrameState(komo.getConfiguration(i))
            traj.append(self.K.getJointState())
        return traj


