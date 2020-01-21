import os, sys

import numpy as np
import shuntil
import subprocess
from threading import Lock, Thread
import yaml

lock = Lock()
n = 0

class SamplingBasedPlanner:
    def __init__(self, config):
        self.config

    def plan(self):
        shuntil.mkdir('./tmp/')
        for _ in range(self.config.threads):
            threads.append(Thread(
                target=self._plan,
                args=()))
            threads[-1].start()

        for thread in threads:
            thread.joint()

    def _plan(self):
        global n
        lock.acquire()
        next_n = n
        n += 1
        lock.release()

        while (next_n < self.config.batch_size):
            outfile = os.path.join('./tmp', str(next_n)+'.out')
            if (not os.path.exists(outfiel)):
                args = args + ['--output', outfile]
                subprocess.run(args)

            lock.acquire()
            next_n = n
            n+= 1
            lock.release()

    def get_trajectory(self):
        raise NotImplementedError
