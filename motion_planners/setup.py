from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
from distutils.sysconfig import get_python_lib
import glob
import os
import sys
import platform

prefix_path = os.environ['HOME']
extensions = [
    Extension('planner', ['planner.pyx', 'KinematicPlanner.cpp', './src/mujoco_ompl_interface.cpp', './src/mujoco_wrapper.cpp',
                          ],
              include_dirs=["./include/", '/usr/local/include/eigen3', './3rd_party/include/',
                            os.path.join(prefix_path, '.mujoco/mujoco200/include/'), '/usr/local/include/ompl'],
              extra_objects=['/usr/local/lib/libompl.so', os.path.join(prefix_path, '.mujoco/mujoco200/bin/libmujoco200.so')],
              extra_compile_args=['-std=c++11'],
              language="c++"),
]
setup(
    name='mujoco-ompl',
    ext_modules=cythonize(extensions),
)
