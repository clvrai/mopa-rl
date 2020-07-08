from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
from distutils.sysconfig import get_python_lib
import glob
import os
import sys
import platform

# export MACOSX_DEPLOYMENT_TARGET=10.13

prefix_path = os.environ['HOME']
ompl_lib_path = os.path.join(prefix_path, '/usr/local/lib/libompl.dylib')
eigen_include_path = '/usr/local/include/eigen3'

extensions = [
    Extension('planner', ['planner.pyx', 'KinematicPlanner.cpp', './src/mujoco_ompl_interface.cpp', './src/mujoco_wrapper.cpp',
                          ],
              include_dirs=["./include/", '/opt/local/include', eigen_include_path, './3rd_party/include/', '/opt/local/include/boost/',
                            os.path.join(prefix_path, '.mujoco/mujoco200/include/'), '/usr/local/include/ompl', '/usr/local/include'],
              extra_objects=[ompl_lib_path, os.path.join(prefix_path, '.mujoco/mujoco200/bin/libmujoco200.dylib')],
              extra_compile_args=['-std=c++11', '-stdlib=libc++'],
              language="c++"),
]
setup(
    name='mujoco-ompl',
    ext_modules=cythonize(extensions),
)
