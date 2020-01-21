from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
from distutils.sysconfig import get_python_lib
import glob
import os
import sys

extensions = [
    Extension('planner', ['planner.pyx', 'Planner.cpp', './src/mujoco_ompl_interface.cpp', './src/mujoco_wrapper.cpp',
                          # "./ompl/src/ompl/base/spaces/src/RealVectorStateProjections.cpp",
                          # "./ompl/src/ompl/base/src/ProjectionEvaluator.cpp",
                          # "./ompl/src/ompl/control/src/SpaceInformation.cpp",
                          # "./ompl/src/ompl/base/src/SpaceInformation.cpp",
                          # "./ompl/src/ompl/base/spaces/src/ReedsSheppStateSpace.cpp"],
                          ],
              include_dirs=["./include/", '/usr/local/include/eigen3', './3rd_party/include/',
                            '/home/ubuntu/.mujoco/mujoco200/include/', '/usr/local/include/ompl'],
              extra_objects=['/usr/local/lib/libompl.so', '/home/ubuntu/.mujoco/mujoco200/bin/libmujoco200.so'],
              language="c++")
]
setup(
    name='mujoco-ompl',
    ext_modules=cythonize(extensions),
)
# setup(name="mujoco-ompl",
#       packages=find_packages(),
#       ext_modules=cythonize(extensions),
#       install_requires=['numpy'],
#       )
#
# ext = Extension('planner', 
#                 ["./src/planner.pyx"],
#                 language="c++",
#                 include_dirs=['./include'])

# setup(
#     name = "mujoco-ompl",
#     package_dir = {'': 'src'},
#     data_files = [(get_python_lib(), glob.glob('src/*.so'))],
#     )
# setup(
#     name='mujoco-ompl',
#     package_dir={'':'src'},
#     ext_modules=cythonize(ext),
#     data_files=[(get_python_lib(), glob.glob('src/*.so'))],
# )
