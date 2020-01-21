from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
from distutils.sysconfig import get_python_lib
import glob
import os
import sys

extensions = [
    Extension("planner",
              sources=["src/plan.pyx", "src/planner.cxx"],
              include_dirs=["./include"],
              language="c++")
]

setup(name="mujoco-ompl",
      packages=find_packages(),
      ext_modules=cythonize(extensions),
      install_requires=['numpy'],
      )

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
