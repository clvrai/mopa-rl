# Motion Planner Augmented Reinforcement Learning

Deep reinforcement learning (RL) agents are able to learn contact-rich manipulation tasks by maximizing a reward signal, but require large amounts of experience, especially in environments with many obstacles that complicate exploration. In contrast, motion planners use explicit models of the agent and environment to plan collision-free paths to faraway goals, but suffer from inaccurate models in tasks that require contacts with the environment. To combine the benefits of both approaches, we propose motion planner augmented RL (MoPA-RL) which augments the action space of an RL agent with the long-horizon planning capabilities of motion planners.

## Prerequisites
- Ubuntu 18.04
- Python 3.7 (`python3.7`, `python3.7-dev`)
- torch
- `libyaml-cpp-dev` (`sudo apt install libyaml-cpp-dev` or `brew install libyaml yaml-cpp`)
- gym 0.15.4
- [MuJoCo 2.0.2.5 ](http://www.mujoco.org/)

## Installation 
1. Install Mujoco 2.0 and add the following environment variables into `~/.bashrc` or `~/.zshrc`.
```
# Download mujoco 2.0
$ wget https://www.roboti.us/download/mujoco200_linux.zip -O mujoco.zip
$ unzip mujoco.zip -d ~/.mujoco
$ mv ~/.mujoco/mujoco200_linux ~/.mujoco/mujoco200

# Copy mujoco license key `mjkey.txt` to `~/.mujoco`

# Add mujoco to LD_LIBRARY_PATH
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200/bin

# For GPU rendering (replace 418 with your nvidia driver version)
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-418

# Only for a headless server
$ export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia-418/libGL.so
```

2. Clone this repository and install python dependencies
```
sudo apt-get install libgl1-mesa-dev libgl1-mesa-glx libosmesa6-dev patchelf libopenmpi-dev libglew-dev python3-pip python3-numpy python3-scipy
git clone git@github.com:youngwoon/hrl-planner.git
# install required python packages in your new env
pip install -r requirements.txt
```

3. Install ompl 

```
# Linux 
sh ./scripts/misc/installEigen.sh #from the home directory # install Eigen
# Mac OS
brew install eigen

git clone git@github.com:ompl/ompl.git ../ompl
cd ../ompl
cmake .
sudo make install
```

4. Compile motion planner 
```
cd ./hrl-planner/motion_planner
python setup.py build_ext --inplace
```

## Usage

```
# 2-D Push
sh ./scripts/2d/baseline.sh  # baseline
sh ./scripts/2d/mopa.sh # MoPA-SAC
sh ./scripts/2d/mopa_ik.sh # MoPA-SAC IK

# Sawyer Push
sh ./scripts/3d/push/baseline.sh # baseline
sh ./scripts/3d/push/mopa.sh # MoPA-SAC
sh ./scripts/3d/push/mopa_ik.sh # MoPA-SAC IK

# Sawyer Lift
sh ./scripts/3d/lift/baseline.sh # baseline
sh ./scripts/3d/lift/mopa.sh # MoPA-SAC
sh ./scripts/3d/lift/mopa_ik.sh # MoPA-SAC IK

# Sawyer Assembly
sh ./scripts/3d/assembly/baseline.sh # baseline
sh ./scripts/3d/assembly/mopa.sh # MoPA-SAC
sh ./scripts/3d/assembly/mopa_ik.sh # MoPA-SAC IK
```

## Trouble shooting

#### pybind11-dev not found
```
wget http://archive.ubuntu.com/ubuntu/pool/universe/p/pybind11/pybind11-dev_2.2.4-2_all.deb
sudo apt install ./pybind11-dev_2.2.4-2_all.deb
```

