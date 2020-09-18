# Motion Planner Augmented Reinforcement Learning
[Jun Yamada<sup>1</sup>](https://junjungoal.tech), [Youngwoon Lee<sup>1</sup>](https://youngwoon.github.io), [Gautam Salhotra<sup>2</sup>](https://www.gautamsalhotra.com/), [Karl Pertsch<sup>1</sup>](https://kpertsch.github.io), [Max Pflueger<sup>2</sup>](https://mpflueger.github.io/), [Gaurav S. Sukhatme<sup>2</sup>](http://robotics.usc.edu/~gaurav), [Joseph J. Lim<sup>1</sup>](https://clvrai.com) [Peter Englert<sup>2</sup>](http://www.peter-englert.net/) at [USC CLVR lab<sup>1</sup>](https://clvrai.com) and [USC RESL lab<sup>2</sup>](https://robotics.usc.edu/resl/) <br/>

<p align="center">
    <img src="docs/img/teaser.gif">
</p>

Deep reinforcement learning (RL) agents are able to learn contact-rich manipulation tasks by maximizing a reward signal, but require large amounts of experience, especially in environments with many obstacles that complicate exploration. In contrast, motion planners use explicit models of the agent and environment to plan collision-free paths to faraway goals, but suffer from inaccurate models in tasks that require contacts with the environment. To combine the benefits of both approaches, we propose motion planner augmented RL (MoPA-RL) which augments the action space of an RL agent with the long-horizon planning capabilities of motion planners.



## Prerequisites
- Ubuntu 18.04
- Python 3.7 (`python3.7`, `python3.7-dev`)
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

2. Download this repository and install python dependencies
```
sudo apt-get install libgl1-mesa-dev libgl1-mesa-glx libosmesa6-dev patchelf libopenmpi-dev libglew-dev python3-pip python3-numpy python3-scipy
unzip code-130.zip
cd code-130
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

# if ompl-x.x (x.x is the version) is installed in /usr/local/include, you need to rename it to ompl
mv /usr/local/include/ompl-x.x /usr/local/include/ompl
```

4. Compile motion planner 
```
cd ./hrl-planner/motion_planner
python setup.py build_ext --inplace
```

## Mujoco GPU rendering
To use GPU rendering for mujoco, you need to add `/usr/lib/nvidia-000` (`000` should be replaced with your NVIDIA driver version) to `LD_LIBRARY_PATH` before installing `mujoco-py`. Then, during `mujoco-py` compilation, it will show you `linuxgpuextension` instead of `linuxcpuextension`. In Ubuntu 18.04, you may encounter an GL-related error while building `mujoco-py`, open `venv/lib/python3.6/site-packages/mujoco_py/gl/eglshim.c` and comment line 5 `#include <GL/gl.h>` and line 7 `#include <GL/glext.h>`.

### Virtual display on headless machines
On servers, you don’t have a monitor. Use this to get a virtual monitor for rendering and put DISPLAY=:1 in front of a command.

```
# Run the next line for Ubuntu
$ sudo apt-get install xserver-xorg libglu1-mesa-dev freeglut3-dev mesa-common-dev libxmu-dev libxi-dev

# Configure nvidia-x
$ sudo nvidia-xconfig -a --use-display-device=None --virtual=1280x1024

# Launch a virtual display
$ sudo /usr/bin/X :1 &

# Run a command with DISPLAY=:1
DISPLAY=:1 <command>
```

## How to run experiments
0. Launch a virtual display (only for a headless server)
```
sudo /usr/bin/X :1 &
```

2. Train policies
-  2-D Push
```
sh ./scripts/2d/baseline.sh  # baseline
sh ./scripts/2d/mopa.sh # MoPA-SAC
sh ./scripts/2d/mopa_ik.sh # MoPA-SAC IK
```

- Sawyer Push
```
sh ./scripts/3d/push/baseline.sh # baseline
sh ./scripts/3d/push/mopa.sh # MoPA-SAC
sh ./scripts/3d/push/mopa_ik.sh # MoPA-SAC IK
```

- Sawyer Lift
```
sh ./scripts/3d/lift/baseline.sh # baseline
sh ./scripts/3d/lift/mopa.sh # MoPA-SAC
sh ./scripts/3d/lift/mopa_ik.sh # MoPA-SAC IK
```

- Sawyer Assembly
```
sh ./scripts/3d/assembly/baseline.sh # baseline
sh ./scripts/3d/assembly/mopa.sh # MoPA-SAC
sh ./scripts/3d/assembly/mopa_ik.sh # MoPA-SAC IK
```

## Directories 
The structure of the repository:

- `rl`: Reinforcement learning code
- `env`: Environment code for simulated experiments (2D Push, and Sawyer tasks)
- `util`: Utility code
- `motion_planners`: Motion Planner code
Log directories:

- `logs/rl.ENV.DATE.PREFIX.SEED`:
  - `cmd.sh`: A command used for running a job
  - `git.txt`: Log gitdiff
  - `prarms.json`: Summary of parameters
  - `video`: Generated evaulation videos (every evalute_interval)
  - `wandb`: Training summary of W&B, like tensorboard summary
  - `ckpt_*.pt`: Stored checkpoints (every ckpt_interval)
  - `replay_*.pt`: Stored replay buffers (every ckpt_interval)

## Reference
- PyTorch implementation of PPO and SAC: https://github.com/clvrai/coordination

## Trouble shooting

#### pybind11-dev not found
```
wget http://archive.ubuntu.com/ubuntu/pool/universe/p/pybind11/pybind11-dev_2.2.4-2_all.deb
sudo apt install ./pybind11-dev_2.2.4-2_all.deb
```

