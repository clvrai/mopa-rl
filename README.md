# hrl-planner

## Prerequisites
- Python 3.7 (`python3.7`, `python3.7-dev`)
- torch
- `libyaml-cpp-dev` (`sudo apt install libyaml-cpp-dev` or `brew install libyaml yaml-cpp`)
- gym 0.15.4
- [MuJoCo 2.0.2.5 ](http://www.mujoco.org/)

## Installation 
It's recommended to use a virtualenv or conda environment
```
virtualenv --python /path/to/python3.7 --no-site-packages <envname> #e.g. hrlenv
source <envname>/bin/activate
```

### Clone repo
Create project folder, you'll need it for other packages like `ompl`
```
mkdir HRLPlanner
cd HRLPLanner/
git clone git@github.com:youngwoon/hrl-planner.git
# install required python packages in your new env
pip install -r requirements.txt
```

### Install ompl
#### Prerequisite: Install Eigen first
Linux install

```
sh ./scripts/installEigen.sh #from the home directory
```
MacOS install

```
brew install eigen
```
#### Continue to install OMPL
Go to the repo home directory
```
git clone git@github.com:ompl/ompl.git ../ompl
cd ../ompl
cmake .
sudo make install
```

### OMPL-mujoco wrapper 

macOS users: ensure DYLD_LIBRARY_PATH is set to the mujoco bin folder
```
export DYLD_LIBRARY_PATH=/Users/gautam/.mujoco/mujoco200/bin:$DYLD_LIBRARY_PATH
```

- Compile cython for ompl-mujoco wrapper

```
cd ./motion_planners
# Note that you need to set `prefix_path` variable in setup.py
# You also need to adapt the path to ompl / eigen if you installed it at a different location
python setup.py build_ext --inplace
```
or on macOS
```
python setup_macos.py build_ext --inplace
```

### Install metaworld
```
git clone https://github.com/rlworkgroup/metaworld.git
cd metaworld
pip install -e .
```

## Usage
### Test environment
Run this line to start the test environment. You might face [this libGL problem](https://github.com/openai/mujoco-py/issues/268) on Ubuntu.
```
cd /path/to/repo/home
python -m env.test_env 
# or 
sh ./scripts/test_env
````

### Create Robosuite xml file

The command below generates a robosuite xml file in `env/assets/xml/` folder.

#### Environments currently supported
- sawyer-pick-place-[single, milk, can, cereal]-v0
- sawyer-move-[single, milk, can, cereal]-v0

```
cd /path/to/repo/home
python -m env.create_robosuite_xml --env sawyer-pick-place-can-v0
# or 
sh ./scripts/create_robosuite_xml.sh
```


### SAC Baseline
```
sh ./scripts/baseline.sh
```

### HRL Baseline
```
sh ./scripts/hrl_baseline.sh
```

### HRL baseline with subgoals
```
sh ./scripts/hrl_subgoal_baseline.sh
```

### Subgoal + Motion planner
```
sh ./scripts/subgoal_mp.sh
```

## Trouble shooting

#### pybind11-dev not found
```
wget http://archive.ubuntu.com/ubuntu/pool/universe/p/pybind11/pybind11-dev_2.2.4-2_all.deb
sudo apt install ./pybind11-dev_2.2.4-2_all.deb
```

