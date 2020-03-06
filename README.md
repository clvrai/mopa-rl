# hrl-planner

## Prerequisites
- Python 3.7
- torch
- gym 0.15.4
- [MuJoCo 2.0.2.5 ](http://www.mujoco.org/)

## Installation 

### Install ompl
```
sh ./scripts/installEigen.sh
git clone git@github.com:ompl/ompl.git ../ompl
cd ../ompl
cmake .
make install
```

### OMPL-mujoco wrapper 

- Install OMPL 
Follow https://ompl.kavrakilab.org/installation.html

- Compile cython for ompl-mujoco wrapper

```
cd ./motion_planners

# Note that you need to change `prefix_path` variable in setup.py
python setup.py build_ext --inlpace
```

## Usage
### Test environment
```
python -m env.test_env 
# or 
sh ./scripts/test_env
````

### Create Robosuite xml file

The command below generates a robosuite xml file in `env/assets/xml/` folder.
```
python -m env.create_robosuite_xml --env sawyer-pick-place-can-v0
# or 
sh ./scripts/craete_robosuite_xml.sh
```
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

