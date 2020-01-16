# hrl-planner

## Prerequisites
- Python 3.7
- torch
- gym 0.15.4
- [MuJoCo 2.0.2.5 ](http://www.mujoco.org/)

## Usage
### Test environment
```
python -m env.test_env 
# or 
sh ./scripts/test_env
```

### Baseline
```
sh ./scripts/baseline.sh
```

## Trouble shooting

#### pybind11-dev not found
```
wget http://archive.ubuntu.com/ubuntu/pool/universe/p/pybind11/pybind11-dev_2.2.4-2_all.deb
sudo apt install ./pybind11-dev_2.2.4-2_all.deb
```

