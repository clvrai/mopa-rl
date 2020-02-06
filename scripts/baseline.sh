# Baseline: SAC Non-HRL
python -m rl.main --log_root_dir ./logs --prefix baseline.td3.v1 --max_global_step 6000000 --env reacher-v0 --gpu 0 --algo ddpg 
