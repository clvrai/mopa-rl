# Baseline: SAC Non-HRL
python -m rl.main --log_root_dir ./logs --prefix baseline.sac --max_global_step 6000000 --env reacher-v0 --gpu 1 --algo sac --buffer_size 100000
