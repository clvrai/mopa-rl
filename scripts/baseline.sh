# Baseline: SAC Non-HRL
python -m rl.main --log_root_dir ./logs --prefix baseline.td3.v3 --max_global_step 6000000 --env reacher-v0 --gpu 0 --algo td3 --lr_actor 1e-3 --lr_critic 1e-3 --buffer_size 10000
