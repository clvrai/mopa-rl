# Baseline: SAC Non-HRL
python -m rl.main --log_root_dir ./logs --prefix baseline.td3.v8.smooth_target --max_global_step 6000000 --env reacher-v0 --gpu 0 --algo td3 --buffer_size 100000 --actor_update_freq 2 --rl_hid_size 512
