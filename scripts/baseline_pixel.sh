# Baseline: SAC Non-HRL
python -m rl.main --log_root_dir ./logs --prefix baseline.sac_pixel_4_layers --max_global_step 10000000 --policy cnn --env reacher-pixel-v0 --gpu 0 --rl_hid_size 512
