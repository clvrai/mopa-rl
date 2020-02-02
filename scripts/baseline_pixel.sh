# Baseline: SAC Non-HRL
python -m rl.main --log_root_dir ./logs --prefix baseline.sac_pixel_without_normalization --max_global_step 10000000 --policy cnn --env reacher-pixel-v0 --gpu 0
