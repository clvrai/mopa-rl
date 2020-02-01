# Baseline: SAC Non-HRL
python -m rl.main --log_root_dir ./logs --wandb True --prefix baseline.sac_pixel_vel2 --max_global_step 10000000 --env reacher-v0 --policy cnn --env reacher-pixel-v0
