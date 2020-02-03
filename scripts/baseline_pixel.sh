# Baseline: SAC Non-HRL
python -m rl.main --log_root_dir ./logs --prefix baseline.sac_pixel.clip_grad_actor.v3 --max_global_step 40000000 --policy cnn --env reacher-pixel-v0 --gpu 0 --rl_hid_size 512 --lr_actor 1e-4 --lr_critic 1e-4  --max_grad_norm 5.
