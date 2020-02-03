# Baseline: SAC Non-HRL
python -m rl.main --log_root_dir ./logs --prefix baseline.ppo.pixel.v1 --max_global_step 40000000 --policy cnn --env reacher-pixel-v0 --gpu 0 --rl_hid_size 512 --max_grad_norm 0.5 --entropy_loss_coeff 0.01 --algo ppo
