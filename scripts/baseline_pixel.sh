# Baseline: SAC Non-HRL
python -m rl.main --log_root_dir ./logs --prefix baseline.sac.pixel.v11 --max_global_step 40000000 --policy cnn --env reacher-pixel-v0 --gpu 0 --rl_hid_size 512 --algo sac --debug True --lr_actor 0.001 --lr_critic 0.001
