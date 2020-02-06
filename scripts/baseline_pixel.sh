# Baseline: SAC Non-HRL -- --
python -m rl.main --log_root_dir ./logs --prefix baseline.sac.pixel.ae.v8 --max_global_step 40000000 --policy cnn --env reacher-pixel-v0 --gpu 0 --rl_hid_size 64 --algo sac --use_ae True --is_rgb True
