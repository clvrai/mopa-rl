# Baseline: SAC Non-HRL -- --
#python -m rl.main --log_root_dir ./logs --prefix baseline.sac.pixel.ae.v12 --max_global_step 40000000 --policy cnn --env reacher-pixel-v0 --gpu 0 --rl_hid_size 128 --algo sac --use_ae True --is_rgb True --buffer_size 10000 --lr_critic 0.001 --lr_actor 0.001 
python -m rl.main --log_root_dir ./logs --prefix baseline.sac.pixel.ae.hid256 --max_global_step 40000000 --policy cnn --env reacher-pixel-v0 --gpu 0 --rl_hid_size 256 --algo sac --use_ae True --is_rgb True --buffer_size 50000 --actor_update_freq 2
