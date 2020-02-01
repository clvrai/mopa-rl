python -m rl.main --env reacher-pixel-v0 --hrl True --log_root_dir ./logs --prefix baseline --max_global_step 10000000 --meta_update_target both --ll_type rl --hl_type rl --policy cnn
