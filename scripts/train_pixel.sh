python -m rl.main --env reacher-obstacle-pixel-v0 --hrl True --log_root_dir ./logs --wandb False --prefix pixexl_reacher-obstacle_baseline --max_global_step 10000000 --meta_update_target both --ll_type rl --hl_type subgoal --policy cnn  --debug True