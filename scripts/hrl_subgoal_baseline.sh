#HRL subgoal baseline
python -m rl.main --log_root_dir ./logs --wandb True --prefix baseline.hrl_reacher-test --max_global_step 6000000 --hrl True --meta_update_target both --env reacher-v0 --hl_type subgoal
