# HRL baseline
python -m rl.main --log_root_dir ./logs --wandb False --prefix baseline.hrl_reacher-test --max_global_step 6000000 --hrl True --meta_update_target both --env reacher-v0 --debug True --primitive_skills skill1 skill2 skill3 --gpu 0
