# Baseline: SAC Non-HRL
python -m rl.main --log_root_dir ./logs --prefix baseline.ddpg.v5.debug --max_global_step 6000000 --env reacher-v0 --gpu 0 --algo ddpg --lr_actor 1e-3 --lr_critic 1e-3 --buffer_size 100000 --debug True
