# Baseline: SAC Non-HRL
mpiexec -n 20 python -m rl.main --log_root_dir ./logs --prefix baseline.ppo.512 --max_global_step 6000000 --env simple-reacher-v0 --gpu 1 --rl_hid_size 512 --algo ppo
