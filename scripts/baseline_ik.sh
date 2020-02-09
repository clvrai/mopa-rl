python -m rl.main --env reacher-v0 --log_root_dir ./logs --prefix baseline.mp.ik.td3.v1 --max_global_step 10000000 --ll_type mp --planner_type rrt --planner_objective state_const_integral --range 15.0 --threshold 0.05 --timelimit 1.5  --gpu 2 --max_mp_steps 150 --buffer_size 50000 --use_ik True  --algo td3
