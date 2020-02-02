python -m rl.main --env reacher-v0 --hrl True --log_root_dir ./logs --prefix baseline.reacher-v0.motion_planner.test --max_global_step 10000000 --meta_update_target both --ll_type mp --planner_type rrt --planner_objective state_const_integral --range 15 --threshold 0.1 --timelimit 2. --hl_type subgoal --gpu 0 --max_mp_steps 50 --max_episode_steps 200
# --debug True  
