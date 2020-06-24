python -m rl.main --log_root_dir ./logs --wandb True --prefix 05.31.SAC.PUSHER.EXACT.RRT --max_global_step 60000000 --env simple-pusher-obstacle-hard-v0 --gpu 1 --rl_hid_size 256 --max_episode_step 200 --evaluate_interval 1000 --entropy_loss_coef 1e-3 --buffer_size 1000000 --num_batches 4 --lr_actor 3e-4 --lr_critic 3e-4 --debug False --rollout_length 1000 --batch_size 256 --clip_param 0.2 --rl_activation relu --algo sac --seed 1237 --ctrl_reward 1e-2 --reward_type dense --comment Sanity Check --start_steps 10000 --actor_num_hid_layers 2 --group 05.31.SAC.PLANNER.PUSHER.EXACT.RRT --env_debug False --log_freq 1000 --log_interval 150 --tanh True --planner_integration True --ignored_contact_geoms None,None --planner_type rrt_connect --planner_objective path_length --range 0.1 --threshold 0.01 --timelimit 1.0 --allow_manipulation_collision True --reward_scale 1.0 --subgoal_hindsight False --reuse_data_type None --relative_goal True --simple_planner_timelimit 0.02 --action_range 2.0 --ac_rl_maximum 0.5 --ac_rl_minimum -0.5 --invalid_planner_rew -0.5 --extended_action False --success_reward 150.0 --has_terminal True --allow_approximate False --allow_invalid False --use_automatic_entropy_tuning True --stochastic_eval True --alpha 0.05 --find_collision_free True --max_reuse_data 30 --min_reuse_span 20 --is_simplified True --simplified_duration 0.01 --simple_planner_type sst