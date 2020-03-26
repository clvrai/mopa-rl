#!/bin/bash

workers="8"
prefix="hrl.subgoal_predictor.debug"
hrl="True"
max_global_step="60000000"
ll_type="mix"
planner_type="sst"
planner_objective="state_const_integral"
range="1.0"
threshold="0.5"
timelimit="0.2"
env="simple-pusher-v0"
gpu="1"
rl_hid_size="256"
meta_update_target="both"
max_episode_steps="150"
evaluate_interval="100"
meta_subgoal_rew="-0.5"
max_meta_len="1"
entropy_loss_coef="0.01"
buffer_size="120000"
num_batches="1"
lr_actor="3e-4"
lr_critic="3e-4"
debug="True"
rollout_length="15000"
batch_size="512"
clip_param="0.2"
reward_type="composition"
reward_scale="3"
comment="debug"
ctrl_reward_coef="1e-1"
actor_num_hid_layers="2"
subgoal_type="joint"
subgoal_reward="True"
relative_subgoal="True"
meta_algo='sac'
start_steps='150000'
distance_threshold='0.06'
success_reward='1.'
primitive_skills="mp push"
subgoal_predictor="True"
seed="1234"

#mpiexec -n $workers
python -m rl.main --log_root_dir ./logs \
    --wandb True \
    --prefix $prefix \
    --hrl $hrl \
    --max_global_step $max_global_step \
    --ll_type $ll_type \
    --planner_type $planner_type \
    --planner_objective $planner_objective \
    --range $range \
    --threshold $threshold \
    --timelimit $timelimit \
    --env $env \
    --gpu $gpu \
    --rl_hid_size $rl_hid_size \
    --meta_update_target $meta_update_target \
    --max_episode_steps $max_episode_steps \
    --evaluate_interval $evaluate_interval \
    --meta_subgoal_rew $meta_subgoal_rew \
    --max_meta_len $max_meta_len \
    --entropy_loss_coef $entropy_loss_coef \
    --buffer_size $buffer_size \
    --num_batches $num_batches \
    --lr_actor $lr_actor \
    --lr_critic $lr_critic \
    --debug $debug \
    --rollout_length $rollout_length \
    --batch_size $batch_size \
    --clip_param $clip_param \
    --reward_type $reward_type \
    --reward_scale $reward_scale \
    --comment $comment \
    --seed $seed \
    --ctrl_reward_coef $ctrl_reward_coef \
    --actor_num_hid_layers $actor_num_hid_layers \
    --subgoal_type $subgoal_type \
    --subgoal_reward $subgoal_reward \
    --relative_subgoal $relative_subgoal \
    --meta_algo $meta_algo \
    --start_steps $start_steps \
    --distance_threshold $distance_threshold \
    --success_reward $success_reward \
    --primitive_skills $primitive_skills \
    --subgoal_predictor $subgoal_predictor
