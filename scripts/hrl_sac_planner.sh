#!/bin/bash

workers="1"
prefix="hl.sac.her.rew.0.5.single.150.debug"
hrl="True"
max_global_step="60000000"
ll_type="mix"
planner_type="sst"
planner_objective="state_const_integral"
range="1.0"
threshold="0.5"
timelimit="0.2"
env="simple-pusher-v0"
hl_type="subgoal"
gpu="3"
rl_hid_size="256"
meta_update_target="HL"
hrl_network_to_update="HL"
max_episode_steps="150"
evaluate_interval="5"
meta_subgoal_rew="-0.5"
max_meta_len="15"
entropy_loss_coef="0.01"
buffer_size="150000"
num_batches="150"
lr_actor="3e-4"
lr_critic="3e-4"
debug="True"
rollout_length="15000"
batch_size="256"
clip_param="0.2"
reward_type="composition"
reward_scale="1"
comment="Fix min and max of subgoal"
seed="1234"
ctrl_reward_coef="1e-3"
primitive_skills="mp push_max_step30"
primitive_dir="primitives"
actor_num_hid_layers="1"
subgoal_type="cart"
goal_replace="True"
subgoal_reward="True"
relative_subgoal="True"
meta_algo='sac'
her='True'

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
    --hl_type $hl_type \
    --gpu $gpu \
    --rl_hid_size $rl_hid_size \
    --meta_update_target $meta_update_target \
    --hrl_network_to_update $hrl_network_to_update \
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
    --primitive_skills $primitive_skills \
    --primitive_dir $primitive_dir \
    --actor_num_hid_layers $actor_num_hid_layers \
    --subgoal_type $subgoal_type \
    --goal_replace $goal_replace \
    --subgoal_reward $subgoal_reward \
    --relative_subgoal $relative_subgoal \
    --meta_algo $meta_algo \
    --her $her
