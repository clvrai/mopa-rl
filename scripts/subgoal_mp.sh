#!/bin/bash

workers="16"
prefix="hl.ppo.inverse.v3"
hrl="True"
max_global_step="60000000"
ll_type="mp"
planner_type="sst"
planner_objective="state_const_integral"
range="1.0"
threshold="0.5"
timelimit="0.2"
env="simple-reacher-obstacle-v0"
hl_type="subgoal"
gpu="0"
rl_hid_size="128"
meta_update_target="both"
hrl_network_to_update="HL"
max_episode_step="150"
evaluate_interval="1"
meta_tanh_policy="True"
meta_subgoal_rew="-2"
max_meta_len="15"
max_grad_norm="0.5"
entropy_loss_coef="0.01"
buffer_size="4096"
num_batches="16"
lr_actor="6e-4"
lr_critic="6e-4"
debug="False"
rollout_length="9450"
batch_size="256"
clip_param="0.2"
rl_activation="tanh"
reward_type='inverse'
reward_coef='400'
comment='check ho the num batces and rolout length makes the training stable'
seed='1234'
ctrl_reward_coef='10'
pos_reward_coef='10'
inv_reward='10.'

#mpiexec -n $workers
python -m rl.main --log_root_dir ./logs \
    --wandb True \
    --prefix $prefix \
    --max_global_step $max_global_step \
    --hrl $hrl \
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
    --max_episode_step $max_episode_step \
    --evaluate_interval $evaluate_interval \
    --meta_tanh_policy $meta_tanh_policy \
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
    --max_grad_norm $max_grad_norm \
    --rl_activation $rl_activation \
    --reward_type $reward_type \
    --reward_coef $reward_coef \
    --comment $comment \
    --seed $seed \
    --ctrl_reward_coef $ctrl_reward_coef \
    --pos_reward_coef $pos_reward_coef \
    --inv_reward $inv_reward
