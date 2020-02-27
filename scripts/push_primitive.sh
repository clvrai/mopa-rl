#!/bin/bash

workers="16"
prefix="ll.push.primitive.sac.dense.pos_reward_10.ctrl.0.1"
hrl="True"
max_global_step="60000000"
ll_type="rl"
env="pusher-push-v0"
gpu="2"
rl_hid_size="128"
meta_update_target="both"
hrl_network_to_update="LL"
max_episode_step="50"
max_meta_len="15"
evaluate_interval="10"
meta_tanh_policy="True"
max_grad_norm="0.5"
entropy_loss_coef="0.01"
buffer_size="20000"
num_batches="50"
lr_actor="6e-4"
lr_critic="6e-4"
debug="False"
rollout_length="1000"
batch_size="256"
clip_param="0.2"
rl_activation="relu"
reward_type='dense'
reward_coef='400'
comment='Train primitive policy'
seed='1234'
ctrl_reward_coef='0.1'
pos_reward_coef='10.'


#mpiexec -n $workers
python -m rl.main --log_root_dir ./logs \
    --wandb True \
    --prefix $prefix \
    --max_global_step $max_global_step \
    --hrl $hrl \
    --ll_type $ll_type \
    --env $env \
    --gpu $gpu \
    --rl_hid_size $rl_hid_size \
    --meta_update_target $meta_update_target \
    --hrl_network_to_update $hrl_network_to_update \
    --max_episode_step $max_episode_step \
    --evaluate_interval $evaluate_interval \
    --meta_tanh_policy $meta_tanh_policy \
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
    --pos_reward_coef $pos_reward_coef