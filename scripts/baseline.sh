#!/bin/bash

workers="8"
prefix="baseline.ppo.dense.pos_reward.10"
max_global_step="60000000"
env="simple-pusher-v0"
gpu="0"
rl_hid_size="128"
max_episode_step="150"
evaluate_interval="1"
max_grad_norm="0.5"
entropy_loss_coef="0.1"
buffer_size="100000"
num_batches="32"
lr_actor="6e-4"
lr_critic="6e-4"
debug="False"
rollout_length="2048"
batch_size="256"
clip_param="0.2"
rl_activation="tanh"
algo='ppo'
seed='1234'
ctrl_reward='1e-3'
reward_type='dense'
comment='PPO baseline for pusher env'
reward_coef='400'
pos_reward_coef='10'

mpiexec -n $workers python -m rl.main --log_root_dir ./logs \
    --wandb True \
    --prefix $prefix \
    --max_global_step $max_global_step \
    --env $env \
    --gpu $gpu \
    --rl_hid_size $rl_hid_size \
    --max_episode_step $max_episode_step \
    --evaluate_interval $evaluate_interval \
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
    --algo $algo \
    --seed $seed \
    --ctrl_reward $ctrl_reward \
    --reward_type $reward_type \
    --comment $comment \
    --reward_coef $reward_coef \
    --pos_reward_coef $pos_reward_coef

