#!/bin/bash

workers="8"
prefix="baseline.ppo.ent.0.large.batch"
max_global_step="60000000"
env="simple-reacher-obstacle-v0"
gpu="0"
rl_hid_size="128"
max_episode_step="150"
evaluate_interval="1"
max_grad_norm="0.5"
entropy_loss_coef="0"
buffer_size="8192"
num_batches="48"
lr_actor="6e-4"
lr_critic="6e-4"
debug="False"
rollout_length="6000"
batch_size="512"
clip_param="0.2"
rl_activation="tanh"
algo='ppo'


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
    --algo $algo

