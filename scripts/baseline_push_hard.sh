#!/bin/bash -x
gpu=$1
seed=$2
algo='sac'
evaluate_interval="9000"
ckpt_interval='100000'
rl_activation="relu"
num_batches="1"
log_interval="1000"

workers="1"
prefix="BASELINE.SAC.SPARSE.v2"
max_global_step="60000000"
env="pusher-obstacle-hard-v0"
gpu=$gpu
rl_hid_size="256"
max_episode_step="300"
lr_actor="3e-4"
lr_critic="3e-4"
debug="False"
batch_size="256"
reward_type='sparse'
comment='Baseline'
start_steps='10000'
# log_root_dir="./logs"
log_root_dir="/data/jun/projects/hrl-planner/logs"
log_freq='1000'
reward_scale='5.'
alpha='1.0'
vis_replay="True"
success_reward='150.'
# max_grad_norm='0.5'

#mpiexec -n $workers
python -m rl.main \
    --log_root_dir $log_root_dir \
    --wandb True \
    --prefix $prefix \
    --max_global_step $max_global_step \
    --env $env \
    --gpu $gpu \
    --rl_hid_size $rl_hid_size \
    --max_episode_step $max_episode_step \
    --evaluate_interval $evaluate_interval \
    --num_batches $num_batches \
    --lr_actor $lr_actor \
    --lr_critic $lr_critic \
    --debug $debug \
    --batch_size $batch_size \
    --rl_activation $rl_activation \
    --algo $algo \
    --seed $seed \
    --reward_type $reward_type \
    --comment $comment \
    --start_steps $start_steps \
    --log_freq $log_freq \
    --log_interval $log_interval \
    --reward_scale $reward_scale \
    --vis_replay $vis_replay \
    --success_reward $success_reward \
    --alpha $alpha
