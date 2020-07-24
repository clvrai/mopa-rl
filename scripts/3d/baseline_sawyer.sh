#!/bin/bash -x
gpu=$1
seed=$2

prefix="BASELINE.test"
env="sawyer-assembly-v0"
algo='sac'
rollout_length="10000"
evaluate_interval="10000"
ckpt_interval='100000'
rl_activation="relu"
num_batches="1"
log_interval="1000"
max_global_step="1500000"
max_episode_step="250"
buffer_size="1000000"
debug="False"
batch_size="256"
reward_type='sparse'
comment='Baseline'
start_steps='10000'
# log_root_dir="/data/jun/projects/hrl-planner/logs"
log_root_dir="./logs"
log_freq='1000'
alpha="1.0"
vis_replay="True"
plot_type='3d'
task_level='easy'
success_reward='150.'
reward_scale="10."
use_ik_target="False"
ik_target="grip_site"
action_range="0.02"

python -m rl.main \
    --log_root_dir $log_root_dir \
    --wandb True \
    --prefix $prefix \
    --max_global_step $max_global_step \
    --env $env \
    --gpu $gpu \
    --max_episode_step $max_episode_step \
    --evaluate_interval $evaluate_interval \
    --buffer_size $buffer_size \
    --num_batches $num_batches \
    --debug $debug \
    --rollout_length $rollout_length \
    --batch_size $batch_size \
    --rl_activation $rl_activation \
    --algo $algo \
    --seed $seed \
    --reward_type $reward_type \
    --comment $comment \
    --start_steps $start_steps \
    --log_freq $log_freq \
    --log_interval $log_interval \
    --alpha $alpha \
    --vis_replay $vis_replay \
    --task_level $task_level \
    --plot_type $plot_type \
    --success_reward $success_reward \
    --reward_scale $reward_scale \
    --use_ik_target $use_ik_target \
    --ckpt_interval $ckpt_interval \
    --ik_target $ik_target \
    --action_range $action_range
