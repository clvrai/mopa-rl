#!/bin/bash -x
gpu=$1
seed=$2
algo='sac'
prefix="BASELINE.SAC.ac_range1.0.v8"
env="pusher-obstacle-hard-v3"
max_episode_step="400"
debug="False"
reward_type='sparse'
# log_root_dir="./logs"
log_root_dir="/data/jun/projects/hrl-planner/logs"
reward_scale='10.'
vis_replay="True"
success_reward='150.'
expand_ac_space="True"
action_range='1.0'
use_smdp_update="True"

python -m rl.main \
    --log_root_dir $log_root_dir \
    --wandb True \
    --prefix $prefix \
    --env $env \
    --gpu $gpu \
    --max_episode_step $max_episode_step \
    --debug $debug \
    --algo $algo \
    --seed $seed \
    --reward_type $reward_type \
    --reward_scale $reward_scale \
    --vis_replay $vis_replay \
    --success_reward $success_reward \
    --expand_ac_space $expand_ac_space \
    --action_range $action_range \
    --use_smdp_update $use_smdp_update
