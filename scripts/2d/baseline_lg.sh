#!/bin/bash -x
gpu=$1
seed=$2
algo='sac'
prefix="BASELINE.SAC.LG"
env="PusherObstacle-v0"
max_episode_step="400"
debug="False"
log_root_dir="./logs"
reward_scale='10.'
vis_replay="True"
success_reward='150.'
expand_ac_space="True"
action_range='1.0'
use_smdp_update="True"

python -m rl.main \
    --log_root_dir $log_root_dir \
    --prefix $prefix \
    --env $env \
    --gpu $gpu \
    --max_episode_step $max_episode_step \
    --debug $debug \
    --algo $algo \
    --seed $seed \
    --reward_scale $reward_scale \
    --vis_replay $vis_replay \
    --success_reward $success_reward \
    --expand_ac_space $expand_ac_space \
    --action_range $action_range \
    --use_smdp_update $use_smdp_update
