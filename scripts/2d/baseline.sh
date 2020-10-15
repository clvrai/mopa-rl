#!/bin/bash -x
gpu=$1
seed=$2
algo='sac'
prefix="BASELINE.SAC"
env="PusherObstacle-v0"
max_episode_step="400"
debug="True"
log_root_dir="./logs"
reward_scale='10.'
vis_replay="True"
success_reward='150.'

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
