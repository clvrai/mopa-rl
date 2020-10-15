#!/bin/bash -x
gpu=$1
seed=$2
prefix="BASELINE.action_range0.5.v8"
env="SawyerLiftObstacle-v0"
algo='sac'
max_episode_step="250"
debug="False"
reward_type='sparse'
log_root_dir="./logs"
vis_replay="True"
plot_type='3d'
success_reward='150.'
reward_scale="10."
expand_ac_space='True'
use_smdp_update="True"
action_range="0.5"

python -m rl.main \
    --log_root_dir $log_root_dir \
    --prefix $prefix \
    --env $env \
    --gpu $gpu \
    --max_episode_step $max_episode_step \
    --debug $debug \
    --algo $algo \
    --seed $seed \
    --reward_type $reward_type \
    --vis_replay $vis_replay \
    --plot_type $plot_type \
    --success_reward $success_reward \
    --reward_scale $reward_scale \
    --action_range $action_range \
    --expand_ac_space $expand_ac_space \
    --use_smdp_update $use_smdp_update 
