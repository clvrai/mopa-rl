#/!/bin/bash -x
gpu=$1
seed=$2

algo='sac'
prefix="MoPA-SAC"
env="pusher-obstacle-hard-v3"
max_episode_step="400"
debug="False"
reward_type='sparse'
# log_root_dir="./logs"
log_root_dir="/data/jun/projects/hrl-planner/logs"
mopa="True"
reward_scale="0.2"
reuse_data="True"
action_range="1.0"
omega='0.5'
use_smdp_update="True"
stochastic_eval="True"
find_collision_free="True"
max_reuse_data='30'
ac_space_type="piecewise"
success_reward="150.0"
log_indiv_entropy="True"


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
    --mopa $mopa \
    --reward_scale $reward_scale \
    --reuse_data $reuse_data \
    --action_range $action_range \
    --omega $omega \
    --success_reward $success_reward \
    --stochastic_eval $stochastic_eval \
    --find_collision_free $find_collision_free \
    --max_reuse_data $max_reuse_data \
    --ac_space_type $ac_space_type \
    --log_indiv_entropy $log_indiv_entropy \
    --use_smdp_update $use_smdp_update
