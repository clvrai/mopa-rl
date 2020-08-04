#/!/bin/bash -x
gpu=$1
seed=$2
algo='sac'
prefix="MoPA-SAC.discrete"
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
stochastic_eval="True"
find_collision_free="True"
max_reuse_data='30'
use_smdp_update="True"
ac_space_type="normal"
success_reward="150.0"
log_indiv_entropy="True"
discrete_action="True"


#mpiexec -n $workers
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
    --comment $comment \
    --mopa $mopa \
    --reward_scale $reward_scale \
    --reuse_data $reuse_data \
    --action_range $action_range \
    --discrete_action $discrete_action \
    --success_reward $success_reward \
    --stochastic_eval $stochastic_eval \
    --find_collision_free $find_collision_free \
    --max_reuse_data $max_reuse_data \
    --use_smdp_update $use_smdp_update \
    --ac_space_type $ac_space_type \
    --use_double_planner $use_double_planner \
    --log_indiv_entropy $log_indiv_entropy \
