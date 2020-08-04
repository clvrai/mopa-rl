#<!/bin/bash -x

prefix="MoPA=SAC.discrete"
gpu=$1
seed=$2
env="sawyer-push-obstacle-v2"
max_episode_step="250"
debug="False"
reward_type='sparse'
# log_root_dir="/data/jun/projects/hrl-planner/logs"
log_root_dir="./logs"
mopa="True"
alpha="1.0"
reuse_data="True"
action_range="1.0"
invalid_planner_rew="-0.0"
stochastic_eval="True"
find_collision_free="True"
vis_replay="True"
plot_type='3d'
ac_space_type="piecewise"
use_smdp_update="True"
use_discount_meta="True"
step_size="0.02"
success_reward="150.0"
discount_factor='0.99'
max_reuse_data='15'
reward_scale="0.5"
log_indiv_entropy="True"
evaluate_interval="10000"
discrete_action="True"


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
    --reuse_data $reuse_data \
    --action_range $action_range \
    --discrete_action $extended_action \
    --stochastic_eval $stochastic_eval \
    --find_collision_free $find_collision_free \
    --vis_replay $vis_replay \
    --plot_type $plot_type \
    --use_smdp_update $use_smdp_update \
    --ac_space_type $ac_space_type \
    --step_size $step_size \
    --success_reward $success_reward \
    --max_reuse_data $max_reuse_data \
    --reward_scale $reward_scale \
    --log_indiv_entropy $log_indiv_entropy \
    --evaluate_interval $evaluate_interval \
    --use_discount_meta $use_discount_meta \
