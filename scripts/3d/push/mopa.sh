#<!/bin/bash -x

prefix="MoPA-SAC.scale1.0.range.0.5.omega0.5.reuse5.v7"
gpu=$1
seed=$2
algo='sac'
env="sawyer-push-obstacle-v2"
max_episode_step="250"
debug="False"
reward_type='sparse'
log_root_dir="/data/jun/projects/hrl-planner/logs"
# log_root_dir="./logs"
mopa="True"
reuse_data="True"
action_range="0.5"
omega='0.5'
stochastic_eval="True"
find_collision_free="True"
vis_replay="True"
plot_type='3d'
ac_space_type="piecewise"
use_smdp_update="True"
use_discount_meta="True"
step_size="0.02"
success_reward="150.0"
max_reuse_data='5'
reward_scale="1.0"
log_indiv_entropy="True"
evaluate_interval="10000"
is_simplified='False'


# variants


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
    --omega $omega \
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
    --is_simplified $is_simplified
