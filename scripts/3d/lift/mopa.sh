#<!/bin/bash -x

prefix="SAC.PLANNER.AUGMENTED.reuse45.scale0.2"
env="sawyer-lift-obstacle-v0"
gpu=$1
seed=$2
algo='sac'
max_episode_step="250"
debug="False"
reward_type='sparse'
# log_root_dir="/data/jun/projects/hrl-planner/logs"
log_root_dir="./logs"
planner_integration="True"
reuse_data_type="random"
action_range="0.5"
ac_rl_minimum="-0.5"
ac_rl_maximum="0.5"
stochastic_eval="True"
find_collision_free="True"
vis_replay="True"
plot_type='3d'
ac_space_type="piecewise"
use_smdp_update="True"
use_discount_meta="True"
step_size="0.02"
success_reward="150.0"
add_curr_rew="True"
max_reuse_data='45'
reward_scale="0.2"
log_indiv_entropy="True"
evaluate_interval="10000"

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
    --planner_integration $planner_integration \
    --reuse_data_type $reuse_data_type \
    --action_range $action_range \
    --ac_rl_maximum $ac_rl_maximum \
    --ac_rl_minimum $ac_rl_minimum \
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
