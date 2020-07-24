#<!/bin/bash -x

prefix="SAC.PLANNER.AUGMENTED.discrete.ac_rang0.5.scale.0.2.no_discount"
gpu=$1
seed=$2
algo='sac'
rl_activation="relu"
num_batches="1"
log_interval="1000"
env="sawyer-peg-insertion-obstacle-v2"
max_episode_step="250"
debug="False"
batch_size="256"
reward_type='sparse'
comment='Sanity Check'
# log_root_dir="/data/jun/projects/hrl-planner/logs"
log_root_dir="./logs"
log_freq='1000'
planner_integration="True"
allow_manipulation_collision="False"
alpha="1.0"
reuse_data_type="random"
action_range="0.5"
ac_rl_minimum="-0.5"
ac_rl_maximum="0.5"
invalid_planner_rew="-0.0"
stochastic_eval="True"
find_collision_free="True"
use_double_planner="False"
vis_replay="True"
use_cum_rew="True"
plot_type='3d'
ac_space_type="normal"
use_smdp_update="True"
# use_discount_meta="False"
step_size="0.02"
success_reward="150.0"
add_curr_rew="True"
discount_factor='0.99'
max_reuse_data='30'
min_reuse_span='20'
reward_scale="0.2"
log_indiv_entropy="True"
evaluate_interval="10000"

#variants
extended_action="True"

python -m rl.main \
    --log_root_dir $log_root_dir \
    --wandb True \
    --prefix $prefix \
    --env $env \
    --gpu $gpu \
    --max_episode_step $max_episode_step \
    --num_batches $num_batches \
    --debug $debug \
    --batch_size $batch_size \
    --rl_activation $rl_activation \
    --algo $algo \
    --seed $seed \
    --reward_type $reward_type \
    --comment $comment \
    --log_freq $log_freq \
    --log_interval $log_interval \
    --planner_integration $planner_integration \
    --allow_manipulation_collision $allow_manipulation_collision \
    --alpha $alpha \
    --reuse_data_type $reuse_data_type \
    --action_range $action_range \
    --ac_rl_maximum $ac_rl_maximum \
    --ac_rl_minimum $ac_rl_minimum \
    --invalid_planner_rew $invalid_planner_rew \
    --extended_action $extended_action \
    --stochastic_eval $stochastic_eval \
    --find_collision_free $find_collision_free \
    --use_double_planner $use_double_planner \
    --vis_replay $vis_replay \
    --use_cum_rew $use_cum_rew \
    --plot_type $plot_type \
    --use_smdp_update $use_smdp_update \
    --ac_space_type $ac_space_type \
    --step_size $step_size \
    --success_reward $success_reward \
    --add_curr_rew $add_curr_rew \
    --discount_factor $discount_factor  \
    --max_reuse_data $max_reuse_data \
    --min_reuse_span $min_reuse_span \
    --reward_scale $reward_scale \
    --log_indiv_entropy $log_indiv_entropy \
    --evaluate_interval $evaluate_interval \
    # --use_discount_meta $use_discount_meta \
