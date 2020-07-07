#<!/bin/bash -x

prefix="SAC.PLANNER.AUGMENTED.piecewise0.7.ac_range1.0.reuse.hindsight.alpha0.2.v4"
gpu=$1
seed=$2
algo='sac'
rollout_length="1000"
evaluate_interval="1000"
ckpt_interval='200000'
rl_activation="relu"
num_batches="1"
log_interval="1000"
max_global_step="60000000"
env="sawyer-peg-insertion-obstacle-v2"
max_episode_step="300"
buffer_size="1000000"
debug="False"
batch_size="256"
reward_type='dense'
comment='Sanity Check'
start_steps='10000'
actor_num_hid_layers='2'
# log_root_dir="/data/jun/projects/hrl-planner/logs"
log_root_dir="./logs"
log_freq='1000'
planner_integration="True"
planner_type="rrt_connect"
planner_objective="path_length"
range="0.1"
simple_planner_range="0.05"
threshold="0.0"
timelimit="1.0"
allow_manipulation_collision="True"
alpha="0.2"
reuse_data_type="random"
action_range="1.0"
ac_rl_minimum="-0.7"
ac_rl_maximum="0.7"
invalid_planner_rew="-0.3"
extended_action="False"
stochastic_eval="True"
find_collision_free="True"
use_double_planner="False"
simple_planner_type='rrt_connect'
simple_planner_timelimit="0.05"
is_simplified="False"
simplified_duration="0.01"
simple_planner_simplified="False"
simple_planner_simplified_duration="0.01"
vis_replay="True"
vis_replay_interval="10000"
use_interpolation="True"
interpolate_type="simple"
joint_margin="0.001"
task_level='easy'
use_cum_rew="True"
plot_type='3d'
contact_threshold="-0.002"
ac_space_type="piecewise"
use_smdp_update="True"
use_discount_meta="True"
temperature="1.0"
step_size="0.02"
success_reward="150.0"
add_curr_rew="True"
discount_factor='1.0'
max_reuse_data='15'
min_reuse_span='20'
hindsight_transition="True"
# max_grad_norm='0.5'

python -m rl.main \
    --log_root_dir $log_root_dir \
    --wandb True \
    --prefix $prefix \
    --max_global_step $max_global_step \
    --env $env \
    --gpu $gpu \
    --max_episode_step $max_episode_step \
    --evaluate_interval $evaluate_interval \
    --buffer_size $buffer_size \
    --num_batches $num_batches \
    --debug $debug \
    --rollout_length $rollout_length \
    --batch_size $batch_size \
    --rl_activation $rl_activation \
    --algo $algo \
    --seed $seed \
    --reward_type $reward_type \
    --comment $comment \
    --start_steps $start_steps \
    --actor_num_hid_layers $actor_num_hid_layers \
    --log_freq $log_freq \
    --log_interval $log_interval \
    --planner_integration $planner_integration \
    --planner_type $planner_type \
    --planner_objective $planner_objective \
    --range $range \
    --threshold $threshold \
    --timelimit $timelimit \
    --allow_manipulation_collision $allow_manipulation_collision \
    --alpha $alpha \
    --reuse_data_type $reuse_data_type \
    --simple_planner_timelimit $simple_planner_timelimit \
    --action_range $action_range \
    --ac_rl_maximum $ac_rl_maximum \
    --ac_rl_minimum $ac_rl_minimum \
    --invalid_planner_rew $invalid_planner_rew \
    --extended_action $extended_action \
    --stochastic_eval $stochastic_eval \
    --find_collision_free $find_collision_free \
    --use_double_planner $use_double_planner \
    --simple_planner_type $simple_planner_type \
    --is_simplified $is_simplified \
    --simplified_duration $simplified_duration \
    --simple_planner_simplified $simple_planner_simplified \
    --simple_planner_simplified_duration $simple_planner_simplified_duration \
    --vis_replay $vis_replay \
    --use_interpolation $use_interpolation \
    --interpolate_type $interpolate_type \
    --joint_margin $joint_margin \
    --task_level $task_level \
    --use_cum_rew $use_cum_rew \
    --plot_type $plot_type \
    --vis_replay_interval $vis_replay_interval \
    --use_smdp_update $use_smdp_update \
    --contact_threshold $contact_threshold \
    --ac_space_type $ac_space_type \
    --use_discount_meta $use_discount_meta \
    --temperature $temperature \
    --step_size $step_size \
    --success_reward $success_reward \
    --add_curr_rew $add_curr_rew \
    --discount_factor $discount_factor  \
    --max_reuse_data $max_reuse_data \
    --min_reuse_span $min_reuse_span \
    --hindsight_transition $hindsight_transition
