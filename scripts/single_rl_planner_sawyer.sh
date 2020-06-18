#!/bin/bash -x
gpu=$1
seed=$2

algo='sac'
rollout_length="1000"
evaluate_interval="1000"
ckpt_interval='200000'
rl_activation="relu"
num_batches="4"
log_interval="1000"

workers="1"
tanh="True"
prefix="SAC.SAWYAER.PLANNER.AUGMENTED.bigger.box.fix_xml.sacle.1.thresh.0.5.sm_margin"
max_global_step="60000000"
env="sawyer-push-v0"
rl_hid_size="256"
max_episode_step="250"
entropy_loss_coef="1e-3"
buffer_size="1000000"
lr_actor="3e-4"
lr_critic="3e-4"
debug="False"
batch_size="256"
clip_param="0.2"
ctrl_reward='1e-2'
reward_type='dense'
comment='Sanity Check'
start_steps='10000'
actor_num_hid_layers='2'
log_root_dir="/data/jun/projects/hrl-planner/logs"
# log_root_dir="./logs"
env_debug='False'
log_freq='1000'
planner_integration="True"
ignored_contact_geoms='None,None'
planner_type="rrt_connect"
planner_objective="path_length"
range="0.2"
simple_planner_range="0.1"
threshold="0.0"
timelimit="1.0"
allow_manipulation_collision="True"
reward_scale="1.0"
subgoal_hindsight="False"
reuse_data_type="None"
relative_goal="True"
action_range="1.0"
ac_rl_minimum="-0.5"
ac_rl_maximum="0.5"
invalid_planner_rew="-0.5"
extended_action="False"
allow_approximate="False"
allow_invalid="False"
use_automatic_entropy_tuning="True"
stochastic_eval="True"
alpha='0.05'
find_collision_free="True"
use_double_planner="False"
simple_planner_type='rrt_connect'
simple_planner_timelimit="0.02"
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
use_smdp_update="True"
actor_bias="-10"
# max_grad_norm='0.5'

#mpiexec -n $workers
python -m rl.main \
    --log_root_dir $log_root_dir \
    --wandb True \
    --prefix $prefix \
    --max_global_step $max_global_step \
    --env $env \
    --gpu $gpu \
    --rl_hid_size $rl_hid_size \
    --max_episode_step $max_episode_step \
    --evaluate_interval $evaluate_interval \
    --entropy_loss_coef $entropy_loss_coef \
    --buffer_size $buffer_size \
    --num_batches $num_batches \
    --lr_actor $lr_actor \
    --lr_critic $lr_critic \
    --debug $debug \
    --rollout_length $rollout_length \
    --batch_size $batch_size \
    --clip_param $clip_param \
    --rl_activation $rl_activation \
    --algo $algo \
    --seed $seed \
    --ctrl_reward $ctrl_reward \
    --reward_type $reward_type \
    --comment $comment \
    --start_steps $start_steps \
    --actor_num_hid_layers $actor_num_hid_layers \
    --env_debug $env_debug \
    --log_freq $log_freq \
    --log_interval $log_interval \
    --tanh $tanh \
    --planner_integration $planner_integration \
    --ignored_contact_geoms $ignored_contact_geoms \
    --planner_type $planner_type \
    --planner_objective $planner_objective \
    --range $range \
    --threshold $threshold \
    --timelimit $timelimit \
    --allow_manipulation_collision $allow_manipulation_collision \
    --reward_scale $reward_scale \
    --subgoal_hindsight $subgoal_hindsight \
    --reuse_data_type $reuse_data_type \
    --relative_goal $relative_goal \
    --simple_planner_timelimit $simple_planner_timelimit \
    --action_range $action_range \
    --ac_rl_maximum $ac_rl_maximum \
    --ac_rl_minimum $ac_rl_minimum \
    --invalid_planner_rew $invalid_planner_rew \
    --extended_action $extended_action \
    --allow_approximate $allow_approximate \
    --allow_invalid $allow_invalid \
    --use_automatic_entropy_tuning $use_automatic_entropy_tuning \
    --stochastic_eval $stochastic_eval \
    --alpha $alpha \
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
    # --actor_bias $actor_bias
