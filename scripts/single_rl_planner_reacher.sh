#!/bin/bash -x
gpu=$1
seed=$2

algo='sac'
rollout_length="1000"
evaluate_interval="1000"
ckpt_interval='200000'
rl_activation="relu"
num_batches="1"
log_interval="200"

workers="1"
tanh="True"
prefix="SAC.REACHER.SPARSE.RRT.SIMPLE.INTERPOLATION.scale.10"
max_global_step="60000000"
env="simple-reacher-obstacle-hard-v0"
rl_hid_size="256"
max_episode_step="200"
entropy_loss_coef="1e-3"
buffer_size="1000000"
lr_actor="3e-4"
lr_critic="3e-4"
debug="False"
batch_size="256"
clip_param="0.2"
ctrl_reward='1e-2'
reward_type='sparse'
comment='Sanity Check'
start_steps='10000'
actor_num_hid_layers='2'
log_root_dir="/data/jun/projects/hrl-planner/logs"
env_debug='False'
log_freq='1000'
planner_integration="True"
ignored_contact_geoms='None,None'
planner_type="rrt_connect"
planner_objective="path_length"
range="0.1"
threshold="0.01"
timelimit="1.0"
allow_manipulation_collision="True"
reward_scale="10.0"
subgoal_hindsight="False"
reuse_data_type="None"
relative_goal="True"
simple_planner_type="sst"
simple_planner_timelimit="0.02"
action_range="2.0"
ac_rl_minimum="-0.5"
ac_rl_maximum="0.5"
invalid_planner_rew="-0.5"
extended_action="False"
allow_approximate="False"
success_reward='0.0'
has_terminal='True'
allow_invalid="False"
use_automatic_entropy_tuning="True"
stochastic_eval="True"
alpha='0.05'
find_collision_free="True"
max_reuse_data='30'
min_reuse_span='20'
is_simplified="False"
simplified_duration="0.02"
use_interpolation="True"
interpolate_type="simple"
construct_time="600"
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
    --success_reward $success_reward \
    --has_terminal $has_terminal \
    --allow_approximate $allow_approximate \
    --allow_invalid $allow_invalid \
    --use_automatic_entropy_tuning $use_automatic_entropy_tuning \
    --stochastic_eval $stochastic_eval \
    --alpha $alpha \
    --find_collision_free $find_collision_free \
    --max_reuse_data $max_reuse_data \
    --min_reuse_span $min_reuse_span \
    --is_simplified $is_simplified \
    --simplified_duration $simplified_duration \
    --simple_planner_type $simple_planner_type \
    --use_interpolation $use_interpolation \
    --interpolate_type $interpolate_type
