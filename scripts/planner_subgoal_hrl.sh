#!/bin/bash -x
v=$1
gpu=$2

if [ $v = 1 ]
then
    env="simple-pusher-v0"
    primitive_skills="reach_mp push"
    ignored_contact_geoms='None,None'
elif [ $v = 2 ]
then
    env="simple-pusher-obstacle-v0"
    primitive_skills="reach_mp push"
    ignored_contact_geoms='None,None'
elif [ $v = 3 ]
then
    env="simple-pusher-obstacle-hard-v0"
    primitive_skills="reach_mp push"
    ignored_contact_geoms='None,None'
elif [ $v = 4 ]
then
    env="simple-mover-v0"
    primitive_skills="reach_mp grasp manipulation_mp"
    ignored_contact_geoms='None,None None,None box,l_finger_g0/box,r_finger_g0/box,gripper_base_geom'
    # primitive_skills="reach grasp manipulation"
    # ignored_contact_geoms='None,None'
elif [ $v = 5 ]
then
    env='simple-mover-obstacle-v0'
    primitive_skills="reach_mp grasp manipulation_mp"
    ignored_contact_geoms='None,None None,None box,l_finger_g0/box,r_finger_g0/box,gripper_base_geom'
elif [ $v = 6 ]
then
    env='simple-reacher-v0'
    primitive_skills="reach_mp"
    ignored_contact_geoms='None,None'
elif [ $v = 7 ]
then
    env='reacher-obstacle-v0'
    primitive_skills="reach_mp reach"
    ignored_contact_geoms="None,None"
fi

workers="1"
prefix="5.07.SAC.HRL.REUSE"
hrl="True"
ll_type="mix"
planner_type="sst"
planner_objective="state_const_integral"
primitive_skills="mp rl"
range="0.5"
threshold="0.1"
timelimit="0.01"
gpu=$gpu
rl_hid_size="256"
meta_update_target="both"
meta_oracle="False"
invalid_planner_rew="-0.3"
max_meta_len="1"
buffer_size="1000000"
num_batches="1"
debug="False"
rollout_length="15000"
batch_size="256"
reward_type="dense"
reward_scale="10."
comment="init buffer size is 10 times batch size"
ctrl_reward_coef="1e-2"
actor_num_hid_layers="2"
subgoal_type="joint"
subgoal_reward="True"
meta_algo='sac'
start_steps='10000'
success_reward='150.'
subgoal_predictor="True"
subgoal_hindsight="True"
seed="1234"
has_terminal='True'
log_root_dir='./logs'
use_automatic_entropy_tuning="True"
group='05.07.SAC.HRL.REUSE.PUSH-OBSTSTACLE-HARD'
log_freq='1000'
allow_self_collision="True"
allow_manipulation_collision="True"
rl_activation="relu"
relative_goal="True"
log_interval="100"
reuse_data="True"

#mpiexec -n $workers
python -m rl.main \
    --log_root_dir $log_root_dir \
    --wandb True \
    --prefix $prefix \
    --hrl $hrl \
    --ll_type $ll_type \
    --planner_type $planner_type \
    --planner_objective $planner_objective \
    --range $range \
    --threshold $threshold \
    --timelimit $timelimit \
    --env $env \
    --gpu $gpu \
    --rl_hid_size $rl_hid_size \
    --meta_update_target $meta_update_target \
    --max_meta_len $max_meta_len \
    --buffer_size $buffer_size \
    --num_batches $num_batches \
    --debug $debug \
    --rollout_length $rollout_length \
    --batch_size $batch_size \
    --reward_type $reward_type \
    --reward_scale $reward_scale \
    --comment $comment \
    --seed $seed \
    --ctrl_reward_coef $ctrl_reward_coef \
    --actor_num_hid_layers $actor_num_hid_layers \
    --subgoal_type $subgoal_type \
    --subgoal_reward $subgoal_reward \
    --meta_algo $meta_algo \
    --start_steps $start_steps \
    --success_reward $success_reward \
    --primitive_skills $primitive_skills \
    --subgoal_predictor $subgoal_predictor \
    --has_terminal $has_terminal \
    --meta_oracle $meta_oracle \
    --ignored_contact_geoms $ignored_contact_geoms \
    --use_automatic_entropy_tuning $use_automatic_entropy_tuning \
    --group $group \
    --subgoal_hindsight $subgoal_hindsight \
    --invalid_planner_rew $invalid_planner_rew \
    --log_freq $log_freq \
    --allow_manipulation_collision $allow_manipulation_collision \
    --allow_self_collision $allow_self_collision \
    --rl_activation $rl_activation \
    --relative_goal $relative_goal \
    --log_interval $log_interval \
    --reuse_data $reuse_data
