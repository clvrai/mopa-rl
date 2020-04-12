#!/bin/bash -x
v=$1
gpu=$2

if [ $v = 1 ]
then
    env="simple-pusher-v0"
    primitive_skills="reach_mp push"
elif [ $v = 2 ]
then
    env="simple-mover-v0"
    primitive_skills="reach_mp grasp manipulation_mp"
elif [ $v = 3 ]
then
    env='simple-mover-obstacle-v0'
    primitive_skills="reach_mp grasp manipulation_mp"
fi

workers="8"
prefix="4.11.hrl.ppo.ll.debug"
hrl="True"
ll_type="mix"
planner_type="sst"
planner_objective="state_const_integral"
range="1.0"
threshold="0.5"
timelimit="0.01"
gpu=$gpu
rl_hid_size="256"
meta_update_target="LL"
meta_oracle="True"
meta_subgoal_rew="0."
max_meta_len="15"
buffer_size="51200"
num_batches="16"
debug="False"
rollout_length="12800"
batch_size="2048"
reward_type="dense"
reward_scale="10."
comment="init buffer size is 10 times batch size"
ctrl_reward_coef="1e-2"
actor_num_hid_layers="1"
subgoal_type="joint"
subgoal_reward="True"
meta_algo='ppo'
success_reward='100.'
subgoal_predictor="True"
seed="1234"
has_terminal='True'
ignored_contact_geoms=' None,None box,l_finger_g0/box,r_finger_g0'
log_root_dir='./logs'
algo='ppo'

mpiexec -n $workers python -m rl.main \
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
    --meta_subgoal_rew $meta_subgoal_rew \
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
    --success_reward $success_reward \
    --primitive_skills $primitive_skills \
    --subgoal_predictor $subgoal_predictor \
    --has_terminal $has_terminal \
    --meta_oracle $meta_oracle \
    --ignored_contact_geoms $ignored_contact_geoms \
    --algo $algo
