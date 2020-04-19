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
    ignored_contact_geoms='None,None box,l_finger_g0/box,r_finger_g0'
elif [ $v = 3 ]
then
    env='simple-mover-obstacle-v0'
    primitive_skills="reach_mp grasp manipulation_mp"
    ignored_contact_geoms='None,None box,l_finger_g0/box,r_finger_g0/box,gripper_base_geom'
elif [ $v = 4 ]
then
    env='simple-reacher-v0'
    primitive_skills="reach_mp"
    ignored_contact_geoms='None,None'
elif [ $v = 5 ]
then
    env='simple-pusher-v0'
    primitive_skills="reach_mp push"
    ignored_contact_geoms='None,None'
fi

workers="1"
prefix="4.16.PPO-REWARD_COLLISION.HINDSIGHT.LONGER_RANGE-2"
hrl="True"
ll_type="mix"
planner_type="sst"
planner_objective="state_const_integral"
range="0.5"
threshold="0.5"
timelimit="0.01"
gpu=$gpu
rl_hid_size="256"
meta_update_target="LL"
meta_oracle="True"
meta_subgoal_rew="0."
max_meta_len="15"
buffer_size="12800"
num_batches="10"
debug="False"
rollout_length="9192"
batch_size="256"
evaluate_interval='10'
ckpt_interval='10'
reward_type="dense"
reward_scale="10."
entropy_loss_coeff='1e-2'
comment="Fixed ignored contacts"
ctrl_reward_coef="1e-2"
actor_num_hid_layers="2"
subgoal_type="joint"
meta_algo='ppo'
success_reward='10.'
subgoal_predictor="True"
seed="1234"
has_terminal='True'
log_root_dir='./logs'
algo='ppo'
group='4.16.PPO'
rl_activation='tanh'
subgoal_hindsight="True"
# max_grad_norm='0.5'

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
    --meta_algo $meta_algo \
    --success_reward $success_reward \
    --primitive_skills $primitive_skills \
    --subgoal_predictor $subgoal_predictor \
    --has_terminal $has_terminal \
    --meta_oracle $meta_oracle \
    --ignored_contact_geoms $ignored_contact_geoms \
    --algo $algo \
    --evaluate_interval $evaluate_interval \
    --ckpt_interval $ckpt_interval \
    --entropy_loss_coeff $entropy_loss_coeff \
    --group $group \
    --rl_activation $rl_activation \
    --subgoal_hindsight $subgoal_hindsight
    # --max_grad_norm $max_grad_norm \ 
