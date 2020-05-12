#!/bin/bash -x
algo=$1
gpu=$2

if [ $algo = 1 ]
then
    algo='ppo'
    rollout_length='1024'
    evaluate_interval="10"
    ckpt_interval='50'
    rl_activation="tanh"
    num_batches="100"
    log_interval="1"
elif [ $algo = 2 ]
then
    algo='sac'
    rollout_length="1000"
    evaluate_interval="1000"
    ckpt_interval='100000'
    rl_activation="relu"
    num_batches="1"
    log_interval="100"
fi

workers="1"
tanh="True"
prefix="05.10.SAC.REUSE.debug"
max_global_step="60000000"
env="simple-pusher-obstacle-v0"
gpu=$gpu
rl_hid_size="256"
max_episode_step="150"
entropy_loss_coef="1e-3"
buffer_size="1000000"
lr_actor="3e-4"
lr_critic="3e-4"
debug="True"
batch_size="256"
clip_param="0.2"
seed='1242'
ctrl_reward='1e-2'
reward_type='dense'
comment='Fix motion planner'
start_steps='10000'
actor_num_hid_layers='2'
success_reward='150.'
has_terminal='True'
log_root_dir="./logs"
group='05.10.SAC.SINGLE.REUSE.PUSH-OBSTACLE'
env_debug='False'
log_freq='1000'
planner_integration="True"
ignored_contact_geoms='None,None'
planner_type="sst"
planner_objective="state_const_integral"
range="0.3"
threshold="0.0"
timelimit="1.0"
allow_self_collision="False"
allow_manipulation_collision="True"
reward_scale="10."
subgoal_hindsight="True"
reuse_data="True"
relative_goal="False"
simple_planner_timelimit="0.01"


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
    --success_reward $success_reward \
    --has_terminal $has_terminal \
    --group $group \
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
    --allow_self_collision $allow_self_collision \
    --reward_scale $reward_scale \
    --subgoal_hindsight $subgoal_hindsight \
    --reuse_data $reuse_data \
    --relative_goal $relative_goal \
    --simple_planner_timelimit $simple_planner_timelimit
