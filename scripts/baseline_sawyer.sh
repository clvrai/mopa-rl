#!/bin/bash -x
gpu=$1
seed=$2

algo='sac'
rollout_length="10000"
evaluate_interval="1000"
ckpt_interval='100000'
rl_activation="relu"
num_batches="1"
log_interval="1000"

workers="1"
prefix="BASELINE.v1"
max_global_step="60000000"
env="sawyer-peg-insertion-v0"
gpu=$gpu
max_episode_step="200"
entropy_loss_coef="1e-3"
buffer_size="1000000"
debug="False"
batch_size="256"
clip_param="0.2"
ctrl_reward='1e-2'
reward_type='dense'
comment='Baseline'
start_steps='10000'
actor_num_hid_layers='2'
# log_root_dir="/data/jun/projects/hrl-planner/logs"
log_root_dir="./logs"
env_debug='False'
log_freq='1000'
alpha="0.2"
vis_replay="True"
plot_type='3d'
task_level='easy'
success_reward='100.'
# has_terminal='True'
# max_grad_norm='0.5'

#mpiexec -n $workers
python -m rl.main \
    --log_root_dir $log_root_dir \
    --wandb True \
    --prefix $prefix \
    --max_global_step $max_global_step \
    --env $env \
    --gpu $gpu \
    --max_episode_step $max_episode_step \
    --evaluate_interval $evaluate_interval \
    --entropy_loss_coef $entropy_loss_coef \
    --buffer_size $buffer_size \
    --num_batches $num_batches \
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
    --alpha $alpha \
    --vis_replay $vis_replay \
    --task_level $task_level \
    --plot_type $plot_type \
    --success_reward $success_reward
    # --has_terminal $has_terminal \ 
