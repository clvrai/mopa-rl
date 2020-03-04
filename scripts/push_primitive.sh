#!/bin/bash
workers="3"
prefix="ll.push.sac.reward_scale.1.wo.norm.v1"
hrl="True"
max_global_step="60000000"
ll_type="rl"
env="pusher-push-v0"
gpu="1"
rl_hid_size="256"
meta_update_target="both"
hrl_network_to_update="both"
hl_type='subgoal'
max_episode_step="150"
max_meta_len="15"
evaluate_interval="1"
meta_tanh_policy="True"
max_grad_norm="0.5"
entropy_loss_coef="0.1"
buffer_size="10000"
num_batches="50"
lr_actor="3e-4"
lr_critic="3e-4"
debug="False"
rollout_length="1000"
batch_size="256"
clip_param="0.2"
rl_activation="relu"
reward_type='dense'
comment='Get rid of unnecessary obs'
seed='1234'
ctrl_reward_coef='1'
start_steps='10000'
reward_scale='1'
actor_num_hid_layers='1'


#mpiexec -n $workers
python -m rl.main --log_root_dir ./logs \
    --wandb True \
    --prefix $prefix \
    --max_global_step $max_global_step \
    --hrl $hrl \
    --ll_type $ll_type \
    --env $env \
    --gpu $gpu \
    --rl_hid_size $rl_hid_size \
    --meta_update_target $meta_update_target \
    --hrl_network_to_update $hrl_network_to_update \
    --max_episode_step $max_episode_step \
    --evaluate_interval $evaluate_interval \
    --meta_tanh_policy $meta_tanh_policy \
    --max_meta_len $max_meta_len \
    --entropy_loss_coef $entropy_loss_coef \
    --buffer_size $buffer_size \
    --num_batches $num_batches \
    --lr_actor $lr_actor \
    --lr_critic $lr_critic \
    --debug $debug \
    --rollout_length $rollout_length \
    --batch_size $batch_size \
    --clip_param $clip_param \
    --max_grad_norm $max_grad_norm \
    --rl_activation $rl_activation \
    --reward_type $reward_type \
    --comment $comment \
    --seed $seed \
    --ctrl_reward_coef $ctrl_reward_coef \
    --start_steps $start_steps \
    --reward_scale $reward_scale \
    --actor_num_hid_layers $actor_num_hid_layers \
    --hl_type $hl_type \
