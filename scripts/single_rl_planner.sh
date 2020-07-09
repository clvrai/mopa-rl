#/!/bin/bash -x
gpu=$1
seed=$2

algo='sac'
evaluate_interval="1000"
ckpt_interval='200000'
rl_activation="relu"
num_batches="4"
log_interval="150"

workers="1"
tanh="True"
prefix="SAC.PUSHER.4OBS.reuse.random.smdp"
env="simple-pusher-obstacle-hard-v0"
rl_hid_size="256"
max_episode_step="200"
debug="False"
batch_size="256"
reward_type='dense'
comment='Sanity Check'
log_root_dir="./logs"
log_freq='1000'
planner_integration="True"
allow_manipulation_collision="True"
reward_scale="1.0"
reuse_data_type="random"
action_range="2.0"
ac_rl_minimum="-0.5"
ac_rl_maximum="0.5"
invalid_planner_rew="-0.5"
extended_action="False"
has_terminal='True'
stochastic_eval="True"
alpha='0.05'
find_collision_free="True"
max_reuse_data='30'
min_reuse_span='20'
use_interpolation="True"
interpolate_type="simple"
use_smdp_update="True"
# max_grad_norm='0.5'

#mpiexec -n $workers
python -m rl.main \
    --log_root_dir $log_root_dir \
    --wandb True \
    --prefix $prefix \
    --env $env \
    --gpu $gpu \
    --rl_hid_size $rl_hid_size \
    --max_episode_step $max_episode_step \
    --evaluate_interval $evaluate_interval \
    --buffer_size $buffer_size \
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
    --tanh $tanh \
    --planner_integration $planner_integration \
    --allow_manipulation_collision $allow_manipulation_collision \
    --reward_scale $reward_scale \
    --reuse_data_type $reuse_data_type \
    --action_range $action_range \
    --ac_rl_maximum $ac_rl_maximum \
    --ac_rl_minimum $ac_rl_minimum \
    --invalid_planner_rew $invalid_planner_rew \
    --extended_action $extended_action \
    --success_reward $success_reward \
    --has_terminal $has_terminal \
    --stochastic_eval $stochastic_eval \
    --alpha $alpha \
    --find_collision_free $find_collision_free \
    --max_reuse_data $max_reuse_data \
    --min_reuse_span $min_reuse_span \
    --use_interpolation $use_interpolation \
    --interpolate_type $interpolate_type \
    --use_smdp_update $use_smdp_update \
