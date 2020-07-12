#/!/bin/bash -x
gpu=$1
seed=$2

algo='sac'
evaluate_interval="1000"
ckpt_interval='200000'
rl_activation="relu"
num_batches="1"
log_interval="1000"

workers="1"
tanh="True"
prefix="SAC.PUSHER.SPARSE.reuse.scale.0.2.ac_range0.5.v3.debug"
env="simple-pusher-obstacle-hard-v0"
rl_hid_size="256"
max_episode_step="200"
debug="debug"
batch_size="256"
reward_type='sparse'
comment='Sanity Check'
log_root_dir="./logs"
log_freq='1000'
planner_integration="True"
allow_manipulation_collision="True"
reward_scale="0.2"
reuse_data_type="random"
action_range="0.3"
ac_rl_minimum="-0.7"
ac_rl_maximum="0.7"
invalid_planner_rew="-0.0"
extended_action="False"
has_terminal='True'
stochastic_eval="True"
alpha='1.0'
find_collision_free="True"
use_double_planner="False"
max_reuse_data='15'
min_reuse_span='20'
use_smdp_update="True"
ac_space_type="piecewise"
discount_factor='0.99'
success_reward="150.0"
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
    --use_smdp_update $use_smdp_update \
    --ac_space_type $ac_space_type \
    --discount_factor $discount_factor \
    --use_double_planner $use_double_planner
