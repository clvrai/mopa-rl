#/!/bin/bash -x
gpu=$1
seed=$2
algo='sac'
prefix="SAC.PUSHER.discrete.action_range.1.0.data30.scale.0.2"
env="pusher-obstacle-hard-v3"
max_episode_step="400"
debug="False"
reward_type='sparse'
# log_root_dir="./logs"
log_root_dir="/data/jun/projects/hrl-planner/logs"
planner_integration="True"
reward_scale="0.2"
reuse_data_type="random"
action_range="1.0"
ac_rl_minimum="-0.5"
ac_rl_maximum="0.5"
stochastic_eval="True"
find_collision_free="True"
max_reuse_data='30'
min_reuse_span='1'
use_smdp_update="True"
ac_space_type="normal"
success_reward="150.0"
log_indiv_entropy="True"

# variants
extended_action="True"


#mpiexec -n $workers
python -m rl.main \
    --log_root_dir $log_root_dir \
    --wandb True \
    --prefix $prefix \
    --env $env \
    --gpu $gpu \
    --max_episode_step $max_episode_step \
    --debug $debug \
    --algo $algo \
    --seed $seed \
    --reward_type $reward_type \
    --comment $comment \
    --planner_integration $planner_integration \
    --reward_scale $reward_scale \
    --reuse_data_type $reuse_data_type \
    --action_range $action_range \
    --ac_rl_maximum $ac_rl_maximum \
    --ac_rl_minimum $ac_rl_minimum \
    --extended_action $extended_action \
    --success_reward $success_reward \
    --stochastic_eval $stochastic_eval \
    --find_collision_free $find_collision_free \
    --max_reuse_data $max_reuse_data \
    --min_reuse_span $min_reuse_span \
    --use_smdp_update $use_smdp_update \
    --ac_space_type $ac_space_type \
    --use_double_planner $use_double_planner \
    --log_indiv_entropy $log_indiv_entropy \
