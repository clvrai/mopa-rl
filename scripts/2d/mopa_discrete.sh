#/!/bin/bash -x
gpu=$1
seed=$2
algo='sac'
prefix="MoPA-SAC.discrete"
env="PusherObstacle-v0"
max_episode_step="400"
debug="False"
reward_type='sparse'
log_root_dir="./logs"
mopa="True"
reward_scale="0.2"
reuse_data="True"
action_range="1.0"
stochastic_eval="True"
invalid_target_handling="True"
max_reuse_data='30'
use_smdp_update="True"
ac_space_type="normal"
success_reward="150.0"
discrete_action="True"


#mpiexec -n $workers
python -m rl.main \
    --log_root_dir $log_root_dir \
    --prefix $prefix \
    --env $env \
    --gpu $gpu \
    --max_episode_step $max_episode_step \
    --debug $debug \
    --algo $algo \
    --seed $seed \
    --reward_type $reward_type \
    --comment $comment \
    --mopa $mopa \
    --reward_scale $reward_scale \
    --reuse_data $reuse_data \
    --action_range $action_range \
    --discrete_action $discrete_action \
    --success_reward $success_reward \
    --stochastic_eval $stochastic_eval \
    --invalid_target_handling $invalid_target_handling \
    --max_reuse_data $max_reuse_data \
    --use_smdp_update $use_smdp_update \
    --ac_space_type $ac_space_type \
