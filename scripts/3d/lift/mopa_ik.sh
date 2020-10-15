#<!/bin/bash -x

prefix="MoPA-SAC.IK"
gpu=$1
seed=$2
algo='sac'
env="SawyerLiftObstacle-v0"
max_episode_step="250"
debug="False"
reward_type='sparse'
log_root_dir="./logs"
mopa="True"
reuse_data="True"
action_range="0.2"
stochastic_eval="True"
invalid_target_handling="True"
vis_replay="True"
plot_type='3d'
use_smdp_update="True"
use_discount_meta="True"
step_size="0.02"
success_reward="150.0"
max_reuse_data='15'
reward_scale="0.2"
use_ik_target="True"
ik_target="grip_site"
omega='0.05'

python -m rl.main \
    --log_root_dir $log_root_dir \
    --prefix $prefix \
    --env $env \
    --gpu $gpu \
    --max_episode_step $max_episode_step \
    --omega $omega \
    --debug $debug \
    --algo $algo \
    --seed $seed \
    --reward_type $reward_type \
    --mopa $mopa \
    --reuse_data $reuse_data \
    --action_range $action_range \
    --stochastic_eval $stochastic_eval \
    --invalid_target_handling $invalid_target_handling \
    --vis_replay $vis_replay \
    --plot_type $plot_type \
    --use_smdp_update $use_smdp_update \
    --step_size $step_size \
    --success_reward $success_reward \
    --max_reuse_data $max_reuse_data \
    --reward_scale $reward_scale \
    --use_ik_target $use_ik_target \
    --ik_target $ik_target \
    --use_discount_meta $use_discount_meta \
