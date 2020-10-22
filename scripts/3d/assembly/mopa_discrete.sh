#<!/bin/bash -x

prefix="SAC.PLANNER.AUGMENTED.discrete"
gpu=$1
seed=$2
algo='sac'
env="SawyerAssemblyObstacle-v0"
max_episode_step="250"
debug="False"
reward_type='sparse'
log_root_dir="./logs"
mopa="True"
reuse_data="True"
action_range="0.5"
stochastic_eval="True"
invalid_target_handling="True"
vis_replay="True"
plot_type='3d'
ac_space_type="normal"
use_smdp_update="True"
step_size="0.02"
success_reward="150.0"
max_reuse_data='30'
reward_scale="1.0"
evaluate_interval="10000"
discrete_action="True"

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
    --mopa $mopa \
    --reuse_data $reuse_data \
    --action_range $action_range \
    --discrete_action $discrete_action \
    --stochastic_eval $stochastic_eval \
    --invalid_target_handling $invalid_target_handling \
    --vis_replay $vis_replay \
    --plot_type $plot_type \
    --use_smdp_update $use_smdp_update \
    --ac_space_type $ac_space_type \
    --step_size $step_size \
    --success_reward $success_reward \
    --max_reuse_data $max_reuse_data \
    --reward_scale $reward_scale \
    --evaluate_interval $evaluate_interval \
