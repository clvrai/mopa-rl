import argparse

from util import str2bool, str2list


def argparser():
    parser = argparse.ArgumentParser(
        'Skill Coordination',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # environment
    parser.add_argument('--env', type=str, default='reacher-obstacle',
                        help='Environment name')
    parser.add_argument('--env_args', type=str, default=None)
    parser.add_argument('--init_qpos_dir', type=str, default=None)

    # training algorithm
    parser.add_argument('--algo', type=str, default='sac',
                        choices=['sac', 'ppo'])
    parser.add_argument('--policy', type=str, default='mlp',
                        choices=['mlp', 'manual'])
    parser.add_argument('--meta', type=str, default=None,
                        choices=[None, 'hard', 'soft'])
    parser.add_argument('--meta_update_target', type=str, default='HL',
                        choices=['HL', 'LL', 'both'])

    # vanilla rl
    parser.add_argument('--rl_hid_size', type=int, default=64)
    parser.add_argument('--rl_activation', type=str, default='relu',
                        choices=['relu', 'elu', 'tanh'])
    parser.add_argument('--tanh_policy', type=str2bool, default=True)

    # coordination
    parser.add_argument('--max_meta_len', type=int, default=25)
    parser.add_argument('--fix_embedding', type=str2bool, default=False, help='fix skill embedding if meta_ac does not change')

    # observation normalization
    parser.add_argument('--ob_norm', type=str2bool, default=True)
    parser.add_argument('--max_ob_norm_step', type=int, default=int(1e7))
    parser.add_argument('--clip_obs', type=float, default=200, help='the clip range of observation')
    parser.add_argument('--clip_range', type=float, default=5, help='the clip range after normalization of observation')

    # off-policy rl
    parser.add_argument('--buffer_size', type=int, default=int(1e3), help='the size of the buffer')
    parser.add_argument('--discount_factor', type=float, default=0.99, help='the discount factor')
    parser.add_argument('--lr_actor', type=float, default=3e-4, help='the learning rate of the actor')
    parser.add_argument('--lr_critic', type=float, default=3e-4, help='the learning rate of the critic')
    parser.add_argument('--polyak', type=float, default=0.995, help='the average coefficient')

    # subdiv
    parser.add_argument('--subdiv', type=str, default=None)
    parser.add_argument('--subdiv_skills', type=str, default=None)
    parser.add_argument('--subdiv_skill_dir', type=str, default=None)
    parser.add_argument('--hard_ob_mask', type=str2bool, default=True)
    parser.add_argument('--ob_mask_size', type=int, default=10)
    parser.add_argument('--custom_loss_weight', type=float, default=1,
                        help='the weight of custom policy loss')
    parser.add_argument('--custom_loss_type', type=str, default='ent',
                        choices=['ent', 'mean'],
                        help='the type of custom policy loss to enforce')

    # training
    parser.add_argument('--is_train', type=str2bool, default=True)
    parser.add_argument('--num_batches', type=int, default=50,
                        help='the times to update the network per epoch')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='the sample batch size')
    parser.add_argument('--max_grad_norm', type=float, default=100)
    parser.add_argument('--max_global_step', type=int, default=int(2e6))
    parser.add_argument('--gpu', type=int, default=None)

    # sac
    parser.add_argument('--reward_scale', type=float, default=1.0, help='reward scale')

    # ppo
    parser.add_argument('--clip_param', type=float, default=0.2)
    parser.add_argument('--value_loss_coeff', type=float, default=0.5)
    parser.add_argument('--action_loss_coeff', type=float, default=1.0)
    parser.add_argument('--entropy_loss_coeff', type=float, default=1e-4)
    parser.add_argument('--rollout_length', type=int, default=1000)
    parser.add_argument('--gae_lambda', type=float, default=0.95)

    # diayn
    parser.add_argument('--diayn', type=str2bool, default=True)
    parser.add_argument('--z_dim', type=int, default=5)
    parser.add_argument('--z_dist', type=str, default='normal',
                        choices=['normal', 'categorical'])

    # log
    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--evaluate_interval', type=int, default=100)
    parser.add_argument('--ckpt_interval', type=int, default=200)
    # parser.add_argument('--log_root_dir', type=str, default='log')
    parser.add_argument('--log_root_dir', type=str, default='/data/jingyuny/projects/subdiv/log')

    # evaluation
    parser.add_argument('--ckpt_num', type=int, default=None)
    parser.add_argument('--num_eval', type=int, default=10)
    parser.add_argument('--save_rollout', type=str2bool, default=False,
                        help='save rollout information during evaluation')
    parser.add_argument('--record', type=str2bool, default=True)
    parser.add_argument('--record_caption', type=str2bool, default=True)
    parser.add_argument('--num_record_samples', type=int, default=1,
                        help='number of trajectories to collect during eval')
    parser.add_argument('--save_qpos', type=str2bool, default=False,
                        help='save entire qpos history of success rollouts to file (for idle primitive training)')
    parser.add_argument('--save_success_qpos', type=str2bool, default=True,
                        help='save later segment of success rollouts to file (for moving and placing primitie trainings)')

    # misc
    parser.add_argument('--prefix', type=str, default='test')
    parser.add_argument('--notes', type=str, default='')
    parser.add_argument('--seed', type=int, default=123, help='Random seed')
    parser.add_argument('--debug', type=str2bool, default=False)

    args, unparsed = parser.parse_known_args()
    args.env_args_str = args.env_args

    return args, unparsed
