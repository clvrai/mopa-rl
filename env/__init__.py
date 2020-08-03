""" Define all environments. """

from gym.envs.registration import register

# register all environments to use

register(
    id='pusher-v0',
    entry_point='env.pusher:PusherEnv',
    kwargs={}
)


register(
    id='pusher-obstacle-hard-v0',
    entry_point='env.pusher:PusherObstacleHardEnv',
    kwargs={}
)
register(
    id='pusher-obstacle-hard-v1',
    entry_point='env.pusher:PusherObstacleHardV1Env',
    kwargs={}
)
register(
    id='pusher-obstacle-hard-v2',
    entry_point='env.pusher:PusherObstacleHardV2Env',
    kwargs={}
)
register(
    id='pusher-obstacle-hard-v3',
    entry_point='env.pusher:PusherObstacleHardV3Env',
    kwargs={}
)

register(
    id='pusher-obstacle-v0',
    entry_point='env.pusher:PusherObstacleEnv',
    kwargs={}
)


register(
    id='sawyer-assembly-v0',
    entry_point='env.sawyer:SawyerAssemblyEnv',
    kwargs={}
)

register(
    id='sawyer-lift-v0',
    entry_point='env.sawyer:SawyerLiftEnv',
    kwargs={}
)
register(
    id='sawyer-lift-obstacle-v0',
    entry_point='env.sawyer:SawyerLiftObstacleEnv',
    kwargs={}
)

register(
    id="sawyer-assembly-easy-v0",
    entry_point="env.sawyer:SawyerAssemblyEasyEnv",
    kwargs={}
)
register(
    id="sawyer-push-v0",
    entry_point="env.sawyer:SawyerPushEnv",
    kwargs={}
)
register(
    id="sawyer-push-obstacle-v0",
    entry_point="env.sawyer:SawyerPushObstacleEnv",
    kwargs={}
)
register(
    id="sawyer-push-obstacle-easy-v0",
    entry_point="env.sawyer:SawyerPushObstacleEasyEnv",
    kwargs={}
)
register(
    id="sawyer-push-obstacle-v1",
    entry_point="env.sawyer:SawyerPushObstacleV1Env",
    kwargs={}
)
register(
    id="sawyer-push-obstacle-v2",
    entry_point="env.sawyer:SawyerPushObstacleV2Env",
    kwargs={}
)
