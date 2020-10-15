""" Define all environments. """

from gym.envs.registration import register


# register all environments to use
register(id="pusher-v0", entry_point="env.pusher:PusherEnv", kwargs={})
register(
    id="pusher-obstacle-v0",
    entry_point="env.pusher:PusherObstacleEnv",
    kwargs={},
)

register(
    id="sawyer-assembly-v0", entry_point="env.sawyer:SawyerAssemblyEnv", kwargs={}
)
register(
    id="sawyer-assembly-obstacle-v0",
    entry_point="env.sawyer:SawyerAssemblyObstacleEnv",
    kwargs={},
)

register(id="sawyer-lift-v0", entry_point="env.sawyer:SawyerLiftEnv", kwargs={})
register(
    id="sawyer-lift-obstacle-v0",
    entry_point="env.sawyer:SawyerLiftObstacleEnv",
    kwargs={},
)

register(id="sawyer-push-v0", entry_point="env.sawyer:SawyerPushEnv", kwargs={})
register(
    id="sawyer-push-obstacle-v0",
    entry_point="env.sawyer:SawyerPushObstacleEnv",
    kwargs={},
)
