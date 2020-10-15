""" Define all environments. """

from gym.envs.registration import register


# register all environments to use
register(
    id="PusherObstacle-v0",
    entry_point="env.pusher:PusherObstacleEnv",
    kwargs={},
)

register(
    id="SawyerAssembly-v0", entry_point="env.sawyer:SawyerAssemblyEnv", kwargs={}
)
register(
    id="SawyerAssemblyObstacle-v0",
    entry_point="env.sawyer:SawyerAssemblyObstacleEnv",
    kwargs={},
)

register(id="SawyerLift-v0", entry_point="env.sawyer:SawyerLiftEnv", kwargs={})
register(
    id="SawyerLiftObstacle-v0",
    entry_point="env.sawyer:SawyerLiftObstacleEnv",
    kwargs={},
)

register(id="SawyerPush-v0", entry_point="env.sawyer:SawyerPushEnv", kwargs={})
register(
    id="SawyerPushObstacle-v0",
    entry_point="env.sawyer:SawyerPushObstacleEnv",
    kwargs={},
)
