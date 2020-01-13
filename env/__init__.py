""" Define all environments. """

from gym.envs.registration import register


# register all environments to use
register(
    id='reacher-obstacle-v0',
    entry_point='env.reacher:ReacherObstacleEnv',
    kwargs={},
)
