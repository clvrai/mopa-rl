""" Define all environments. """

from gym.envs.registration import register


# register all environments to use
register(
    id='reacher-obstacle-v0',
    entry_point='env.reacher:ReacherObstacleEnv',
    kwargs={},
)

register(
    id='simple-reacher-obstacle-v0',
    entry_point='env.reacher:SimpleReacherObstacleEnv',
    kwargs={},
)

register(
    id='simple-reacher-obstacle-toy-v0',
    entry_point='env.reacher:SimpleReacherObstacleToyEnv',
    kwargs={},
)

register(
    id='simple-reacher-obstacle-pixel-v0',
    entry_point='env.reacher:SimpleReacherObstaclePixelEnv',
    kwargs={},
)

register(
    id='reacher-v0',
    entry_point='env.reacher:ReacherEnv',
    kwargs={},
)

register(
    id='simple-reacher-v0',
    entry_point='env.reacher:SimpleReacherEnv',
    kwargs={},
)

register(
    id='reacher-obstacle-test-v0',
    entry_point='env.reacher:ReacherObstacleTestEnv',
    kwargs={},
)

register(
    id='reacher-obstacle-pixel-v0',
    entry_point='env.reacher:ReacherObstaclePixelEnv',
    kwargs={},
)

register(
    id='reacher-pixel-v0',
    entry_point='env.reacher:ReacherPixelEnv',
    kwargs={},
)

register(
    id='sawyer-pick-place-v0',
    entry_point='env.robosuite:SawyerPickPlaceEnv',
    kwargs={}
)

register(
    id='sawyer-push-v0',
    entry_point='env.robosuite:SawyerPushEnv',
    kwargs={}
)

register(
    id='sawyer-stack-v0',
    entry_point='env.robosuite:SawyerStackEnv',
    kwargs={}
)
