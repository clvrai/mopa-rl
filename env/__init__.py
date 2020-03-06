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
    id='sawyer-move-v0',
    entry_point='env.robosuite:SawyerMoveEnv',
    kwargs={}
)
register(
    id='sawyer-move-single-v0',
    entry_point='env.robosuite:SawyerMoveSingleEnv',
    kwargs={}
)
register(
    id='sawyer-move-milk-v0',
    entry_point='env.robosuite:SawyerMoveMilkEnv',
    kwargs={}
)
register(
    id='sawyer-move-bread-v0',
    entry_point='env.robosuite:SawyerMoveBreadEnv',
    kwargs={}
)
register(
    id='sawyer-move-cereal-v0',
    entry_point='env.robosuite:SawyerMoveCerealEnv',
    kwargs={}
)

register(
    id='sawyer-move-can-v0',
    entry_point='env.robosuite:SawyerMoveCanEnv',
    kwargs={}
)

register(
    id='sawyer-pick-place-single-v0',
    entry_point='env.robosuite:SawyerPickPlaceSingleEnv',
    kwargs={}
)
register(
    id='sawyer-pick-place-milk-v0',
    entry_point='env.robosuite:SawyerPickPlaceMilkEnv',
    kwargs={}
)
register(
    id='sawyer-pick-place-bread-v0',
    entry_point='env.robosuite:SawyerPickPlaceBreadEnv',
    kwargs={}
)
register(
    id='sawyer-pick-place-cereal-v0',
    entry_point='env.robosuite:SawyerPickPlaceCerealEnv',
    kwargs={}
)

register(
    id='sawyer-pick-place-can-v0',
    entry_point='env.robosuite:SawyerPickPlaceCanEnv',
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

register(
    id='simple-pusher-v0',
    entry_point='env.pusher:SimplePusherEnv',
    kwargs={}
)

register(
    id='simple-mover-v0',
    entry_point='env.mover:SimpleMoverEnv',
    kwargs={}
)

register(
    id='pusher-push-v0',
    entry_point='env.pusher.primitives.pusher_push:PusherPushEnv',
    kwargs={}
)

register(
    id='pusher-push-obstacle-v0',
    entry_point="env.pusher.primitives.pusher_push_obstacle:PusherPushObstacleEnv",
    kwargs={}
)

register(
    id='simple-pusher-obstacle-v0',
    entry_point='env.pusher:SimplePusherObstacleEnv',
    kwargs={}
)
