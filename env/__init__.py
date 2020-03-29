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
    id='simple-mover-obstacle-v0',
    entry_point='env.mover:SimpleMoverObstacleEnv',
    kwargs={}
)

register(
    id='pusher-v0',
    entry_point='env.pusher:PusherEnv',
    kwargs={}
)

register(
    id='simple-pusher-pixel-v0',
    entry_point='env.pusher:SimplePusherPixelEnv',
    kwargs={}
)

register(
    id='pusher-push-v0',
    entry_point='env.pusher.primitives.pusher_push:PusherPushEnv',
    kwargs={}
)
register(
    id='simple-pusher-push-v0',
    entry_point='env.pusher.primitives.simple_pusher_push:SimplePusherPushEnv',
    kwargs={}
)

register(
    id='pusher-push-pixel-v0',
    entry_point='env.pusher.primitives.pusher_push_pixel:PusherPushPixelEnv',
    kwargs={}
)

register(
    id='pusher-push-obstacle-v0',
    entry_point="env.pusher.primitives.pusher_push_obstacle:PusherPushObstacleEnv",
    kwargs={}
)

register(
    id='simple-pusher-push-obstacle-v0',
    entry_point="env.pusher.primitives.simple_pusher_push_obstacle:SimplePusherPushObstacleEnv",
    kwargs={}
)

register(
    id='simple-pusher-obstacle-v0',
    entry_point='env.pusher:SimplePusherObstacleEnv',
    kwargs={}
)

register(
    id='pusher-obstacle-v0',
    entry_point='env.pusher:PusherObstacleEnv',
    kwargs={}
)

register(
    id='pusher-push-obstacle-pixel-v0',
    entry_point="env.pusher.primitives.pusher_push_obstacle_pixel:PusherPushObstaclePixelEnv",
    kwargs={}
)

register(
    id='simple-pusher-obstacle-pixel-v0',
    entry_point='env.pusher:SimplePusherObstaclePixelEnv',
    kwargs={}
)

register(
    id='sawyer-pick-place-v0',
    entry_point='env.sawyer:SawyerPickAndPlaceEnv',
    kwargs={}
)

register(
    id='sawyer-reach-push-pick-place-v0',
    entry_point='env.sawyer:SawyerReachPushPickPlaceEnv',
    kwargs={}
)

register(
    id='sawyer-reach-push-pick-place-wall-v0',
    entry_point='env.sawyer:SawyerReachPushPickPlaceWallEnv',
    kwargs={}
)

register(
    id='sawyer-reach-push-pick-place-obstacle-v0',
    entry_point='env.sawyer:SawyerReachPushPickPlaceObstacleEnv',
    kwargs={}
)

register(
    id='sawyer-assembly-peg-v0',
    entry_point='env.sawyer:SawyerNutAssemblyEnv',
    kwargs={}
)

register(
    id='sawyer-pick-place-robosuite-v0',
    entry_point='env.robosuite:SawyerPickPlaceEnv',
    kwargs={}
)
register(
    id='sawyer-move-robosuite-v0',
    entry_point='env.robosuite:SawyerMoveEnv',
    kwargs={}
)
register(
    id='sawyer-move-single-robosuite-v0',
    entry_point='env.robosuite:SawyerMoveSingleEnv',
    kwargs={}
)
register(
    id='sawyer-move-milk-robosuite-v0',
    entry_point='env.robosuite:SawyerMoveMilkEnv',
    kwargs={}
)
register(
    id='sawyer-move-bread-robosuite-v0',
    entry_point='env.robosuite:SawyerMoveBreadEnv',
    kwargs={}
)
register(
    id='sawyer-move-cereal-robosuite-v0',
    entry_point='env.robosuite:SawyerMoveCerealEnv',
    kwargs={}
)

register(
    id='sawyer-move-can-robosuite-v0',
    entry_point='env.robosuite:SawyerMoveCanEnv',
    kwargs={}
)

register(
    id='sawyer-pick-place-single-robosuite-v0',
    entry_point='env.robosuite:SawyerPickPlaceSingleEnv',
    kwargs={}
)
register(
    id='sawyer-pick-place-milk-robosuite-v0',
    entry_point='env.robosuite:SawyerPickPlaceMilkEnv',
    kwargs={}
)
register(
    id='sawyer-pick-place-bread-robosuite-v0',
    entry_point='env.robosuite:SawyerPickPlaceBreadEnv',
    kwargs={}
)
register(
    id='sawyer-pick-place-cereal-robosuite-v0',
    entry_point='env.robosuite:SawyerPickPlaceCerealEnv',
    kwargs={}
)

register(
    id='sawyer-pick-place-can-robosuite-v0',
    entry_point='env.robosuite:SawyerPickPlaceCanEnv',
    kwargs={}
)

register(
    id='sawyer-push-robosuite-v0',
    entry_point='env.robosuite:SawyerPushEnv',
    kwargs={}
)

register(
    id='sawyer-stack-robosuite-v0',
    entry_point='env.robosuite:SawyerStackEnv',
    kwargs={}
)

register(
    id='sawyer-test-robosuite-v0',
    entry_point='env.robosuite:SawyerTestEnv',
    kwargs={}
)
