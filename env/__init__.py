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
    id='simple-reacher-obstacle-hard-v0',
    entry_point='env.reacher:SimpleReacherObstacleHardEnv',
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
    id='simple-peg-insertion-v0',
    entry_point='env.peg_insertion:SimplePegInsertionEnv',
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
    id='simple-pusher-obstacle-hard-v0',
    entry_point='env.pusher:SimplePusherObstacleHardEnv',
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
    id='simple-pusher-obstacle-hard-v1',
    entry_point='env.pusher:SimplePusherObstacleHardV1Env',
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
    id='sawyer-pick-place-robosuite-v0',
    entry_point='env.robosuite:SawyerPickPlaceEnv',
    kwargs={}
)
register(
    id='sawyer-reach-robosuite-v0',
    entry_point='env.robosuite:SawyerReachEnv',
    kwargs={}
)
register(
    id='sawyer-reach-obstacle-robosuite-v0',
    entry_point='env.robosuite:SawyerReachObstacleEnv',
    kwargs={}
)
register(
    id='sawyer-pick-place-can-robosuite-v0',
    entry_point='env.robosuite:SawyerPickPlaceCanEnv',
    kwargs={}
)
register(
    id='sawyer-nut-assembly-single-robosuite-v0',
    entry_point='env.robosuite:SawyerNutAssemblySingleEnv',
    kwargs={}
)
register(
    id='sawyer-nut-assembly-square-robosuite-v0',
    entry_point='env.robosuite:SawyerNutAssemblySquareEnv',
    kwargs={}
)
register(
    id='sawyer-nut-assembly-round-robosuite-v0',
    entry_point='env.robosuite:SawyerNutAssemblyRoundEnv',
    kwargs={}
)
register(
    id='sawyer-nut-assembly-robosuite-v0',
    entry_point='env.robosuite:SawyerNutAssemblyEnv',
    kwargs={}
)
register(
    id='sawyer-pick-move-robosuite-v0',
    entry_point='env.robosuite:SawyerPickMoveEnv',
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
register(
    id='sawyer-lift-robosuite-v0',
    entry_point='env.robosuite:SawyerLiftEnv',
    kwargs={}
)
register(
    id="sawyer-reach-v0",
    entry_point="env.sawyer:SawyerReachEnv",
    kwargs={}
)

register(
    id="sawyer-reach-obstacle-v0",
    entry_point="env.sawyer:SawyerReachObstacleEnv",
    kwargs={}
)

register(
    id="sawyer-reach-float-obstacle-v0",
    entry_point="env.sawyer:SawyerReachFloatObstacleEnv",
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
    id="sawyer-peg-insertion-v0",
    entry_point="env.sawyer:SawyerPegInsertionEnv",
    kwargs={}
)
register(
    id="sawyer-peg-insertion-obstacle-v0",
    entry_point="env.sawyer:SawyerPegInsertionObstacleEnv",
    kwargs={}
)
register(
    id="sawyer-peg-insertion-obstacle-v1",
    entry_point="env.sawyer:SawyerPegInsertionObstacleV1Env",
    kwargs={}
)
register(
    id="sawyer-peg-insertion-obstacle-v2",
    entry_point="env.sawyer:SawyerPegInsertionObstacleV2Env",
    kwargs={}
)
register(
    id="sawyer-pick-place-v0",
    entry_point="env.sawyer:SawyerPickPlaceEnv",
    kwargs={}
)
register(
    id="sawyer-push-obstacle-v1",
    entry_point="env.sawyer:SawyerPushObstacleV1Env",
    kwargs={}
)
register(
    id="sawyer-pick-place-obstacle-v0",
    entry_point="env.sawyer:SawyerPickPlaceObstacleEnv",
    kwargs={}
)
