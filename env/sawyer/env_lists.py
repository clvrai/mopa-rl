'''
This file provide lists of environment for multitask learning.
'''

from metaworld.envs.mujoco.sawyer_xyz.sawyer_reach_push_pick_place import SawyerReachPushPickPlaceEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_door import SawyerDoorEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_hand_insert import SawyerHandInsertEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_assembly_peg import SawyerNutAssemblyEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_sweep import SawyerSweepEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_window_open import SawyerWindowOpenEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_hammer import SawyerHammerEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_window_close import SawyerWindowCloseEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_dial_turn import SawyerDialTurnEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_lever_pull import SawyerLeverPullEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_drawer_open import SawyerDrawerOpenEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_button_press_topdown import SawyerButtonPressTopdownEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_drawer_close import SawyerDrawerCloseEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_box_close import SawyerBoxCloseEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_peg_insertion_side import SawyerPegInsertionSideEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_bin_picking import SawyerBinPickingEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_drawer_close import SawyerDrawerCloseEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_box_close import SawyerBoxCloseEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_stick_push import SawyerStickPushEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_stick_pull import SawyerStickPullEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_button_press import SawyerButtonPressEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_shelf_place import SawyerShelfPlaceEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_door_close import SawyerDoorCloseEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_sweep_into_goal import SawyerSweepIntoGoalEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_coffee_button import SawyerCoffeeButtonEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_coffee_push import SawyerCoffeePushEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_coffee_pull import SawyerCoffeePullEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_faucet_open import SawyerFaucetOpenEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_faucet_close import SawyerFaucetCloseEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_peg_unplug_side import SawyerPegUnplugSideEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_soccer import SawyerSoccerEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_basketball import SawyerBasketballEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_reach_push_pick_place_wall import SawyerReachPushPickPlaceWallEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_push_back import SawyerPushBackEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_pick_out_of_hole import SawyerPickOutOfHoleEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_shelf_remove import SawyerShelfRemoveEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_disassemble_peg import SawyerNutDisassembleEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_door_lock import SawyerDoorLockEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_door_unlock import SawyerDoorUnlockEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_sweep_tool import SawyerSweepToolEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_button_press_wall import SawyerButtonPressWallEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_button_press_topdown_wall import SawyerButtonPressTopdownWallEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_handle_press import SawyerHandlePressEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_handle_pull import SawyerHandlePullEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_handle_press_side import SawyerHandlePressSideEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_handle_pull_side import SawyerHandlePullSideEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_plate_slide import SawyerPlateSlideEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_plate_slide_back import SawyerPlateSlideBackEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_plate_slide_side import SawyerPlateSlideSideEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_plate_slide_back_side import SawyerPlateSlideBackSideEnv


# easy mode for CoRL
EASY_MODE_LIST = [
    SawyerReachPushPickPlaceEnv,
    SawyerReachPushPickPlaceEnv,
    SawyerReachPushPickPlaceEnv,
    SawyerDoorEnv,
    SawyerDrawerOpenEnv,
    SawyerDrawerCloseEnv,
    SawyerButtonPressTopdownEnv,
    SawyerPegInsertionSideEnv,
    SawyerWindowOpenEnv,
    SawyerWindowCloseEnv,
]

MEDIUM_TRAIN_LIST = [
    SawyerReachPushPickPlaceEnv,
    SawyerReachPushPickPlaceEnv,
    SawyerReachPushPickPlaceEnv,
    SawyerDoorEnv,
    SawyerDrawerCloseEnv,
    SawyerButtonPressTopdownEnv,
    SawyerPegInsertionSideEnv,
    SawyerWindowOpenEnv,
    SawyerSweepEnv,
    SawyerBasketballEnv,
]


MEDIUM_TRAIN_AND_TEST_LIST = [
    SawyerReachPushPickPlaceEnv,
    SawyerReachPushPickPlaceEnv,
    SawyerReachPushPickPlaceEnv,
    SawyerDoorEnv,
    SawyerDrawerCloseEnv,
    SawyerButtonPressTopdownEnv,
    SawyerPegInsertionSideEnv,
    SawyerWindowOpenEnv,
    SawyerSweepEnv,
    SawyerBasketballEnv,
    #Test
    SawyerDrawerCloseEnv,
    SawyerDoorCloseEnv,
    SawyerShelfPlaceEnv,
    SawyerSweepEnv,
    SawyerLeverPullEnv
]

GRAD_MODE_LIST = [
    SawyerReachPushPickPlaceEnv,
    SawyerReachPushPickPlaceEnv,
    SawyerReachPushPickPlaceEnv,
]

# hard mode for CoRL
HARD_MODE_LIST = [
    SawyerReachPushPickPlaceEnv,
    SawyerReachPushPickPlaceEnv,
    SawyerReachPushPickPlaceEnv,
    SawyerReachPushPickPlaceWallEnv,
    SawyerReachPushPickPlaceWallEnv,
    SawyerReachPushPickPlaceWallEnv,
    SawyerDoorEnv,
    SawyerDoorCloseEnv,
    SawyerDrawerOpenEnv,
    SawyerDrawerCloseEnv,
    SawyerButtonPressTopdownEnv,
    SawyerButtonPressEnv,
    SawyerButtonPressTopdownWallEnv,
    SawyerButtonPressWallEnv,
    SawyerPegInsertionSideEnv,
    SawyerPegUnplugSideEnv,
    SawyerWindowOpenEnv,
    SawyerWindowCloseEnv,
    SawyerNutDisassembleEnv,
    SawyerHammerEnv,
    SawyerPlateSlideEnv,
    SawyerPlateSlideSideEnv,
    SawyerPlateSlideBackEnv, 
    SawyerPlateSlideBackSideEnv,
    SawyerHandlePressEnv,
    SawyerHandlePullEnv,
    SawyerHandlePressSideEnv,
    SawyerHandlePullSideEnv,
    SawyerStickPushEnv,
    SawyerStickPullEnv,
    SawyerBasketballEnv,
    SawyerSoccerEnv,
    SawyerFaucetOpenEnv,
    SawyerFaucetCloseEnv,
    SawyerCoffeePushEnv,
    SawyerCoffeePullEnv,
    SawyerCoffeeButtonEnv,
    SawyerSweepEnv,
    SawyerSweepIntoGoalEnv,
    SawyerPickOutOfHoleEnv,
    SawyerNutAssemblyEnv,
    SawyerShelfPlaceEnv,
    SawyerPushBackEnv,
    SawyerLeverPullEnv,
    SawyerDialTurnEnv,
    SawyerBinPickingEnv,
    SawyerBoxCloseEnv,
    SawyerHandInsertEnv,
    SawyerDoorLockEnv,
    SawyerDoorUnlockEnv
]


def verify_env_list_space(env_list):
    '''
    This method verifies the action_space and observation_space
    of all environments in env_list are the same.
    '''
    prev_action_space = None
    prev_obs_space = None
    for env_cls in env_list:
        env = env_cls()
        if prev_action_space is None or prev_obs_space is None:
            prev_action_space = env.action_space
            prev_obs_space = env.observation_space
            continue
        assert env.action_space.shape == prev_action_space.shape,\
            '{}, {}, {}'.format(env, env.action_space.shape, prev_action_space)
        assert env.observation_space.shape == prev_obs_space.shape,\
            '{}, {}, {}'.format(env, env.observation_space.shape, prev_obs_space)
        prev_action_space = env.action_space
        prev_obs_space = env.observation_space
