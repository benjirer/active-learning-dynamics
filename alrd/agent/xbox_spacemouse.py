from alrd.agent.absagent import AgentReset
from alrd.utils.xbox.xbox_joystick_factory import XboxJoystickFactory
from alrd.utils.spacemouse import SpaceMouseExpert
from typing import Optional
import numpy as np


class SpotXboxSpacemouse(AgentReset):
    """SpotXboxSpacemouse class provides mapping between xbox controller commands and actions for base and spacemouse commands and actions for end effector of SpotEEVelEnv."""

    def __init__(
        self, base_speed: float = 1.0, base_angular: float = 1.0, ee_speed: float = 0.5
    ):
        super().__init__()

        # controller objects
        self.joy = XboxJoystickFactory.get_joystick()
        self.mouse = SpaceMouseExpert()

        # speed parameters
        self.base_speed = base_speed
        self.base_angular = base_angular
        self.ee_speed = ee_speed

    def _move(
        self,
        left_x,
        left_y,
        right_x,
        right_y,
        sm_forward_backward,
        sm_left_right,
        sm_up_down,
    ):
        # base velocity control
        v_y = -left_x * self.base_speed
        v_x = left_y * self.base_speed
        v_rot = -right_x * self.base_angular

        # ee velocity control
        v_z = sm_up_down * self.ee_speed
        v_az = sm_left_right * self.ee_speed
        v_r = sm_forward_backward * self.ee_speed

        return np.array([v_x, v_y, v_rot, v_r, v_az, v_z])

    def description(self):
        return """
        Mapping
        Button Combination      -> Functionality
        --------------------------------------
        Xbox:
            LB + RB + B         -> Return None
            Left Stick          -> Body linear velocity
            Right Stick         -> Body angular velocity

        SpaceMouse:
            Forward-Backward    -> End effector radial velocity
            Left-Right          -> End effector azimuthal velocity
            Up-Down             -> End effector vertical velocity
        """

    def act(self, obs: np.ndarray) -> Optional[np.ndarray]:
        """Controls robot base from an xbox controller and end effector from a spacemouse.

        Mapping
        Button Combination      -> Functionality
        --------------------------------------
        Xbox:
            LB + RB + B         -> Return None
            Left Stick          -> Body linear velocity
            Right Stick         -> Body angular velocity

        SpaceMouse:
            Forward-Backward    -> End effector radial velocity
            Left-Right          -> End effector azimuthal velocity
            Up-Down             -> End effector vertical velocity

        Args:
            obs: Observation from the environment.
        """
        # xbox base control
        xbox_left_x = self.joy.left_x()
        xbox_left_y = self.joy.left_y()
        xbox_right_x = self.joy.right_x()
        xbox_right_y = self.joy.right_y()

        # spacemouse end effector control
        actions, buttons = self.mouse.get_action()
        sm_forward_backward = actions[0]
        sm_left_right = actions[5]
        sm_up_down = actions[2]

        # exit
        if self.joy.left_bumper() and self.joy.right_bumper() and self.joy.B():
            return None

        return self._move(
            xbox_left_x,
            xbox_left_y,
            xbox_right_x,
            xbox_right_y,
            sm_forward_backward,
            sm_left_right,
            sm_up_down,
        )
