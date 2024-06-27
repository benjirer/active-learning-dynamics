from alrd.agent import AgentReset
from alrd.utils.xbox.xbox_joystick_factory import XboxJoystickFactory
from alrd.utils.spacemouse import SpaceMouseExpert
from typing import Optional
import numpy as np


class SpotSpaceBox(AgentReset):
    """SpotSpaceBox class provides mapping between:
    - Xbox controller commands and Spot2D actions for base
    - SpaceMouse commands and Spot2D actions for arm
    """

    def __init__(
        self, base_speed: float = 1.0, base_angular: float = 1.0, arm_speed: float = 0.5
    ):
        super().__init__()

        # controller objects
        self.joy = XboxJoystickFactory.get_joystick()
        self.mouse = SpaceMouseExpert()

        # speed parameters
        self.base_speed = base_speed
        self.base_angular = base_angular
        self.arm_speed = arm_speed

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
        """Commands the robot and arm using:
        - Xbox left stick for robot base linear velocity
        - Xbox right stick for robot base angular velocity
        - SpaceMouse:
            - forward-backward for v_r radial velocity
            - left-right for v_az azimuthal velocity
            - up-down for v_z vertical velocity
            - TODO: turn for v_rot_wrist wrist rotation velocity

        Args:
            left_x: Xbox left stick x-axis value
            left_y: Xbox left stick y-axis value
            right_x: Xbox right stick x-axis value
            right_y: Xbox right stick y-axis value
            sm_forward_backward: SpaceMouse forward-backward value
            sm_left_right: SpaceMouse left-right value
            sm_up_down: SpaceMouse up-down value
        """

        # Stick left_x controls robot v_y
        v_y = -left_x * self.base_speed

        # Stick left_y controls robot v_x
        v_x = left_y * self.base_speed

        # Stick right_x controls robot v_rot
        v_rot = -right_x * self.base_angular

        # SpaceMouse forward-backward controls arm v_r
        v_r = -sm_forward_backward * self.arm_speed

        # SpaceMouse left-right controls arm v_az
        v_az = -sm_left_right * self.arm_speed

        # SpaceMouse up-down controls arm v_z
        v_z = sm_up_down * self.arm_speed

        return np.array([v_x, v_y, v_rot, v_r, v_az, v_z])

    def description(self):
        return """
        Mapping
        Button Combination    -> Functionality
        --------------------------------------
        Xbox:
        LB + RB + B           -> Return None
          Left Stick          -> Move
          Right Stick         -> Rotate

        SpaceMouse:
            Forward-Backward    -> v_r radial velocity
            Left-Right          -> v_az azimuthal velocity
            Up-Down             -> v_z vertical velocity
        """

    def act(self, obs: np.ndarray) -> Optional[np.ndarray]:
        """Controls robot base from an Xbox controller and arm from a Spacemouse.

        Mapping
        Button Combination    -> Functionality
        --------------------------------------
        Xbox:
        LB + RB + B           -> Return None
          Left Stick          -> Move
          Right Stick         -> Rotate

        SpaceMouse:
            Forward-Backward    -> v_r radial velocity
            Left-Right          -> v_az azimuthal velocity
            Up-Down             -> v_z vertical velocity

        Args:
            obs: Observation from the environment.
        """
        # xbox base control
        xbox_left_x = self.joy.left_x()
        xbox_left_y = self.joy.left_y()
        xbox_right_x = self.joy.right_x()
        xbox_right_y = self.joy.right_y()

        # spacemouse arm control
        actions, buttons = self.mouse.get_action()
        sm_forward_backward = actions[0]
        sm_left_right = actions[5]
        sm_up_down = actions[2]

        # sm_forward_backward = 0
        # sm_left_right = 0
        # sm_up_down = 0

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
