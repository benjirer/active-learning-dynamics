from alrd.agent import AgentReset
from alrd.utils.xbox.xbox_joystick_factory import XboxJoystickFactory
from typing import Optional
import numpy as np


class SpotXbox2D(AgentReset):
    """SpotXbox2D class provides mapping between xbox controller commands and Spot2D actions."""

    def __init__(
        self, base_speed: float = 1.0, base_angular: float = 1.0, arm_speed: float = 0.5
    ):
        super().__init__()
        self.joy = XboxJoystickFactory.get_joystick()
        self.base_speed = base_speed
        self.base_angular = base_angular
        self.arm_speed = arm_speed

    def _move(self, left_x, left_y, right_x, right_y):
        """Commands the robot with a velocity command based on left stick values.
        Commands the robot arm velicoty command based on right stick values.

        Args:
            left_x: X value of left stick.
            left_y: Y value of left stick.
            right_x: X value of right stick.
        """

        # Stick left_x controls robot v_y
        v_y = -left_x * self.base_speed

        # Stick left_y controls robot v_x
        v_x = left_y * self.base_speed

        v_rot = 0
        # v_rot = -right_x * self.base_angular

        # Stick right_y controls arm v_r
        v_r = right_y * self.arm_speed

        v_az = right_x * self.arm_speed
        return np.array([v_x, v_y, v_rot, v_r, v_az, 0])

    def description(self):
        return """
        Xbox control:
        --------------
        LB + RB + B: Exit
        Left Stick: Move
        Right Stick: Arm
        """

    def act(self, obs: np.ndarray) -> Optional[np.ndarray]:
        """Controls robot from an Xbox controller.

        Mapping
        Button Combination    -> Functionality
        --------------------------------------
        LB + RB + B           -> Return None
          Left Stick          -> Move
          Right Stick         -> Arm

        Args:
            frequency: Max frequency to send commands to robot
        """

        left_x = self.joy.left_x()
        left_y = self.joy.left_y()
        right_x = self.joy.right_x()
        right_y = self.joy.right_y()

        if self.joy.left_bumper() and self.joy.right_bumper() and self.joy.B():
            return None

        if left_x != 0.0 or left_y != 0.0 or right_x != 0.0 or right_y != 0.0:
            return self._move(left_x, left_y, right_x, right_y)
        else:
            return self._move(0.0, 0.0, 0.0, 0.0)
