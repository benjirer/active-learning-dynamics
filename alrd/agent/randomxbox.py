from alrd.agent import AgentReset
from alrd.utils.xbox.xbox_joystick_factory import XboxJoystickFactory
from typing import Optional
import numpy as np


class SpotRandomXbox(AgentReset):
    """SpotSpaceBox class provides mapping between:
    - Xbox controller commands and Spot2D actions for base
    - Random sample of velocities for arm joints from colored noise
    """

    def __init__(
        self, base_speed: float = 1.0, base_angular: float = 1.0, arm_speed: float = 0.5
    ):
        super().__init__()

        # controller objects
        self.joy = XboxJoystickFactory.get_joystick()

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
    ):
        """Commands the robot and arm using:
        - Xbox left stick for robot base linear velocity
        - Xbox right stick for robot base angular velocity
        - Random sample of velocities for arm joints from colored noise

        Args:
            left_x: Xbox left stick x-axis value
            left_y: Xbox left stick y-axis value
            right_x: Xbox right stick x-axis value
            right_y: Xbox right stick y-axis value

        Returns:
            np.ndarray: Action for the robot and arm
                v_x: robot base x-axis linear velocity
                v_y: robot base y-axis linear velocity
                v_rot: robot base angular velocity
                sh0: arm joint 0 velocity
                sh1: arm joint 1 velocity
                el0: arm joint 2 velocity
                el1: arm joint 3 velocity
                wr0: arm joint 4 velocity
                wr1: arm joint 5 velocity
        """

        # Stick left_x controls robot v_y
        v_y = -left_x * self.base_speed

        # Stick left_y controls robot v_x
        # v_x = left_y * self.base_speed
        v_x = 0

        # Stick right_x controls robot v_rot
        v_rot = -right_x * self.base_angular

        # test using right_y for first arm joint
        # sh0 = right_y * self.arm_speed / 10
        sh0 = -left_y * self.arm_speed

        return np.array([v_x, v_y, v_rot, sh0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def description(self):
        return """
        Mapping
        Button Combination    -> Functionality
        --------------------------------------
        Xbox:
        LB + RB + B           -> Return None
          Left Stick          -> Move
          Right Stick         -> Rotate
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

        Args:
            obs: Observation from the environment.
        """
        # xbox base control
        xbox_left_x = self.joy.left_x()
        xbox_left_y = self.joy.left_y()
        xbox_right_x = self.joy.right_x()
        xbox_right_y = self.joy.right_y()

        # exit
        if self.joy.left_bumper() and self.joy.right_bumper() and self.joy.B():
            return None

        return self._move(
            xbox_left_x,
            xbox_left_y,
            xbox_right_x,
            xbox_right_y,
        )
