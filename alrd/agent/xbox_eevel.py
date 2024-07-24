from alrd.agent.absagent import AgentReset
from alrd.utils.xbox.xbox_joystick_factory import XboxJoystickFactory
from typing import Optional
import numpy as np


class SpotXboxEEVel(AgentReset):
    """SpotXboxEEVel class provides mapping between xbox controller commands and SpotEEVelEnv actions."""

    def __init__(
        self, base_speed: float = 1.0, base_angular: float = 1.0, ee_speed: float = 0.5
    ):
        super().__init__()

        # controller objects
        self.joy = XboxJoystickFactory.get_joystick()

        # speed parameters
        self.base_speed = base_speed
        self.base_angular = base_angular
        self.ee_speed = ee_speed

    def _move(self, left_x, left_y, right_x, right_y):
        # base velocity control
        v_y = -left_x * self.base_speed
        v_x = left_y * self.base_speed

        # base rotation control (not used as right stick is used for end effector)
        v_rot = 0
        # v_rot = -right_x * self.base_angular

        # end effector velocity control
        # if cylindrical: v_1 = v_r (radial), v_2 = v_az (azimuthal)
        # if cartesian: v_1 = v_x, v_2 = v_y
        v_1 = right_y * self.ee_speed
        v_2 = right_x * self.ee_speed
        return np.array([v_x, v_y, v_rot, v_1, v_2, 0])

    def description(self):
        return """
        Mapping
        Button Combination      -> Functionality
        --------------------------------------
        LB + RB + B             -> Return None
        Left Stick              -> Body linear velocity
        Right Stick             -> End effector velocity (cylindrical or cartesian depending on MobilityCommand class used)
        """

    def act(self, obs: np.ndarray) -> Optional[np.ndarray]:
        """Controls robot base and end effector from an xbox controller.

        Mapping
        Button Combination      -> Functionality
        --------------------------------------
        LB + RB + B             -> Return None
        Left Stick              -> Body linear velocity
        Right Stick             -> End effector velocity (cylindrical or cartesian depending on MobilityCommand class used)

        Args:
            obs: Observation from the environment.
        """

        # xbox base control
        left_x = self.joy.left_x()
        left_y = self.joy.left_y()

        # xbox end effector control
        right_x = self.joy.right_x()
        right_y = self.joy.right_y()

        # exit
        if self.joy.left_bumper() and self.joy.right_bumper() and self.joy.B():
            return None

        return self._move(left_x, left_y, right_x, right_y)
