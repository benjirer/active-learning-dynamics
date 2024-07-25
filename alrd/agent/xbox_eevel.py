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

    def _move(self, left_x, left_y, right_x, right_y, right_trigger, left_trigger):
        # set all to 0
        v_x, v_y, v_rot, v_1, v_2, v_3, v_4, v_5, v_6 = 0, 0, 0, 0, 0, 0, 0, 0, 0

        # base linear velocity control
        # if left_trigger is not pressed, left_x and left_y are used for base velocity
        if not left_trigger:
            v_y = -left_x * self.base_speed
            v_x = left_y * self.base_speed

        # base angular velocity control
        # if right_trigger is pressed, right_x is used for base rotation
        if right_trigger:
            v_rot = -right_x * self.base_angular

        # end effector linear velocity control
        # if right_trigger is not pressed, right_x and right_y are used for end effector velocity
        # if left_trigger is pressed, left_x is used for v_z
        # if cylindrical: v_1 = v_r (radial), v_2 = v_az (azimuthal), v_3 = v_z
        # if cartesian: v_1 = v_x, v_2 = v_y, v_3 = v_z
        if not right_trigger:
            v_1 = right_y * self.ee_speed
            v_2 = -right_x * self.ee_speed
        if left_trigger:
            v_3 = left_y * self.ee_speed

        # end effector angular velocity control (not used)
        # both cylindrical and cartesian: v_4 = vrx, v_5 = vry, v_6 = vrz
        v_4 = 0
        v_5 = 0
        v_6 = 0

        return np.array([v_x, v_y, v_rot, v_1, v_2, v_3, v_4, v_5, v_6])

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

        # xbox toggles
        right_trigger = self.joy.right_trigger()
        left_trigger = self.joy.left_trigger()

        # exit
        if self.joy.left_bumper() and self.joy.right_bumper() and self.joy.B():
            return None

        return self._move(left_x, left_y, right_x, right_y, right_trigger, left_trigger)
