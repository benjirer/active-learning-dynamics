from alrd.agent.absagent import AgentReset
from alrd.utils.xbox.xbox_joystick_factory import XboxJoystickFactory
from alrd.utils.xbox_spacemouse import XboxSpaceMouse
from typing import Optional
import numpy as np


class SpotXboxSpacemouse(AgentReset):
    """SpotXboxSpacemouse class provides mapping between xbox controller commands and actions for base and spacemouse commands and actions for end effector of SpotEEVelEnv."""

    def __init__(
        self, base_speed: float = 1.0, base_angular: float = 1.0, ee_speed: float = 0.5
    ):
        super().__init__()

        # controller object
        self.controller = XboxSpaceMouse()

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
        v_1 = sm_forward_backward * self.ee_speed
        v_2 = sm_left_right * self.ee_speed
        v_3 = sm_up_down * self.ee_speed

        return np.array([v_x, v_y, v_rot, v_1, v_2, v_3])

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
            Forward-Backward    -> End effector velocity radial for cylindrical or x for cartesian depending on MobilityCommand class used
            Left-Right          -> End effector velocity azimuthal for cylindrical or y for cartesian depending on MobilityCommand class used
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
            Forward-Backward    -> End effector velocity radial for cylindrical or x for cartesian depending on MobilityCommand class used
            Left-Right          -> End effector velocity azimuthal for cylindrical or y for cartesian depending on MobilityCommand class used
            Up-Down             -> End effector vertical velocity

        Args:
            obs: Observation from the environment.
        """

        # get controller state
        spacemouse_actions, spacemouse_buttons, xbox_actions = (
            self.controller.get_action()
        )

        # xbox base control
        xbox_left_x = xbox_actions[0]
        xbox_left_y = xbox_actions[1]
        xbox_right_x = xbox_actions[2]
        xbox_right_y = xbox_actions[3]
        xbox_left_trigger = xbox_actions[4]
        xbox_right_trigger = xbox_actions[5]

        # spacemouse end effector control
        sm_forward_backward = spacemouse_actions[0]
        sm_left_right = spacemouse_actions[5]
        sm_up_down = spacemouse_actions[2]

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
