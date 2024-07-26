from alrd.agent.absagent import AgentReset
from alrd.utils.xbox.xbox_joystick_factory import XboxJoystickFactory
from alrd.utils.xbox_spacemouse import XboxSpaceMouse
from alrd.spot_gym.model.robot_state import SpotState
from bosdyn.client.frame_helpers import (
    express_se3_velocity_in_new_frame,
    ODOM_FRAME_NAME,
    BODY_FRAME_NAME,
)
from bosdyn.client.math_helpers import SE3Velocity
from typing import Optional
import numpy as np


class SpotXboxSpacemouse(AgentReset):
    """SpotXboxSpacemouse class provides mapping between xbox controller commands and actions for base and spacemouse commands and actions for end effector of SpotEEVelEnv."""

    def __init__(
        self,
        base_speed: float = 1.0,
        base_angular: float = 1.0,
        ee_speed: float = 0.5,
        ee_angular: float = 0.5,
        ee_control_mode: str = "cartesian",
    ):
        super().__init__()

        # controller object
        self.controller = XboxSpaceMouse()

        # speed parameters
        self.base_speed = base_speed
        self.base_angular = base_angular
        self.ee_speed = ee_speed
        self.ee_angular = ee_angular

        # end effector control mode
        self.ee_control_mode = ee_control_mode

    def _move(
        self,
        left_x,
        left_y,
        right_x,
        right_y,
        sm_forward_backward,
        sm_left_right,
        sm_up_down,
        sm_roll,
        sm_pitch,
        sm_yaw,
        sm_button_1,
        sm_button_2,
        last_state,
    ):
        # set all to 0
        v_x, v_y, v_rot, v_1, v_2, v_3, v_4, v_5, v_6 = 0, 0, 0, 0, 0, 0, 0, 0, 0

        # base linear velocity control
        v_y = -left_x * self.base_speed
        v_x = left_y * self.base_speed

        # base angular velocity control
        v_rot = -right_x * self.base_angular

        # ee linear velocity control
        # if cylindrical: v_1 = v_r (radial), v_2 = v_az (azimuthal), v_3 = v_z
        # if cartesian: v_1 = v_x, v_2 = v_y, v_3 = v_z
        if not sm_button_1:
            v_1 = -sm_left_right * self.ee_speed
            v_2 = -sm_forward_backward * self.ee_speed
            v_3 = sm_up_down * self.ee_speed

        # ee angular velocity control
        # both cylindrical and cartesian: v_4 = vrx, v_5 = vry, v_6 = vrz
        if sm_button_1:
            v_4 = sm_roll * self.ee_angular
            v_5 = sm_pitch * self.ee_angular
            v_6 = sm_yaw * self.ee_angular

        # if cylindrical: return without converting hand velocity
        if self.ee_control_mode == "cylindrical":
            return np.array([v_x, v_y, v_rot, v_1, v_2, v_3, v_4, v_5, v_6])
        # if cartesian: convert hand velocity from body to odom frame
        elif self.ee_control_mode == "cartesian":
            hand_vel_in_body = SE3Velocity(
                lin_x=v_1, lin_y=v_2, lin_z=v_3, ang_x=v_4, ang_y=v_5, ang_z=v_6
            )

            hand_vel_in_odom_proto = express_se3_velocity_in_new_frame(
                last_state.transforms_snapshot,
                ODOM_FRAME_NAME,
                BODY_FRAME_NAME,
                hand_vel_in_body.to_proto(),
            )

            hand_vel_in_odom = hand_vel_in_odom_proto.to_vector()

            return np.array(
                [
                    v_x,
                    v_y,
                    v_rot,
                    hand_vel_in_odom[0],
                    hand_vel_in_odom[1],
                    hand_vel_in_odom[2],
                    hand_vel_in_odom[3],
                    hand_vel_in_odom[4],
                    hand_vel_in_odom[5],
                ]
            )
        else:
            raise NotImplementedError(
                f"End effector control mode {self.ee_control_mode} not implemented."
            )

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
            button_1 + roll     -> End effector x angular velocity
            button_1 + pitch    -> End effector y angular velocity
            button_1 + yaw      -> End effector z angular velocity
        """

    def act(self, obs: np.ndarray, last_state: SpotState) -> Optional[np.ndarray]:
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
            button_1 + roll     -> End effector x angular velocity
            button_1 + pitch    -> End effector y angular velocity
            button_1 + yaw      -> End effector z angular velocity

        Args:
            obs: Observation from the environment.
            last_state: Latest state of the robot.
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
        sm_left_right = spacemouse_actions[1]
        sm_up_down = spacemouse_actions[2]
        sm_roll = spacemouse_actions[5]
        sm_pitch = spacemouse_actions[4]
        sm_yaw = spacemouse_actions[3]
        sm_button_1 = spacemouse_buttons[0]
        sm_button_2 = spacemouse_buttons[1]

        # # exit
        # if self.joy.left_bumper() and self.joy.right_bumper() and self.joy.B():
        #     return None

        return self._move(
            xbox_left_x,
            xbox_left_y,
            xbox_right_x,
            xbox_right_y,
            sm_forward_backward,
            sm_left_right,
            sm_up_down,
            sm_roll,
            sm_pitch,
            sm_yaw,
            sm_button_1,
            sm_button_2,
            last_state,
        )


if __name__ == "__main__":
    agent = SpotXboxSpacemouse()
    print(agent.description())
