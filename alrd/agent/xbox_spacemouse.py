from alrd.agent.absagent import AgentReset
from alrd.utils.xbox.xbox_joystick_factory import XboxJoystickFactory
from alrd.utils.xbox_spacemouse import XboxSpaceMouse
from alrd.spot_gym.model.robot_state import SpotState
from bosdyn.client.frame_helpers import (
    express_se3_velocity_in_new_frame,
    ODOM_FRAME_NAME,
    BODY_FRAME_NAME,
)
from opax.optimizers.icem_trajectory_optimizer import powerlaw_psd_gaussian_numpy
from bosdyn.client.math_helpers import SE3Velocity
from typing import Optional
import numpy as np
import pickle


class SpotXboxSpacemouse(AgentReset):
    """SpotXboxSpacemouse class provides mapping between xbox controller commands and actions for base and spacemouse commands and actions for end effector of SpotEEVelEnv."""

    def __init__(
        self,
        base_speed: float = 1.0,
        base_angular: float = 1.0,
        ee_speed: float = 1.0,
        ee_angular: float = 1.0,
        ee_control_mode: str = "basic",
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

        # # create random commands for end effector roll, pitch, yaw
        # np.random.seed(8)
        # # combined_array = np.random.uniform(-1, 1, 100000 * 3)
        # from scipy.stats import truncnorm

        # samples = truncnorm.rvs(-1, 1, loc=0, scale=1, size=100000)
        # combined_array = np.repeat(samples, 8)
        # pink_noise = powerlaw_psd_gaussian_numpy(exponent=1, size=len(combined_array))
        # pink_noise /= np.max(np.abs(pink_noise))
        # pink_noise *= 0.2
        # combined_array = np.clip(combined_array + pink_noise, -1, 1)
        # combined_array = combined_array[: len(combined_array) // 3 * 3]
        # self.ee_vrx, self.ee_vry, self.ee_vrz = np.split(combined_array, 3)

        # import actions for ee angular velocity from prerecorded data
        ee_ori_actions_traj_path = "/home/bhoffman/Documents/MT FS24/active-learning-dynamics/alrd/ee_ori_action_traj.pickle"
        ee_ori_actions_traj = pickle.load(open(ee_ori_actions_traj_path, "rb"))
        self.ee_vrx, self.ee_vry, self.ee_vrz = (
            ee_ori_actions_traj[..., 6],
            ee_ori_actions_traj[..., 7],
            ee_ori_actions_traj[..., 8],
        )

        self.idx = 0
        self.max_idx = len(self.ee_vrx)

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
        if self.ee_control_mode == "basic":
            v_x, v_y, v_rot, v_1, v_2, v_3 = 0, 0, 0, 0, 0, 0
        elif self.ee_control_mode == "augmented":
            v_x, v_y, v_rot, v_1, v_2, v_3, v_4, v_5, v_6 = 0, 0, 0, 0, 0, 0, 0, 0, 0
        else:
            raise NotImplementedError(
                f"End effector control mode {self.ee_control_mode} not implemented."
            )

        # base linear velocity control
        v_y = -left_x * self.base_speed
        v_x = left_y * self.base_speed

        # base angular velocity control
        v_rot = -right_x * self.base_angular

        # ee linear velocity control
        # if cylindrical: v_1 = v_r (radial), v_2 = v_az (azimuthal), v_3 = v_z
        # if cartesian: v_1 = v_x, v_2 = v_y, v_3 = v_z
        if not sm_button_1 or self.ee_control_mode == "basic":
            v_1 = -sm_left_right * self.ee_speed
            v_2 = -sm_forward_backward * self.ee_speed
            v_3 = sm_up_down * self.ee_speed

        # # ee angular velocity control
        # # both cylindrical and cartesian: v_4 = vrx, v_5 = vry, v_6 = vrz
        # if sm_button_1 and self.ee_control_mode == "augmented":
        #     v_4 = sm_roll * self.ee_angular
        #     v_5 = sm_pitch * self.ee_angular
        #     v_6 = sm_yaw * self.ee_angular

        # get random command
        # v_4 = self.ee_vrx[self.idx] * self.ee_angular
        # v_5 = self.ee_vry[self.idx] * self.ee_angular
        # v_6 = self.ee_vrz[self.idx] * self.ee_angular

        # switch direction every 20 steps
        # sign = 1 if self.idx % 20 < 10 else -1
        # v_4 = 0.5 * sign
        # v_5 = 0
        # v_6 = 0

        # self.idx += 1

        # # use joystick to control end effector angular velocity
        # v_4 = left_x * self.ee_angular
        # v_5 = left_y * self.ee_angular
        # v_6 = right_x * self.ee_angular

        # get from prerecorded data
        v_4 = self.ee_vrx[self.idx]
        v_5 = self.ee_vry[self.idx]
        v_6 = self.ee_vrz[self.idx]

        if self.idx < self.max_idx - 1:
            self.idx += 1

        # if basic: return only linear velocities for ee
        if self.ee_control_mode == "basic":
            return np.array([v_x, v_y, v_rot, v_1, v_2, v_3])
        # if augmented: return both linear and angular velocities for ee
        elif self.ee_control_mode == "augmented":
            return np.array([v_x, v_y, v_rot, v_1, v_2, v_3, v_4, v_5, v_6])

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
            button_1 + roll     -> End effector x angular velocity (only in augmented mode)
            button_1 + pitch    -> End effector y angular velocity (only in augmented mode)
            button_1 + yaw      -> End effector z angular velocity (only in augmented mode)
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
            button_1 + roll     -> End effector x angular velocity (only in augmented mode)
            button_1 + pitch    -> End effector y angular velocity (only in augmented mode)
            button_1 + yaw      -> End effector z angular velocity (only in augmented mode)

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
        sm_left_right = spacemouse_actions[0]
        sm_forward_backward = spacemouse_actions[1]
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
    # agent = SpotXboxSpacemouse()
    # print(agent.description())
    # create random commands for end effector roll, pitch, yaw
    np.random.seed(42)
    # combined_array = np.random.uniform(-1, 1, 100000 * 3)
    from scipy.stats import truncnorm

    samples = truncnorm.rvs(-1, 1, loc=0, scale=1, size=200)
    combined_array = np.repeat(samples, 5)
    pink_noise = powerlaw_psd_gaussian_numpy(exponent=1, size=len(combined_array))
    pink_noise /= np.max(np.abs(pink_noise))
    pink_noise *= 0.2
    combined_array = np.clip(combined_array + pink_noise, -1, 1)
    # make sure we can split the array into 3
    combined_array = combined_array[: len(combined_array) // 3 * 3]
    ee_vrx, ee_vry, ee_vrz = np.split(combined_array, 3)
    # plot velocities in subplots
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.subplot(311)
    plt.plot(ee_vrx)
    plt.title("ee_vrx")
    plt.subplot(312)
    plt.plot(ee_vry)
    plt.title("ee_vry")
    plt.subplot(313)
    plt.plot(ee_vrz)
    plt.title("ee_vrz")

    plt.legend()
    plt.savefig("ee_v.png")
    plt.show()
