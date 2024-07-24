from alrd.agent.absagent import AgentReset
from alrd.utils.xbox.xbox_joystick_factory import XboxJoystickFactory
from typing import Optional
import numpy as np
from opax.optimizers.icem_trajectory_optimizer import powerlaw_psd_gaussian_numpy
from alrd.spot_gym.utils.utils import MAX_ARM_JOINT_VEL


class SpotXboxRandomJointPos(AgentReset):
    """SpotXboxRandomJointPos class provides mapping between xbox controller commands and actions for base and randomly sampled arm joint dq commands and actions for arm joints of SpotJointPosEnv."""

    def __init__(
        self,
        base_speed: float = 1.0,
        base_angular: float = 1.0,
        arm_joint_speed: float = 0.5,
        cmd_freq: float = 10,
        steps: int = 1000,
        random_seed: int = 0,
    ):
        super().__init__()

        # controller objects
        self.joy = XboxJoystickFactory.get_joystick()

        # speed parameters
        self.base_speed = base_speed
        self.base_angular = base_angular
        self.arm_joint_speed = arm_joint_speed

        # random arm joint dq series
        # generates one series of colored noise samples for arm joint dq for the entire current episode given number of steps
        arm_joint_dq_series_pre = powerlaw_psd_gaussian_numpy(
            exponent=1.0,  # pink noise
            size=steps * 6,
            random_state=random_seed,
        )
        # scale samples to MAX_JOINT_DQ
        safety_factor = 0.1
        MAX_JOINT_DQ = safety_factor * MAX_ARM_JOINT_VEL / cmd_freq
        max_sample_value = np.max(np.abs(arm_joint_dq_series_pre))
        self.arm_joint_dq_series = (
            arm_joint_dq_series_pre * MAX_JOINT_DQ / max_sample_value
        )
        # clip just in case
        self.arm_joint_dq_series = np.clip(
            self.arm_joint_dq_series, -MAX_JOINT_DQ, MAX_JOINT_DQ
        )
        self.arm_joint_dq_series_index = 0

    def _move(
        self,
        left_x,
        left_y,
        right_x,
        right_y,
        sh0,
        sh1,
        el0,
        el1,
        wr0,
        wr1,
    ):
        v_x = 0
        v_y = 0
        v_rot = 0
        # base velocity control
        # v_y = -left_x * self.base_speed
        # v_x = left_y * self.base_speed
        # v_rot = -right_x * self.base_angular

        # arm joint dq control
        # for testing, only move selected joints
        sh0 = 0.0
        sh1 = 0.0
        el0 = 0.0
        el1 = 0.0
        wr0 = 0.0
        wr1 = 0.0

        sh1 = left_y * 0.05
        wr0 = right_y * 0.05

        return np.array([v_x, v_y, v_rot, sh0, sh1, el0, el1, wr0, wr1])

    def description(self):
        return """
        Mapping
        Button Combination      -> Functionality
        --------------------------------------
        Xbox:
            LB + RB + B         -> Return None
            Left Stick          -> Body linear velocity
            Right Stick         -> Body angular velocity
        """

    def act(self, obs: np.ndarray) -> Optional[np.ndarray]:
        """Controls robot base from an xbox controller and arm joint dq from colored noise samples.

        Mapping
        Button Combination      -> Functionality
        --------------------------------------
        Xbox:
            LB + RB + B         -> Return None
            Left Stick          -> Body linear velocity
            Right Stick         -> Body angular velocity

        Args:
            obs: Observation from the environment.
        """
        # xbox base control
        xbox_left_x = self.joy.left_x()
        xbox_left_y = self.joy.left_y()
        xbox_right_x = self.joy.right_x()
        xbox_right_y = self.joy.right_y()

        # random arm joint dq control
        next_index = self.arm_joint_dq_series_index + 6
        current_arm_dq = self.arm_joint_dq_series[
            self.arm_joint_dq_series_index : next_index
        ]
        self.arm_joint_dq_series_index = next_index

        # exit
        if self.joy.left_bumper() and self.joy.right_bumper() and self.joy.B():
            return None

        return self._move(
            xbox_left_x,
            xbox_left_y,
            xbox_right_x,
            xbox_right_y,
            current_arm_dq[0],
            current_arm_dq[1],
            current_arm_dq[2],
            current_arm_dq[3],
            current_arm_dq[4],
            current_arm_dq[5],
        )
