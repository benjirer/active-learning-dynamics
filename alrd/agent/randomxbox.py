from alrd.agent import AgentReset
from alrd.utils.xbox.xbox_joystick_factory import XboxJoystickFactory
from typing import Optional
import numpy as np
from opax.optimizers.icem_trajectory_optimizer import powerlaw_psd_gaussian_numpy
from alrd.spot_gym.utils.utils import MAX_ARM_JOINT_VEL


class SpotRandomXbox(AgentReset):
    """SpotSpaceBox class provides mapping between:
    - Xbox controller commands and Spot2D actions for base
    - Random sample from colored noise for arm dq
    """

    def __init__(
        self,
        base_speed: float = 1.0,
        base_angular: float = 1.0,
        arm_speed: float = 0.5,
        cmd_freq: float = 10,
        steps: int = 1000,
    ):
        super().__init__()

        # controller objects
        self.joy = XboxJoystickFactory.get_joystick()

        # speed parameters
        self.base_speed = base_speed
        self.base_angular = base_angular
        self.arm_speed = arm_speed

        # random arm dq series
        # generates one series of colored noise samples for arm dq for the entire current episode given number of steps
        # TODO: pass these arguments at construction, especially use seed generator to get new seed at each episode in run_spot, not here
        rng = np.random.default_rng(seed=0)
        arm_dq_series_pre = powerlaw_psd_gaussian_numpy(
            exponent=1.0,  # pink noise
            size=steps * 6,
            random_state=rng,
        )
        # scale samples to MAX_JOINT_DQ
        # TODO: this is not so elegant
        safety_factor = 0.1
        MAX_JOINT_DQ = safety_factor * MAX_ARM_JOINT_VEL / cmd_freq
        max_sample_value = np.max(np.abs(arm_dq_series_pre))
        self.arm_dq_series = arm_dq_series_pre * MAX_JOINT_DQ / max_sample_value
        # clip just in case
        self.arm_dq_series = np.clip(self.arm_dq_series, -MAX_JOINT_DQ, MAX_JOINT_DQ)
        self.arm_dq_series_index = 0

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
        """Commands the robot and arm using:
        - Xbox left stick for robot base linear velocity
        - Xbox right stick for robot base angular velocity
        - Random sample of position dq for arm joints from colored noise

        Args:
            left_x: Xbox left stick x-axis value
            left_y: Xbox left stick y-axis value
            right_x: Xbox right stick x-axis value
            right_y: Xbox right stick y-axis value
            sh0: arm joint 0 dq
            sh1: arm joint 1 dq
            el0: arm joint 2 dq
            el1: arm joint 3 dq
            wr0: arm joint 4 dq
            wr1: arm joint 5 dq

        Returns:
            np.ndarray: Action for the robot and arm
                v_x: robot base x-axis linear velocity
                v_y: robot base y-axis linear velocity
                v_rot: robot base angular velocity
                sh0: arm joint 0 dq
                sh1: arm joint 1 dq
                el0: arm joint 2 dq
                el1: arm joint 3 dq
                wr0: arm joint 4 dq
                wr1: arm joint 5 dq
        """

        # control robot base commands
        # Stick left_x controls robot v_y
        v_y = -left_x * self.base_speed

        # Stick left_y controls robot v_x
        v_x = left_y * self.base_speed

        # Stick right_x controls robot v_rot
        v_rot = -right_x * self.base_angular

        # for testing, only move sh0
        # sh0 = 0.0
        sh1 = 0.0
        el0 = 0.0
        el1 = 0.0
        # wr0 = 0.0
        # wr1 = 0.0

        return np.array([v_x, v_y, v_rot, sh0, sh1, el0, el1, wr0, wr1])

    def description(self):
        return """Controls robot base from an Xbox controller and arm dq from colored noise samples.

        Mapping
        Button Combination    -> Functionality
        --------------------------------------
        Xbox:
        LB + RB + B           -> Return None
          Left Stick          -> Move
          Right Stick         -> Rotate
        """

    def act(self, obs: np.ndarray) -> Optional[np.ndarray]:
        """Controls robot base from an Xbox controller and arm dq from colored noise samples.

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

        # random arm dq
        next_index = self.arm_dq_series_index + 6
        current_arm_dq = self.arm_dq_series[self.arm_dq_series_index : next_index]
        self.arm_dq_series_index = next_index

        # exit command
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
