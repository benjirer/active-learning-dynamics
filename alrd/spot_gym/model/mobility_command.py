from __future__ import annotations
from copy import copy
import numpy as np
import logging
import csv
from alrd.spot_gym.model.command import Command, CommandEnum, LocomotionHint
from alrd.spot_gym.model.robot_state import SpotState
from bosdyn.client.robot_command import RobotCommandBuilder
from bosdyn.geometry import EulerZXY
from bosdyn.api import arm_command_pb2, robot_command_pb2
from alrd.spot_gym.utils.utils import (
    MAX_ARM_JOINT_VEL,
    SH0_POS_MIN,
    SH0_POS_MAX,
    SH1_POS_MIN,
    SH1_POS_MAX,
    EL0_POS_MIN,
    EL0_POS_MAX,
    EL1_POS_MIN,
    EL1_POS_MAX,
    WR0_POS_MIN,
    WR0_POS_MAX,
    WR1_POS_MIN,
    WR1_POS_MAX,
)
from alrd.spot_gym.utils.spot_arm_fk import SpotArmFK
from alrd.spot_gym.utils.spot_arm_ik import SpotArmIK
from dataclasses import asdict, dataclass

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)


@dataclass
class MobilityCommand(Command):
    cmd_type = CommandEnum.MOBILITY

    # forward and inverse kinematics
    spot_arm_fk: SpotArmFK = SpotArmFK()
    spot_arm_ik: SpotArmIK = SpotArmIK()

    # previous robot state
    prev_state: SpotState

    # command frequency
    cmd_freq: float

    # body commands
    vx: float
    vy: float
    w: float
    height: float
    pitch: float
    locomotion_hint: LocomotionHint
    stair_hint: bool

    # arm commands
    sh0_dq: float
    sh1_dq: float
    el0_dq: float
    el1_dq: float
    wr0_dq: float
    wr1_dq: float

    # commanded arm joint positions
    commanded_arm_joint_positions: np.ndarray = np.zeros(6)

    # safety check infringed
    safety_check_infringed: bool = False

    def __post_init__(self) -> None:

        # make body command
        orientation = EulerZXY(roll=0.0, pitch=self.pitch, yaw=0.0)
        mobility_params = RobotCommandBuilder.mobility_params(
            body_height=self.height,
            footprint_R_body=orientation,
            locomotion_hint=self.locomotion_hint,
            stair_hint=self.stair_hint,
        )

        # make arm command
        # only if arm dq commands are not zero and previous state is not None
        if (
            self.sh0_dq != 0.0
            or self.sh1_dq != 0.0
            or self.el0_dq != 0.0
            or self.el1_dq != 0.0
            or self.wr0_dq != 0.0
            or self.wr1_dq != 0.0
            and self.prev_state is not None
        ):
            # as the API only allows for arm joint position commands, the agent commands the arm joint dq and here we command q_new (q_new = q_prev + dq) via the robot command
            joint_positions_prev = self.prev_state.arm_joint_positions
            logger.info("Joint positions prev: {}".format(joint_positions_prev))

            joint_positions_new_pre = [
                joint_positions_prev[0] + self.sh0_dq,
                joint_positions_prev[1] + self.sh1_dq,
                joint_positions_prev[2] + self.el0_dq,
                joint_positions_prev[3] + self.el1_dq,
                joint_positions_prev[4] + self.wr0_dq,
                joint_positions_prev[5] + self.wr1_dq,
            ]

            joint_positions_new = [copy(joint_positions_prev[i]) for i in range(6)]

            # perform three safety checks:
            # 1. check if joint limits are exceeded
            # 2. check if end effector height is at least 0.1m
            # 3. use forward kinematics to check if end effector height is still at least 0.1m
            # note: self collision is handled by the robot

            # # 1. check joint limits
            # joint_positions_new[0] = max(
            #     SH0_POS_MIN, min(SH0_POS_MAX, joint_positions_new_pre[0])
            # )
            # joint_positions_new[1] = max(
            #     SH1_POS_MIN, min(SH1_POS_MAX, joint_positions_new_pre[1])
            # )
            # joint_positions_new[2] = max(
            #     EL0_POS_MIN, min(EL0_POS_MAX, joint_positions_new_pre[2])
            # )
            # joint_positions_new[3] = max(
            #     EL1_POS_MIN, min(EL1_POS_MAX, joint_positions_new_pre[3])
            # )
            # joint_positions_new[4] = max(
            #     WR0_POS_MIN, min(WR0_POS_MAX, joint_positions_new_pre[4])
            # )
            # joint_positions_new[5] = max(
            #     WR1_POS_MIN, min(WR1_POS_MAX, joint_positions_new_pre[5])
            # )

            # if joint_positions_new != joint_positions_new_pre:
            #     self.safety_check_infringed = True
            #     logger.info("Safety check infringed due to joint limit")

            joint_positions_new = joint_positions_new_pre

            # 2. check if end effector height is at least 0.1m
            if self.prev_state.pose_of_hand[2] <= 0.1:
                joint_positions_new = joint_positions_prev
                self.safety_check_infringed = True
                logger.info(
                    "Safety check infringed due to height. Joint position command reset to previous position."
                )

            # 3. use forward kinematics to check if end effector height is still at least 0.1m
            # TODO: test this
            hand_pose_fk = self.spot_arm_fk.get_ee_position(joint_positions_new)
            if hand_pose_fk[2] <= 0.1:
                joint_positions_new = self.spot_arm_ik.compute_ik(
                    current_arm_joint_states=joint_positions_prev,
                    ee_target=[hand_pose_fk[0], hand_pose_fk[1], 0.11],
                )
                self.safety_check_infringed = True
                logger.info(
                    "Safety check infringed due to height. Joint position command reset using inverse kinematics."
                )

            # make arm joint command
            self.commanded_arm_joint_positions = joint_positions_new

            arm_cmd = RobotCommandBuilder.arm_joint_command(
                sh0=joint_positions_new[0],
                sh1=joint_positions_new[1],
                el0=joint_positions_new[2],
                el1=joint_positions_new[3],
                wr0=joint_positions_new[4],
                wr1=joint_positions_new[5],
                # max joint velocity is passed as safety
                max_vel=MAX_ARM_JOINT_VEL,
            )

            # make synchro velocity command with arm as build_on_command
            cmd = RobotCommandBuilder.synchro_velocity_command(
                v_x=self.vx,
                v_y=self.vy,
                v_rot=self.w,
                params=mobility_params,
                build_on_command=arm_cmd,
            )
        else:
            # make synchro velocity command (no arm command provided)
            cmd = RobotCommandBuilder.synchro_velocity_command(
                v_x=self.vx,
                v_y=self.vy,
                v_rot=self.w,
                params=mobility_params,
            )

        super().__init__(cmd)

    def __array__(self, dtype=None) -> np.ndarray:
        return np.array(
            [
                self.vx,
                self.vy,
                self.w,
                self.height,
                self.pitch,
                self.locomotion_hint,
                self.stair_hint,
                self.sh0_dq,
                self.sh1_dq,
                self.el0_dq,
                self.el1_dq,
                self.wr0_dq,
                self.wr1_dq,
            ],
            dtype=dtype,
        )

    @staticmethod
    def fromarray(arr: np.ndarray) -> MobilityCommand:
        return MobilityCommand(
            prev_state=arr[0],
            cmd_freq=arr[1],
            vx=arr[2],
            vy=arr[3],
            w=arr[4],
            height=arr[5],
            pitch=arr[6],
            locomotion_hint=arr[7],
            stair_hint=arr[8],
            sh0_dq=arr[9],
            sh1_dq=arr[10],
            el0_dq=arr[11],
            el1_dq=arr[12],
            wr0_dq=arr[13],
            wr1_dq=arr[14],
        )

    def asdict(self) -> dict:
        return {**super().asdict(), **asdict(self)}

    @staticmethod
    def fromdict(d: dict) -> MobilityCommand:
        return MobilityCommand(
            prev_state=d["prev_state"],
            cmd_freq=d["cmd_freq"],
            vx=d["vx"],
            vy=d["vy"],
            w=d["w"],
            height=d["height"],
            pitch=d["pitch"],
            locomotion_hint=d["locomotion_hint"],
            stair_hint=d["stair_hint"],
            sh0_dq=d["sh0_dq"],
            sh1_dq=d["sh1_dq"],
            el0_dq=d["el0_dq"],
            el1_dq=d["el1_dq"],
            wr0_dq=d["wr0_dq"],
            wr1_dq=d["wr1_dq"],
        )

    def to_str(self) -> str:
        s = "commands {\n"
        s += "\tx: {:.5f}\n".format(self.vx)
        s += "\ty: {:.5f}\n".format(self.vy)
        s += "\trot: {:.5f}\n".format(self.w)
        s += "\theight: {:.5f}\n".format(self.height)
        s += "\tpitch: {:.5f}\n".format(self.pitch)
        s += "\tlocomotion_hint: {}\n".format(self.locomotion_hint)
        s += "\tstair_hint: {}\n".format(self.stair_hint)
        s += "\tsh0_dq: {:.5f}\n".format(self.sh0_dq)
        s += "\tsh1_dq: {:.5f}\n".format(self.sh1_dq)
        s += "\tel0_dq: {:.5f}\n".format(self.el0_dq)
        s += "\tel1_dq: {:.5f}\n".format(self.el1_dq)
        s += "\twr0_dq: {:.5f}\n".format(self.wr0_dq)
        s += "\twr1_dq: {:.5f}\n".format(self.wr1_dq)
        s += "}"
        s += "metadata {\n"
        s += "safe_check_infringed: {}\n".format(self.safety_check_infringed)
        s += "commanded_sh0: {:.5f}\n".format(self.commanded_arm_joint_positions[0])
        s += "commanded_sh1: {:.5f}\n".format(self.commanded_arm_joint_positions[1])
        s += "commanded_el0: {:.5f}\n".format(self.commanded_arm_joint_positions[2])
        s += "commanded_el1: {:.5f}\n".format(self.commanded_arm_joint_positions[3])
        s += "commanded_wr0: {:.5f}\n".format(self.commanded_arm_joint_positions[4])
        s += "commanded_wr1: {:.5f}\n".format(self.commanded_arm_joint_positions[5])
        s += "}"

        return s

    @staticmethod
    def size() -> int:
        return 15
