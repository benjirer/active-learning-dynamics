from __future__ import annotations
from copy import copy
import numpy as np
import logging
from alrd.spot_gym.model.command import Command, CommandEnum, LocomotionHint
from alrd.spot_gym.model.robot_state import SpotState
from bosdyn.client.robot_command import RobotCommandBuilder
from bosdyn.client.frame_helpers import BODY_FRAME_NAME
from bosdyn.geometry import EulerZXY
from bosdyn.api import arm_command_pb2, robot_command_pb2
from bosdyn.api.geometry_pb2 import Vec3
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
    """
    This MobilityCommand class provides mapping between SpotEEVelEnv action and robot command for body velocity and end effector cartesian velocity commands.
    """

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

    # hand commands
    hand_vx: float
    hand_vy: float
    hand_vz: float
    hand_vrx: float
    hand_vry: float
    hand_vrz: float

    # command type
    cmd_type = CommandEnum.MOBILITY

    # forward and inverse kinematics
    spot_arm_fk: SpotArmFK = SpotArmFK()
    spot_arm_ik: SpotArmIK = SpotArmIK()

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
        # only if ee vel commands are not zero
        if self.hand_vx != 0 or self.hand_vy != 0 or self.hand_vz != 0:
            # arm cartesian velocity command
            cartesian_velocity = arm_command_pb2.ArmVelocityCommand.CartesianVelocity()
            cartesian_velocity.frame_name = BODY_FRAME_NAME
            cartesian_velocity.velocity_in_frame_name.x = self.hand_vx
            cartesian_velocity.velocity_in_frame_name.y = self.hand_vy
            cartesian_velocity.velocity_in_frame_name.z = self.hand_vz

            # arm angular velocity command
            hand_angular_velocity = Vec3(x=0.0, y=0.0, z=0.0)

            arm_velocity_command = arm_command_pb2.ArmVelocityCommand.Request(
                cartesian_velocity=cartesian_velocity,
                angular_velocity_of_hand_rt_odom_in_hand=hand_angular_velocity,
            )

            robot_command = robot_command_pb2.RobotCommand()
            print(robot_command)
            robot_command.synchronized_command.arm_command.arm_velocity_command.CopyFrom(
                arm_velocity_command
            )

            # build command
            cmd = RobotCommandBuilder.synchro_velocity_command(
                v_x=self.vx,
                v_y=self.vy,
                v_rot=self.w,
                params=mobility_params,
                build_on_command=robot_command,
            )

        else:
            # build command
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
                self.hand_vx,
                self.hand_vy,
                self.hand_vz,
                self.hand_vrx,
                self.hand_vry,
                self.hand_vrz,
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
            hand_vx=arr[9],
            hand_vy=arr[10],
            hand_vz=arr[11],
            hand_vrx=arr[12],
            hand_vry=arr[13],
            hand_vrz=arr[14],
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
            hand_vx=d["hand_vx"],
            hand_vy=d["hand_vy"],
            hand_vz=d["hand_vz"],
            hand_vrx=d["hand_vrx"],
            hand_vry=d["hand_vry"],
            hand_vrz=d["hand_vrz"],
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
        s += "\thand_x: {:.5f}\n".format(self.hand_vx)
        s += "\thand_y: {:.5f}\n".format(self.hand_vy)
        s += "\thand_z: {:.5f}\n".format(self.hand_vz)
        s += "\thand_rx: {:.5f}\n".format(self.hand_vrx)
        s += "\thand_ry: {:.5f}\n".format(self.hand_vry)
        s += "\thand_rz: {:.5f}\n".format(self.hand_vrz)
        s += "}"
        s += "metadata {\n"
        s += "safe_check_infringed: {}\n".format(self.safety_check_infringed)
        s += "}"

        return s

    @staticmethod
    def size() -> int:
        return 15
