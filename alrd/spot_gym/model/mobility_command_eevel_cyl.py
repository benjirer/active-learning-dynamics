from __future__ import annotations
from copy import copy
import numpy as np
import logging
from alrd.spot_gym.model.command import Command, CommandEnum, LocomotionHint
from alrd.spot_gym.model.robot_state import SpotState
from bosdyn.client.robot_command import RobotCommandBuilder
from bosdyn.geometry import EulerZXY
from bosdyn.api import arm_command_pb2, robot_command_pb2
from alrd.spot_gym.utils.utils import (
    ARM_MAX_JOINT_VEL,
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
    This MobilityCommand class provides mapping between SpotEEVelEnv action and robot command for body velocity and end effector cylindrical velocity commands.
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

    # arm commands
    vr: float
    vaz: float
    vz: float

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
        if self.vr != 0 or self.vaz != 0 or self.vz != 0:
            # arm command
            cylindrical_velocity = (
                arm_command_pb2.ArmVelocityCommand.CylindricalVelocity()
            )
            cylindrical_velocity.linear_velocity.r = self.vr
            cylindrical_velocity.linear_velocity.theta = self.vaz
            cylindrical_velocity.linear_velocity.z = self.vz

            arm_velocity_command = arm_command_pb2.ArmVelocityCommand.Request(
                cylindrical_velocity=cylindrical_velocity
            )

            robot_command = robot_command_pb2.RobotCommand()
            print(robot_command)
            robot_command.synchronized_command.arm_command.arm_velocity_command.CopyFrom(
                arm_velocity_command
            )

            # build command
            # TODO: add arm cmd as build on command
            cmd = RobotCommandBuilder.synchro_velocity_command(
                v_x=self.vx,
                v_y=self.vy,
                v_rot=self.w,
                params=mobility_params,
                build_on_command=robot_command,
            )

        else:
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
                self.vr,
                self.vaz,
                self.vz,
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
            vr=arr[9],
            vaz=arr[10],
            vz=arr[11],
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
            vr=d["vr"],
            vaz=d["vaz"],
            vz=d["vz"],
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
        s += "\tarm_r: {:.5f}\n".format(self.vr)
        s += "\tarm_az: {:.5f}\n".format(self.vaz)
        s += "\tarm_z: {:.5f}\n".format(self.vz)
        s += "}"
        s += "metadata {\n"
        s += "safe_check_infringed: {}\n".format(self.safety_check_infringed)
        s += "}"

        return s

    @staticmethod
    def size() -> int:
        return 12
