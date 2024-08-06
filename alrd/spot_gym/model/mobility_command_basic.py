from __future__ import annotations
from copy import copy
import numpy as np
import logging
from alrd.spot_gym.model.command import Command, CommandEnum, LocomotionHint
from alrd.spot_gym.model.robot_state import SpotState
from bosdyn.client.robot_command import RobotCommandBuilder
from bosdyn.client.frame_helpers import (
    BODY_FRAME_NAME,
    ODOM_FRAME_NAME,
    express_se3_velocity_in_new_frame,
)
from bosdyn.client.math_helpers import SE3Velocity, Vec3
from bosdyn.geometry import EulerZXY
from bosdyn.api import arm_command_pb2, robot_command_pb2
from alrd.spot_gym.utils.spot_arm_fk import SpotArmFK
from alrd.spot_gym.utils.spot_arm_ik import SpotArmIK
from dataclasses import asdict, dataclass

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)


@dataclass
class MobilityCommandBasic(Command):
    """
    This MobilityCommandBasic class provides mapping between SpotBasicEnv action and robot command for base velocity and end effector cartesian velocity commands.
    """

    # previous robot state
    prev_state: SpotState

    # command frequency
    cmd_freq: float

    # base actions
    vx: float
    vy: float
    vrz: float
    height: float
    pitch: float
    locomotion_hint: LocomotionHint
    stair_hint: bool

    # ee actions
    ee_vx: float
    ee_vy: float
    ee_vz: float

    # command type
    cmd_type = CommandEnum.MOBILITY

    # safety check infringed
    safety_check_infringed: bool = False

    def __post_init__(self) -> None:

        # make base command
        orientation = EulerZXY(roll=0.0, pitch=self.pitch, yaw=0.0)
        mobility_params = RobotCommandBuilder.mobility_params(
            body_height=self.height,
            footprint_R_body=orientation,
            locomotion_hint=self.locomotion_hint,
            stair_hint=self.stair_hint,
        )

        # make ee command
        # only if ee velocity commands are not zero
        if self.ee_vx != 0 or self.ee_vy != 0 or self.ee_vz != 0:
            # ee cartesian linear velocity command
            cartesian_velocity = arm_command_pb2.ArmVelocityCommand.CartesianVelocity()
            cartesian_velocity.frame_name = BODY_FRAME_NAME
            cartesian_velocity.velocity_in_frame_name.x = self.ee_vx
            cartesian_velocity.velocity_in_frame_name.y = self.ee_vy
            cartesian_velocity.velocity_in_frame_name.z = self.ee_vz

            arm_velocity_command = arm_command_pb2.ArmVelocityCommand.Request(
                cartesian_velocity=cartesian_velocity,
            )
            robot_command = robot_command_pb2.RobotCommand()
            robot_command.synchronized_command.arm_command.arm_velocity_command.CopyFrom(
                arm_velocity_command
            )

            # build command
            cmd = RobotCommandBuilder.synchro_velocity_command(
                v_x=self.vx,
                v_y=self.vy,
                v_rot=self.vrz,
                params=mobility_params,
                build_on_command=robot_command,
            )

        else:
            # build command
            cmd = RobotCommandBuilder.synchro_velocity_command(
                v_x=self.vx,
                v_y=self.vy,
                v_rot=self.vrz,
                params=mobility_params,
            )
        super().__init__(cmd)

    def __array__(self, dtype=None) -> np.ndarray:
        return np.array(
            [
                self.vx,
                self.vy,
                self.vrz,
                self.height,
                self.pitch,
                self.locomotion_hint,
                self.stair_hint,
                self.ee_vx,
                self.ee_vy,
                self.ee_vz,
            ],
            dtype=dtype,
        )

    @staticmethod
    def fromarray(arr: np.ndarray) -> MobilityCommandBasic:
        return MobilityCommandBasic(
            prev_state=arr[0],
            cmd_freq=arr[1],
            vx=arr[2],
            vy=arr[3],
            vrz=arr[4],
            height=arr[5],
            pitch=arr[6],
            locomotion_hint=arr[7],
            stair_hint=arr[8],
            ee_vx=arr[9],
            ee_vy=arr[10],
            ee_vz=arr[11],
        )

    def asdict(self) -> dict:
        return {**super().asdict(), **asdict(self)}

    @staticmethod
    def fromdict(d: dict) -> MobilityCommandBasic:
        return MobilityCommandBasic(
            prev_state=d["prev_state"],
            cmd_freq=d["cmd_freq"],
            vx=d["vx"],
            vy=d["vy"],
            vrz=d["vrz"],
            height=d["height"],
            pitch=d["pitch"],
            locomotion_hint=d["locomotion_hint"],
            stair_hint=d["stair_hint"],
            ee_vx=d["ee_vx"],
            ee_vy=d["ee_vy"],
            ee_vz=d["ee_vz"],
        )

    def to_str(self) -> str:
        s = "commands {\n"
        s += "\tx: {:.5f}\n".format(self.vx)
        s += "\ty: {:.5f}\n".format(self.vy)
        s += "\trot: {:.5f}\n".format(self.vrz)
        s += "\theight: {:.5f}\n".format(self.height)
        s += "\tpitch: {:.5f}\n".format(self.pitch)
        s += "\tlocomotion_hint: {}\n".format(self.locomotion_hint)
        s += "\tstair_hint: {}\n".format(self.stair_hint)
        s += "\tee_x: {:.5f}\n".format(self.ee_vx)
        s += "\tee_y: {:.5f}\n".format(self.ee_vy)
        s += "\tee_z: {:.5f}\n".format(self.ee_vz)
        s += "}"
        s += "metadata {\n"
        s += "safe_check_infringed: {}\n".format(self.safety_check_infringed)
        s += "}"

        return s

    @staticmethod
    def size() -> int:
        return 15
