from __future__ import annotations
from copy import copy
import numpy as np
import logging
from alrd.spot_gym.model.command import Command, LocomotionHint
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
from dataclasses import asdict, dataclass

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)


@dataclass
class MobilityCommandBasic(Command):
    """
    This MobilityCommandBasic class provides mapping between SpotEnvBasic action and robot
    command for base velocity and end effector cartesian velocity commands.
    """

    # previous robot state
    prev_state: SpotState

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
            vx=arr[1],
            vy=arr[2],
            vrz=arr[3],
            height=arr[4],
            pitch=arr[5],
            locomotion_hint=arr[6],
            stair_hint=arr[7],
            ee_vx=arr[8],
            ee_vy=arr[9],
            ee_vz=arr[10],
        )

    def asdict(self) -> dict:
        return {**super().asdict(), **asdict(self)}

    @staticmethod
    def fromdict(d: dict) -> MobilityCommandBasic:
        return MobilityCommandBasic(
            prev_state=d["prev_state"],
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

        return s

    @staticmethod
    def size() -> int:
        return 11


@dataclass
class MobilityCommandAugmented(MobilityCommandBasic):
    """
    MobilityCommandAugmented wraps MobilityCommandBasic and adds the ability to command
    end-effector angular velocities (ee_vrx, ee_vry, ee_vrz).
    """

    # ee angular velocities
    ee_vrx: float
    ee_vry: float
    ee_vrz: float

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
        if (
            self.ee_vx != 0
            or self.ee_vy != 0
            or self.ee_vz != 0
            or self.ee_vrx != 0
            or self.ee_vry != 0
            or self.ee_vrz != 0
        ):
            # ee cartesian linear velocity command
            cartesian_velocity = arm_command_pb2.ArmVelocityCommand.CartesianVelocity()
            cartesian_velocity.frame_name = BODY_FRAME_NAME
            cartesian_velocity.velocity_in_frame_name.x = self.ee_vx
            cartesian_velocity.velocity_in_frame_name.y = self.ee_vy
            cartesian_velocity.velocity_in_frame_name.z = self.ee_vz

            # ee angular velocity command
            # note: we have to convert ee angular velocity from body frame to odom frame
            # since the SDK only accepts ee angular velocity in odom frame
            ee_vel_in_body = SE3Velocity(
                lin_x=self.ee_vx,
                lin_y=self.ee_vy,
                lin_z=self.ee_vz,
                ang_x=self.ee_vrx,
                ang_y=self.ee_vry,
                ang_z=self.ee_vrz,
            )
            ee_vel_in_odom_proto = express_se3_velocity_in_new_frame(
                self.prev_state.transforms_snapshot,
                BODY_FRAME_NAME,
                ODOM_FRAME_NAME,
                ee_vel_in_body.to_proto(),
            )
            ee_angular_velocity = Vec3(
                x=ee_vel_in_odom_proto.angular_velocity_x,
                y=ee_vel_in_odom_proto.angular_velocity_y,
                z=ee_vel_in_odom_proto.angular_velocity_z,
            )

            arm_velocity_command = arm_command_pb2.ArmVelocityCommand.Request(
                cartesian_velocity=cartesian_velocity,
                angular_velocity_of_hand_rt_odom_in_hand=ee_angular_velocity.to_proto(),
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
            super().__post_init__()
            return

        super(Command, self).__init__(cmd)

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
                self.ee_vrx,
                self.ee_vry,
                self.ee_vrz,
            ],
            dtype=dtype,
        )

    @staticmethod
    def fromarray(arr: np.ndarray) -> MobilityCommandAugmented:
        return MobilityCommandAugmented(
            prev_state=arr[0],
            vx=arr[1],
            vy=arr[2],
            vrz=arr[3],
            height=arr[4],
            pitch=arr[5],
            locomotion_hint=arr[6],
            stair_hint=arr[7],
            ee_vx=arr[8],
            ee_vy=arr[9],
            ee_vz=arr[10],
            ee_vrx=arr[11],
            ee_vry=arr[12],
            ee_vrz=arr[13],
        )

    def asdict(self) -> dict:
        return {**super().asdict(), **asdict(self)}

    @staticmethod
    def fromdict(d: dict) -> MobilityCommandAugmented:
        return MobilityCommandAugmented(
            prev_state=d["prev_state"],
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
            ee_vrx=d["ee_vrx"],
            ee_vry=d["ee_vry"],
            ee_vrz=d["ee_vrz"],
        )

    def to_str(self) -> str:
        s = "commands {\n"
        s += "\tvx: {:.5f}\n".format(self.vx)
        s += "\tvy: {:.5f}\n".format(self.vy)
        s += "\tvrz: {:.5f}\n".format(self.vrz)
        s += "\theight: {:.5f}\n".format(self.height)
        s += "\tpitch: {:.5f}\n".format(self.pitch)
        s += "\tlocomotion_hint: {}\n".format(self.locomotion_hint)
        s += "\tstair_hint: {}\n".format(self.stair_hint)
        s += "\tee_vx: {:.5f}\n".format(self.ee_vx)
        s += "\tee_vy: {:.5f}\n".format(self.ee_vy)
        s += "\tee_vz: {:.5f}\n".format(self.ee_vz)
        s += "\tee_vrx: {:.5f}\n".format(self.ee_vrx)
        s += "\tee_vry: {:.5f}\n".format(self.ee_vry)
        s += "\tee_vrz: {:.5f}\n".format(self.ee_vrz)
        s += "}"
        return s

    @staticmethod
    def size() -> int:
        return 14
