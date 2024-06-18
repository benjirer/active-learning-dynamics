from __future__ import annotations
import numpy as np
from alrd.spot_gym_with_arm.model.command import Command, CommandEnum, LocomotionHint
from bosdyn.client.robot_command import RobotCommandBuilder
from bosdyn.geometry import EulerZXY
from bosdyn.api import arm_command_pb2


from dataclasses import asdict, dataclass


@dataclass
class MobilityCommand(Command):
    cmd_type = CommandEnum.MOBILITY

    # body command inputs
    vx: float
    vy: float
    w: float
    height: float
    pitch: float
    locomotion_hint: LocomotionHint
    stair_hint: bool

    # arm command inputs
    vr: float
    vaz: float
    vz: float

    def __post_init__(self) -> None:
        # body command
        orientation = EulerZXY(roll=0.0, pitch=self.pitch, yaw=0.0)
        mobility_params = RobotCommandBuilder.mobility_params(
            body_height=self.height,
            footprint_R_body=orientation,
            locomotion_hint=self.locomotion_hint,
            stair_hint=self.stair_hint,
        )

        # arm command
        cylindrical_velocity = arm_command_pb2.ArmVelocityCommand.CylindricalVelocity()
        cylindrical_velocity.linear_velocity.r = self.vr
        cylindrical_velocity.linear_velocity.theta = self.vaz
        cylindrical_velocity.linear_velocity.z = self.vz

        # might need to change this, not sure if correct
        # maybe "robot_command.synchronized_command.arm_command.arm_velocity_command.CopyFrom(arm_vel_command) is needed instead
        arm_vel_cmd = arm_command_pb2.ArmVelocityCommand.Request(
            cylindrical_velocity=cylindrical_velocity
        )
        arm_cmd = arm_command_pb2.ArmCommand.Request(arm_velocity_command=arm_vel_cmd)

        # build command
        cmd = RobotCommandBuilder.synchro_velocity_command(
            v_x=self.vx,
            v_y=self.vy,
            v_rot=self.w,
            params=mobility_params,
            build_on_command=arm_cmd,
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
        return MobilityCommand(*arr[:5], int(arr[5]), int(arr[6]), *arr[7:])

    def asdict(self) -> dict:
        return {**super().asdict(), **asdict(self)}

    @staticmethod
    def fromdict(d: dict) -> MobilityCommand:
        return MobilityCommand(
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
        s = "velocity_commands {\n"
        s += "\tx: {:.5f}\n".format(self.vx)
        s += "\ty: {:.5f}\n".format(self.vy)
        s += "\trot: {:.5f}\n".format(self.w)
        s += "\tarm_r: {:.5f}\n".format(self.vr)
        s += "\tarm_az: {:.5f}\n".format(self.vaz)
        s += "\tarm_z: {:.5f}\n".format(self.vz)
        s += "}"
        return s

    @staticmethod
    def size() -> int:
        return 10
