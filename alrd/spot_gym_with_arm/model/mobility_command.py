from __future__ import annotations
import numpy as np
import csv
from alrd.spot_gym_with_arm.model.command import Command, CommandEnum, LocomotionHint
from alrd.spot_gym_with_arm.model.robot_state import SpotState
from bosdyn.client.robot_command import RobotCommandBuilder
from bosdyn.geometry import EulerZXY
from bosdyn.api import arm_command_pb2, robot_command_pb2
from alrd.spot_gym_with_arm.utils.utils import MAX_JOINT_VEL
from dataclasses import asdict, dataclass


@dataclass
class MobilityCommand(Command):
    cmd_type = CommandEnum.MOBILITY

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
    sh0_vel: float
    sh1_vel: float
    el0_vel: float
    el1_vel: float
    wr0_vel: float
    wr1_vel: float

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
        # only if velocity commands are not zero and previous state is not None
        if (
            self.sh0_vel != 0.0
            or self.sh1_vel != 0.0
            or self.el0_vel != 0.0
            or self.el1_vel != 0.0
            or self.wr0_vel != 0.0
            or self.wr1_vel != 0.0
            and self.prev_state is not None
        ):
            # as the API only allows for arm joint position commands, we need to integrate the velocity commands
            # we take the prior state q and integrate the velocity commands to get the new state q_new
            joint_positions_prev = self.prev_state.arm_joint_positions

            # # log in csv: prev state, new state, velocity commands
            # myCsvRow = (
            #     str(self.prev_state.arm_joint_positions[0])
            #     + ","
            #     + str(self.prev_state.arm_joint_positions[1])
            #     + ","
            #     + str(self.prev_state.arm_joint_positions[2])
            #     + ","
            #     + str(self.prev_state.arm_joint_positions[3])
            #     + ","
            #     + str(self.prev_state.arm_joint_positions[4])
            #     + ","
            #     + str(self.prev_state.arm_joint_positions[5])
            #     + ","
            #     + str(joint_positions_new[0])
            #     + ","
            #     + str(joint_positions_new[1])
            #     + ","
            #     + str(joint_positions_new[2])
            #     + ","
            #     + str(joint_positions_new[3])
            #     + ","
            #     + str(joint_positions_new[4])
            #     + ","
            #     + str(joint_positions_new[5])
            #     + ","
            #     + str(self.sh0_vel / self.cmd_freq)
            #     + ","
            #     + str(self.sh1_vel / self.cmd_freq)
            #     + ","
            #     + str(self.el0_vel / self.cmd_freq)
            #     + ","
            #     + str(self.el1_vel / self.cmd_freq)
            #     + ","
            #     + str(self.wr0_vel / self.cmd_freq)
            #     + ","
            #     + str(self.wr1_vel / self.cmd_freq)
            #     + "\n"
            # )

            # with open("logger.csv", "a") as fd:
            #     fd.write(myCsvRow)

            arm_cmd = RobotCommandBuilder.arm_joint_command(
                sh0=joint_positions_prev[0]
                + self.sh0_vel * MAX_JOINT_VEL / self.cmd_freq,
                sh1=joint_positions_prev[1] + self.sh1_vel / self.cmd_freq,
                el0=joint_positions_prev[2] + self.el0_vel / self.cmd_freq,
                el1=joint_positions_prev[3] + self.el1_vel / self.cmd_freq,
                wr0=joint_positions_prev[4] + self.wr0_vel / self.cmd_freq,
                wr1=joint_positions_prev[5] + self.wr1_vel / self.cmd_freq,
                max_vel=MAX_JOINT_VEL,
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
                self.sh0_vel,
                self.sh1_vel,
                self.el0_vel,
                self.el1_vel,
                self.wr0_vel,
                self.wr1_vel,
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
            sh0_vel=arr[9],
            sh1_vel=arr[10],
            el0_vel=arr[11],
            el1_vel=arr[12],
            wr0_vel=arr[13],
            wr1_vel=arr[14],
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
            sh0_vel=d["sh0_vel"],
            sh1_vel=d["sh1_vel"],
            el0_vel=d["el0_vel"],
            el1_vel=d["el1_vel"],
            wr0_vel=d["wr0_vel"],
            wr1_vel=d["wr1_vel"],
        )

    def to_str(self) -> str:
        s = "velocity_commands {\n"
        s += "\tx: {:.5f}\n".format(self.vx)
        s += "\ty: {:.5f}\n".format(self.vy)
        s += "\trot: {:.5f}\n".format(self.w)
        s += "\theight: {:.5f}\n".format(self.height)
        s += "\tpitch: {:.5f}\n".format(self.pitch)
        s += "\tlocomotion_hint: {}\n".format(self.locomotion_hint)
        s += "\tstair_hint: {}\n".format(self.stair_hint)
        s += "\tsh0_vel: {:.5f}\n".format(self.sh0_vel)
        s += "\tsh1_vel: {:.5f}\n".format(self.sh1_vel)
        s += "\tel0_vel: {:.5f}\n".format(self.el0_vel)
        s += "\tel1_vel: {:.5f}\n".format(self.el1_vel)
        s += "\twr0_vel: {:.5f}\n".format(self.wr0_vel)
        s += "\twr1_vel: {:.5f}\n".format(self.wr1_vel)
        s += "}"
        return s

    @staticmethod
    def size() -> int:
        return 15
