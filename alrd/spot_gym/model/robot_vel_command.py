from __future__ import annotations
from alrd.spot_gym.model.command import Command, CommandEnum
from alrd.spot_gym.model.mobility_command import MobilityCommand
from alrd.spot_gym.model.arm_command import ArmCylindricalVelocityCommand
from bosdyn.client.robot_command import RobotCommandBuilder
import numpy as np
from dataclasses import dataclass


@dataclass
class RobotVelCommand(Command):
    cmd_type = CommandEnum.ROBOT_VEL
    mobility_command: MobilityCommand
    arm_command: ArmCylindricalVelocityCommand

    def __post_init__(self) -> None:
        cmd = RobotCommandBuilder.build_synchro_command(
            self.mobility_command.cmd, self.arm_command.cmd
        )
        super().__init__(cmd)

    def __array__(self, dtype=None) -> np.ndarray:
        return np.concatenate(
            [
                np.array(self.mobility_command, dtype=dtype),
                np.array(self.arm_command, dtype=dtype),
            ]
        )

    @staticmethod
    def fromarray(arr: np.ndarray) -> RobotVelCommand:
        mobility_command = MobilityCommand.fromarray(arr[: MobilityCommand.size()])
        arm_command = ArmCylindricalVelocityCommand.fromarray(
            arr[MobilityCommand.size() :]
        )
        return RobotVelCommand(mobility_command, arm_command)
