import numpy as np
from alrd.spot_gym.utils.utils import (
    ARM_LINK_0_LENGTH,
    ARM_LINK_1_LENGTH,
    ARM_LINK_2_LENGTH,
    ARM_LINK_3_LENGTH,
    ARM_LINK_4_LENGTH,
    ARM_LINK_5_LENGTH,
)


def transformation_matrix_z(angle, length):
    """
    Create a transformation matrix for a rotation around the z-axis by `angle` radians
    and a translation along the x-axis by `length`.
    """
    return np.array(
        [
            [np.cos(angle), -np.sin(angle), 0, length],
            [np.sin(angle), np.cos(angle), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )


def transformation_matrix_y(angle, length):
    """
    Create a transformation matrix for a rotation around the y-axis by `angle` radians
    and a translation along the x-axis by `length`.
    """
    return np.array(
        [
            [np.cos(angle), 0, np.sin(angle), length],
            [0, 1, 0, 0],
            [-np.sin(angle), 0, np.cos(angle), 0],
            [0, 0, 0, 1],
        ]
    )


def forward_kinematics(sh0, sh1, el0, el1, wr0, wr1):
    """
    Compute the position of the end effector given joint angles with named arguments.

    Args:
        sh0, sh1, el0, el1, wr0, wr1: Angles (in radians) for each corresponding joint

    Returns:
        x,y,z: end effector's position
    """
    # link lengths
    lengths = [
        ARM_LINK_0_LENGTH,
        ARM_LINK_1_LENGTH,
        ARM_LINK_2_LENGTH,
        ARM_LINK_3_LENGTH,
        ARM_LINK_4_LENGTH,
        ARM_LINK_5_LENGTH,
    ]

    # transforms
    T = np.eye(4)
    T = T @ transformation_matrix_z(sh0, lengths[0])
    T = T @ transformation_matrix_y(sh1, lengths[1])
    T = T @ transformation_matrix_y(el0, lengths[2])
    T = T @ transformation_matrix_z(el1, lengths[3])
    T = T @ transformation_matrix_y(wr0, lengths[4])
    T = T @ transformation_matrix_z(wr1, lengths[5])

    # end effector position
    x, y, z = T[0, 3], T[1, 3], T[2, 3]

    return x, y, z


if __name__ == "__main__":
    joint_states = [
        np.pi / 4,  # sh0
        np.pi / 4,  # sh1
        np.pi / 4,  # el0
        np.pi / 4,  # el1
        np.pi / 4,  # wr0
        np.pi / 4,  # wr1
    ]
    end_effector_position = forward_kinematics(*joint_states)
    # TODO: plot in 3D along with the robot arm
