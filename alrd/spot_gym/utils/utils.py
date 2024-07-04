from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import textwrap
from enum import Enum
from typing import Tuple

# robot constants and limits

# body constants
DIST_TO_FRONT = 0.55  # Distance from body frame origin to front of the robot (m)
SPOT_LENGTH = 1.1  # Length of the robot from rear to front (m)
SPOT_WIDTH = 0.5  # Width of the robot from left to right side (m)

# arm constants
# from https://d3cjkvgbik1jtv.cloudfront.net/Spot+IFU/spot_arm_information_for_use_EN_v1.0.pdf
# TODO: add real lengths
ARM_FULL_LENGTH = 0.985  # Full length of the arm (m)
ARM_LINK_0_LENGTH = 0.1  # Length of the base - sh0 link of the arm (m)
ARM_LINK_1_LENGTH = 0.1  # Length of the sh0 - sh1 link of the arm (m)
ARM_LINK_2_LENGTH = 0.3385  # Length of the sh1 - el0 link of the arm (m)
ARM_LINK_3_LENGTH = 0.2  # Length of the el0 - el1 link of the arm (m)
ARM_LINK_4_LENGTH = 0.3  # Length of the el1 - wr0 link of the arm (m)
ARM_LINK_5_LENGTH = 0.1  # Length of the wr0 - wr1 link of the arm (m)

# body limits
MAX_SPEED = 0.8  # Maximum linear velocity of the robot (m/s)
MAX_ANGULAR_SPEED = 1.5  # Maximum angular velocity of the robot (rad/s)

# arm limits
MIN_HEIGHT = 0.0  # Minimum height of the arm (m)
MAX_HEIGHT = 1.8  # Maximum height of the arm (m)
MIN_AZIMUTHAL = -(150 / 180) * np.pi  # Minimum azimuthal angle of the arm (rad)
MAX_AZIMUTHAL = np.pi  # Maximum azimuthal angle of the arm (rad)
MIN_RADIAL_POS = -1.0  # Minimum radial position of the arm (m)
MAX_RADIAL_POS = 1.0  # Maximum radial position of the arm (m)
ARM_MAX_LINEAR_VELOCITY = 0.5  # Maximum linear velocity of the arm (m/s)
MAX_RADIAL_VEL = ARM_MAX_LINEAR_VELOCITY  # Maximum radial velocity of the arm (m/s)
MAX_VERTICAL_VEL = ARM_MAX_LINEAR_VELOCITY  # Maximum vertical velocity of the arm (m/s)
MAX_AZIMUTHAL_VEL = np.pi / 4  # Maximum azimuthal velocity of the arm (rad/s)

# arm joint position limits (rad)
# from https://d3cjkvgbik1jtv.cloudfront.net/Spot+IFU/spot_arm_information_for_use_EN_v1.0.pdf
SH0_POS_MIN = -150.0 * (np.pi / 180.0)  # Minimum position of shoulder joint 0 (rad)
SH0_POS_MAX = 180.0 * (np.pi / 180.0)  # Maximum position of shoulder joint 0 (rad)
SH1_POS_MIN = -180.0 * (np.pi / 180.0)  # Minimum position of shoulder joint 1 (rad)
SH1_POS_MAX = 30.0 * (np.pi / 180.0)  # Maximum position of shoulder joint 1 (rad)
EL0_POS_MIN = 0.0 * (np.pi / 180.0)  # Minimum position of elbow joint 0 (rad)
EL0_POS_MAX = 180.0 * (np.pi / 180.0)  # Maximum position of elbow joint 0 (rad)
EL1_POS_MIN = -160.0 * (np.pi / 180.0)  # Minimum position of elbow joint 1 (rad)
EL1_POS_MAX = 160.0 * (np.pi / 180.0)  # Maximum position of elbow joint 1 (rad)
WR0_POS_MIN = -105.0 * (np.pi / 180.0)  # Minimum position of wrist joint 0 (rad)
WR0_POS_MAX = 105.0 * (np.pi / 180.0)  # Maximum position of wrist joint 0 (rad)
WR1_POS_MIN = -165.0 * (np.pi / 180.0)  # Minimum position of wrist joint 1 (rad)
WR1_POS_MAX = 165.0 * (np.pi / 180.0)  # Maximum position of wrist joint 1 (rad)

# arm joint velocity limits (rad/s)
MAX_ARM_JOINT_VEL = 10.0


def get_front_coord(x: float, y: float, cos: float, sin: float) -> Tuple[float, float]:
    return x + cos * DIST_TO_FRONT, y + sin * DIST_TO_FRONT


def get_hitbox(x: float, y: float, angle: float) -> np.ndarray:
    """
    Returns the coordinates of the hitbox of the robot in the given frame.

    Returns:
        np.ndarray: 4x2 array of coordinates of the hitbox (front left, front right, back right, back left)
    """
    cos, sin = np.cos(angle), np.sin(angle)
    v_front = np.array(get_front_coord(0, 0, cos, sin))
    v_back = np.array(
        [-cos * (SPOT_LENGTH - DIST_TO_FRONT), -sin * (SPOT_LENGTH - DIST_TO_FRONT)]
    )
    v_left = np.array([-sin * SPOT_WIDTH / 2, cos * SPOT_WIDTH / 2])
    return np.array([x, y]) + np.array(
        [v_front + v_left, v_front - v_left, v_back - v_left, v_back + v_left]
    )


class Sensor(Enum):
    VISION = 0
    ODOM = 1


class Frame(Enum):
    VISION = 0
    ODOM = 1
    BODY = 2


@dataclass
class Vector3D:
    x: float
    y: float
    z: float

    def __array__(self, dtype=None) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=dtype)

    def fromarray(arr: np.ndarray) -> Vector3D:
        return Vector3D(*arr)


@dataclass
class Quaternion:
    w: float
    x: float
    y: float
    z: float

    def __array__(self, dtype=None) -> np.ndarray:
        return np.array([self.w, self.x, self.y, self.z], dtype=dtype)

    def fromarray(arr: np.ndarray) -> Quaternion:
        return Quaternion(*arr)


@dataclass
class SE3Velocity:
    linear: Vector3D
    angular: Vector3D

    def __array__(self, dtype=None) -> np.ndarray:
        return np.concatenate(
            (np.array(self.linear, dtype=dtype), np.array(self.angular, dtype=dtype)),
            dtype=dtype,
        )

    def fromarray(arr: np.ndarray) -> SE3Velocity:
        return SE3Velocity(Vector3D.fromarray(arr[:3]), Vector3D.fromarray(arr[3:]))

    def __str__(self) -> str:
        return textwrap.dedent(
            f"""\
            linear {{
            \tx: {self.linear.x}
            \ty: {self.linear.y}
            \tz: {self.linear.z}
            }}
            angular {{
            \tx: {self.angular.x}
            \ty: {self.angular.y}
            \tz: {self.angular.z}
            }}"""
        )


@dataclass
class CylindricalVelocity:
    r: float  # radial velocity
    theta: float  # azimuthal velocity
    z: float  # vertical velocity

    def __array__(self, dtype=None) -> np.ndarray:
        return np.array([self.r, self.theta, self.z], dtype=dtype)

    def fromarray(arr: np.ndarray) -> CylindricalVelocity:
        return CylindricalVelocity(*arr)


@dataclass
class SE3Pose:
    position: Vector3D
    rotation: Quaternion

    def __array__(self, dtype=None) -> np.ndarray:
        return np.concatenate(
            (
                np.array(self.position, dtype=dtype),
                np.array(self.rotation, dtype=dtype),
            ),
            dtype=dtype,
        )

    def fromarray(arr: np.ndarray) -> SE3Pose:
        return SE3Pose(Vector3D.fromarray(arr[:3]), Quaternion.fromarray(arr[3:]))

    def __str__(self) -> str:
        return textwrap.dedent(
            f"""\
            position {{
            \tx: {self.position.x}
            \ty: {self.position.y}
            \tz: {self.position.z}
            }}
            rotation {{
            \tw: {self.rotation.w}
            \tx: {self.rotation.x}
            \ty: {self.rotation.y}
            \tz: {self.rotation.z}
            }}"""
        )
