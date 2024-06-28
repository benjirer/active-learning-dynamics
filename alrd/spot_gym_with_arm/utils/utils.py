from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import textwrap
from enum import Enum
from typing import Tuple

# body limits
MAX_SPEED = 0.8  # Maximum linear velocity of the robot (m/s)
MAX_ANGULAR_SPEED = 1.5  # Maximum angular velocity of the robot (rad/s)
DIST_TO_FRONT = 0.55  # Distance from body frame origin to front of the robot (m)
SPOT_LENGTH = 1.1  # Length of the robot from rear to front (m)
SPOT_WIDTH = 0.5  # Width of the robot from left to right side (m)

# arm limits
MIN_HEIGHT = 0.0
MAX_HEIGHT = 1.8
MIN_AZIMUTHAL = -(150 / 180) * np.pi
MAX_AZIMUTHAL = np.pi
MIN_RADIAL_POS = -1.0
MAX_RADIAL_POS = 1.0
ARM_MAX_LINEAR_VELOCITY = 0.5
MAX_RADIAL_VEL = ARM_MAX_LINEAR_VELOCITY
MAX_VERTICAL_VEL = ARM_MAX_LINEAR_VELOCITY
MAX_AZIMUTHAL_VEL = np.pi / 4

# arm joint position limits (rad)
# from https://d3cjkvgbik1jtv.cloudfront.net/Spot+IFU/spot_arm_information_for_use_EN_v1.0.pdf
SH0_POS_MIN = -150.0 * (np.pi / 180.0)
SH0_POS_MAX = 180.0 * (np.pi / 180.0)
SH1_POS_MIN = -180.0 * (np.pi / 180.0)
SH1_POS_MAX = 30.0 * (np.pi / 180.0)
EL0_POS_MIN = 0.0
EL0_POS_MAX = 180.0 * (np.pi / 180.0)
EL1_POS_MIN = -160.0 * (np.pi / 180.0)
EL1_POS_MAX = 160.0 * (np.pi / 180.0)
WR0_POS_MIN = -105.0 * (np.pi / 180.0)
WR0_POS_MAX = 105.0 * (np.pi / 180.0)
WR1_POS_MIN = -165.0 * (np.pi / 180.0)
WR1_POS_MAX = 165.0 * (np.pi / 180.0)

# arm joint velocity limits (rad/s)
MAX_JOINT_VEL = 10.0


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
