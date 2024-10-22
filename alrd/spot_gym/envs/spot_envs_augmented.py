from __future__ import annotations
import textwrap
from pathlib import Path
from typing import Any, Optional, Tuple, Callable
from jax import jit
import jax
import jax.numpy as jnp
import numpy as np
from gym import spaces
from scipy.spatial.transform import Rotation as R
from flax import struct

from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from opax.models.reward_model import RewardModel
from jdm_control.rewards import get_tolerance_fn

from alrd.spot_gym.model.command import Command
from alrd.spot_gym.model.mobility_command import (
    MobilityCommandBasic,
    MobilityCommandAugmented,
)
from alrd.spot_gym.model.robot_state import SpotState
from alrd.spot_gym.envs.spot_env_base import SpotEnvBase
from alrd.spot_gym.model.spot import SpotEnvironmentConfig
from alrd.utils.utils import change_frame_2d, rotate_2d_vector, Frame2D
from alrd.agent.keyboard import KeyboardResetAgent, KeyboardAgent
from alrd.spot_gym.utils.utils import (
    MIN_X,
    MAX_X,
    MIN_Y,
    MAX_Y,
    BODY_MAX_ANGULAR_VEL,
    BODY_MAX_VEL,
    ARM_MIN_HEIGHT,
    ARM_MAX_HEIGHT,
    ARM_MAX_X,
    ARM_MAX_Y,
    ARM_MAX_LINEAR_VEL,
    ARM_MIN_AZIMUTHAL,
    ARM_MAX_AZIMUTHAL,
    ARM_MAX_RADIAL,
    ARM_MAX_RADIAL_VEL,
    ARM_MAX_VERTICAL_VEL,
    ARM_MAX_AZIMUTHAL_VEL,
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
    ARM_MAX_JOINT_VEL,
)


def norm(x: jax.Array, axis: int):
    norm = jnp.sum(x * x, axis=axis)
    return jnp.sqrt(norm + 1e-12)


# reward options
_DEFAULT_VALUE_AT_MARGIN = 0.1


# Reward functions
@struct.dataclass
class DistReward(RewardModel):
    tolerance_fn: Callable = struct.field(pytree_node=False)
    goal_pos: jax.Array
    margin: float

    @staticmethod
    def create(
        goal_pos: jax.Array,
        margin: float,
        sigmoid: str = "long_tail",
        bounds=None,
        value_at_margin=_DEFAULT_VALUE_AT_MARGIN,
    ):
        if bounds is None:
            bounds = (0.0, 0.0)
        bounds = np.array(bounds)
        assert goal_pos.shape == (3,)
        tolerance_fn = get_tolerance_fn(
            bounds=bounds / margin,
            margin=1.0,
            sigmoid=sigmoid,
            value_at_margin=value_at_margin,
        )
        return DistReward(tolerance_fn, goal_pos, margin)

    @jit
    def predict(self, obs, action, next_obs=None, rng=None):
        # dist = jnp.linalg.norm(obs[..., 0:2] - self.goal_pos, axis=-1)
        dist = norm(obs[..., 7:10] - self.goal_pos, -1)
        return self.tolerance_fn(dist / self.margin)


@struct.dataclass
class ActionCostSimple(RewardModel):
    action_cost_fn: Callable = struct.field(pytree_node=False)
    """Just takes the squared sum of the action vector"""

    @staticmethod
    def create(sigmoid: str = "long_tail"):
        action_cost_fn = lambda action: jnp.sum(action**2, axis=-1)
        return ActionCostSimple(action_cost_fn)

    @jit
    def predict(self, obs, action, next_obs=None, rng=None):
        act_norm = norm(action[..., :3], axis=-1)
        return 1 - act_norm / jnp.linalg.norm(jnp.ones([3]))


@struct.dataclass
class Spot2DReward(RewardModel):
    dist_rew: DistReward
    act_cost: ActionCostSimple
    action_coeff: float = struct.field(default=0.0)

    @staticmethod
    def create(
        goal_pos: np.ndarray | jnp.ndarray = None,
        action_coeff: float = 0.001,
        dist_margin: float = 5.0,
        sigmoid: str = "long_tail",
        dist_bound: float = 0.0,
    ):

        if goal_pos is None:
            goal_pos = jnp.zeros([3])

        dist_rew = DistReward.create(
            goal_pos, dist_margin, sigmoid=sigmoid, bounds=(0.0, dist_bound)
        )

        act_cost = ActionCostSimple.create(sigmoid=sigmoid)

        return Spot2DReward(
            dist_rew,
            act_cost,
            action_coeff,
        )

    @jit
    def predict(self, obs, action, next_obs=None, rng=None, last_obs=None):
        distance_reward = self.dist_rew.predict(obs, action, next_obs, rng)
        action_cost = self.act_cost.predict(obs, action, next_obs, rng)
        if last_obs is not None:
            last_action = jnp.concatenate(
                [last_obs[..., 4:7], last_obs[..., 10:13]], axis=-1
            )
            action_difference_cost = -jnp.sum((action - last_action) ** 2, axis=-1)
        else:
            action_difference_cost = 0.0

        ee_body_reward = 0.1  # always given on real robot

        return (
            distance_reward
            + self.action_coeff * action_cost
            + 0.01 * action_difference_cost
            + ee_body_reward
        )


class SpotEnvBasic(SpotEnvBase):
    """
    SpotEnvBasic allows for control of the robot base and end-effector.

    Base State Observation:
        x: x position of the base in global frame
        y: y position of the base in global frame
        sin(theta): sine of the heading angle theta of the base in global frame
        cos(theta): cosine of the heading angle theta of the base global frame
        vx: x velocity of the base in global frame
        vy: y velocity of the base in global frame
        vrz: yaw angular velocity of the base in global frame

    EE State Observation:
        x: position of ee in global frame
        y: position of ee in global frame
        z: position of ee in global frame
        vx: x velocity of ee in global frame
        vy: y velocity of ee in global frame
        vz: z velocity of ee in global frame

    Base Action:
        vx: x linear velocity command for robot base
        vy: y linear velocity command for robot base
        vrz: yaw angular velocity command for robot base

    EE Action:
        vx: x linear velocity command for ee in body frame
        vy: y linear velocity command for ee in body frame
        vz: z linear velocity command for ee in body frame
    """

    obs_shape = (13,)
    action_shape = (6,)

    def __init__(
        self,
        config: SpotEnvironmentConfig,
        cmd_freq: float,
        goal_pos: np.ndarray = None,
        monitor_freq: float = 30,
        action_cost=0.0,
        skip_ui: bool = False,
    ):
        super().__init__(
            config,
            cmd_freq,
            monitor_freq,
        )

        # command frequency
        self._cmd_freq = cmd_freq

        # observation space limits
        self.observation_space = spaces.Box(
            low=np.array(
                [
                    MIN_X,
                    MIN_Y,
                    -1,
                    -1,
                    -BODY_MAX_VEL,
                    -BODY_MAX_VEL,
                    -BODY_MAX_ANGULAR_VEL,
                    -ARM_MAX_X,
                    -ARM_MAX_Y,
                    ARM_MIN_HEIGHT,
                    -ARM_MAX_LINEAR_VEL,
                    -ARM_MAX_LINEAR_VEL,
                    -ARM_MAX_VERTICAL_VEL,
                ]
            ),
            high=np.array(
                [
                    MAX_X,
                    MAX_Y,
                    1,
                    1,
                    BODY_MAX_VEL,
                    BODY_MAX_VEL,
                    BODY_MAX_ANGULAR_VEL,
                    ARM_MAX_X,
                    ARM_MAX_Y,
                    ARM_MAX_HEIGHT,
                    ARM_MAX_LINEAR_VEL,
                    ARM_MAX_LINEAR_VEL,
                    ARM_MAX_VERTICAL_VEL,
                ]
            ),
        )

        # action space limits
        self.action_space = spaces.Box(
            low=np.array(
                [
                    -BODY_MAX_VEL,
                    -BODY_MAX_VEL,
                    -BODY_MAX_ANGULAR_VEL,
                    -ARM_MAX_LINEAR_VEL,
                    -ARM_MAX_LINEAR_VEL,
                    -ARM_MAX_VERTICAL_VEL,
                ]
            ),
            high=np.array(
                [
                    BODY_MAX_VEL,
                    BODY_MAX_VEL,
                    BODY_MAX_ANGULAR_VEL,
                    ARM_MAX_LINEAR_VEL,
                    ARM_MAX_LINEAR_VEL,
                    ARM_MAX_VERTICAL_VEL,
                ]
            ),
        )

        # reward model
        self.reward = Spot2DReward.create(action_coeff=action_cost, goal_pos=goal_pos)
        self.__keyboard = KeyboardResetAgent(KeyboardAgent(0.5, 0.5))
        self.__skip_ui = skip_ui

    def start(self):
        super().start()

    def get_obs_from_state(self, state: SpotState) -> np.ndarray:
        # base state observations
        x, y, z, qx, qy, qz, qw = state.pose_of_body_in_vision
        theta = R.from_quat([qx, qy, qz, qw]).as_euler("xyz", degrees=False)[2]
        theta = (theta + np.pi) % (2 * np.pi) - np.pi
        vx, vy, vz, vrx, vry, vrz = state.velocity_of_body_in_vision

        # ee state observations
        ee_x, ee_y, ee_z, ee_qx, ee_qy, ee_qz, ee_qw = state.pose_of_hand_in_vision
        ee_rx, ee_ry, ee_rz = R.from_quat([ee_qx, ee_qy, ee_qz, ee_qw]).as_euler(
            "xyz", degrees=False
        )
        ee_vx, ee_vy, ee_vz, ee_vrx, ee_vry, ee_vrz = state.velocity_of_hand_in_vision

        return np.array(
            [
                x,
                y,
                np.sin(theta),
                np.cos(theta),
                vx,
                vy,
                vrz,
                ee_x,
                ee_y,
                ee_z,
                ee_vx,
                ee_vy,
                ee_vz,
            ]
        )

    def get_cmd_from_action(
        self, action: np.ndarray, prev_state: np.ndarray
    ) -> Command:
        return MobilityCommandBasic(
            prev_state=prev_state,
            cmd_freq=self._cmd_freq,
            vx=action[0],
            vy=action[1],
            vrz=action[2],
            height=0.0,
            pitch=0.0,
            locomotion_hint=spot_command_pb2.HINT_AUTO,
            stair_hint=0,
            ee_vx=action[3],
            ee_vy=action[4],
            ee_vz=action[5],
        )

    @staticmethod
    def get_action_from_command(cmd: MobilityCommandBasic) -> np.ndarray:
        return np.array(
            [
                cmd.vx,
                cmd.vy,
                cmd.vrz,
                cmd.ee_vx,
                cmd.ee_vy,
                cmd.ee_vz,
            ]
        )

    def get_reward(self, action, next_obs, last_obs=None) -> float:
        return 0
        # return self.reward.predict(obs=next_obs, action=action, last_obs=last_obs)

    def is_done(self, obs: np.ndarray) -> bool:
        return False

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminate, truncate, info = super().step(action)
        info["dist"] = np.linalg.norm(obs[:2])
        info["angle"] = np.abs(np.arctan2(obs[3], obs[2]))
        return obs, reward, terminate, truncate, info

    def _show_ui(self):
        MANUAL_CONTROL_FREQ = 10
        input("called reset, press enter to continue...")
        prompt = input('type "yes" to enter options menu or press enter to continue: ')
        if len(prompt) == 0:
            return
        if prompt != "yes":
            print(f'entered "{prompt}". continuing to reset...')
        else:
            option = None
            while option != "c":
                option = input(
                    textwrap.dedent(
                        """
                    Options:
                    --------------
                    k: keyboard control
                    r: reset base position to current position
                    b: print hitbox
                    c: continue
                    h: why am I seeing this?
                    answer: """
                    )
                )
                while option not in ["k", "r", "c", "h", "b"]:
                    print(f'entered "{option}". invalid option...')
                    option = input("answer: ")
                if option == "k":
                    print(
                        self.__keyboard.kb_agent.description()
                        + "\nk: end manual control"
                    )
                    action = self.__keyboard.act(None)
                    while action is not None:
                        success, result = self._issue_unmonitored_command(
                            self.get_cmd_from_action(action), 1 / MANUAL_CONTROL_FREQ
                        )
                        if not success:
                            print(
                                "command failed, exiting manual control. press enter to continue..."
                            )
                            break
                        action = self.__keyboard.act(None)
                    input("manual control ended. press enter to go back to options...")
                elif option == "b":
                    print("hitbox:")
                    for row in self._get_spot_hitbox(self._read_robot_state()):
                        print(row)
                    input("press enter to go back to options...")
                elif option == "r":
                    print(
                        "WARNING: boundary safety checks and reset position will now be computed relative to the current pose"
                    )
                    confirm = input("confirm action by entering yes: ")
                    if confirm == "yes":
                        self.set_start_frame()
                        print("origin reset done")
                    else:
                        print("origin reset cancelled")
                elif option == "h":
                    print(
                        textwrap.dedent(
                            """
                        The robot continualy updates an estimate of its position using its onboard sensors
                        and camera. After long periods of operation, this estimate can drift from the truth,
                        which affects the safety checks done by the environment.
                        This interface allows you to move the robot to the position where the program was
                        started and reset the base pose that the environment uses to compute the safety
                        checks, which are expressed relative to this base pose."""
                        )
                    )
                    input("press enter to go back to options...")
                else:
                    print("continuing...")

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[Any, dict]:
        """
        Args:
            options:
                goal: [x,y,angle] sets the goal pose
        """
        if not self.__skip_ui:
            success = self._issue_stop()
            if not success:
                self.logger.error("Reset stop failed")
                raise RuntimeError("Reset stop failed")
            self._show_ui()
        if options is None:
            options = {}
        goal = options.get(
            "goal", None
        )  # optional goal expressed relative to environment frame
        if goal is None:
            goal = (self.config.start_x, self.config.start_y, self.config.start_angle)
        self.logger.info("Resetting environment with goal {}".format(goal))
        goal_pos = self.body_start_frame.inverse_pose(*goal)  # convert to vision frame
        self.__goal_frame = Frame2D(*goal_pos)
        return super().reset(seed=seed, options=options)


class SpotEnvAugmented(SpotEnvBasic):
    """
    SpotEnvAugmented extends SpotEnvBasic by allowing for control of the ee angular velocities.

    Additions:
    EE State Observation:
        sin(ee_rx): sin of roll of ee in global frame
        cos(ee_rx): cos of roll of ee in global frame
        sin(ee_ry): sin of pitch of ee in global frame
        cos(ee_ry): cos of pitch of ee in global frame
        sin(ee_rz): sin of yaw of ee in global frame
        cos(ee_rz): cos of yaw of ee in global frame

    EE Action:
        ee_vrx: roll angular velocity command for ee in body frame
        ee_vry: pitch angular velocity command for ee in body frame
        ee_vrz: yaw angular velocity command for ee in body frame
    """

    obs_shape = (13 + 6 + 3,)
    action_shape = (6 + 3,)

    def __init__(
        self,
        config,
        cmd_freq: float,
        goal_pos: np.ndarray = None,
        monitor_freq: float = 30,
        action_cost=0.0,
        skip_ui: bool = False,
    ):
        super().__init__(
            config=config,
            cmd_freq=cmd_freq,
            goal_pos=goal_pos,
            monitor_freq=monitor_freq,
            action_cost=action_cost,
            skip_ui=skip_ui,
        )

        # update observation and action spaces
        obs_low = np.concatenate(
            (
                self.observation_space.low,
                np.array([-1] * 6),
                np.array([-ARM_MAX_JOINT_VEL] * 3),
            )
        )
        obs_high = np.concatenate(
            (
                self.observation_space.high,
                np.array([1] * 6),
                np.array([ARM_MAX_JOINT_VEL] * 3),
            )
        )
        self.observation_space = spaces.Box(low=obs_low, high=obs_high)
        action_low = np.concatenate(
            (
                self.action_space.low,
                np.array([-ARM_MAX_JOINT_VEL] * 3),
            )
        )
        action_high = np.concatenate(
            (
                self.action_space.high,
                np.array([ARM_MAX_JOINT_VEL] * 3),
            )
        )
        self.action_space = spaces.Box(low=action_low, high=action_high)

    def get_obs_from_state(self, state: SpotState) -> np.ndarray:
        obs = super().get_obs_from_state(state)

        # ee orientation
        ee_pose = state.pose_of_hand_in_vision
        ee_qx, ee_qy, ee_qz, ee_qw = ee_pose[3], ee_pose[4], ee_pose[5], ee_pose[6]
        ee_rx, ee_ry, ee_rz = R.from_quat([ee_qx, ee_qy, ee_qz, ee_qw]).as_euler(
            "xyz", degrees=False
        )

        # ee angular velocities
        ee_vel = state.velocity_of_hand_in_vision
        ee_vrx, ee_vry, ee_vrz = ee_vel[3], ee_vel[4], ee_vel[5]

        # add to obs
        new_obs = np.array(
            [
                np.sin(ee_rx),
                np.cos(ee_rx),
                np.sin(ee_ry),
                np.cos(ee_ry),
                np.sin(ee_rz),
                np.cos(ee_rz),
                ee_vrx,
                ee_vry,
                ee_vrz,
            ]
        )
        obs = np.concatenate((obs, new_obs))
        return obs

    def get_cmd_from_action(self, action: np.ndarray, prev_state: SpotState) -> Command:
        return MobilityCommandAugmented(
            prev_state=prev_state,
            vx=action[0],
            vy=action[1],
            vrz=action[2],
            height=0.0,
            pitch=0.0,
            locomotion_hint=spot_command_pb2.HINT_AUTO,
            stair_hint=False,
            ee_vx=action[3],
            ee_vy=action[4],
            ee_vz=action[5],
            ee_vrx=action[6],
            ee_vry=action[7],
            ee_vrz=action[8],
        )

    @staticmethod
    def get_action_from_command(cmd: MobilityCommandAugmented) -> np.ndarray:
        return np.array(
            [
                cmd.vx,
                cmd.vy,
                cmd.vrz,
                cmd.ee_vx,
                cmd.ee_vy,
                cmd.ee_vz,
                cmd.ee_vrx,
                cmd.ee_vry,
                cmd.ee_vrz,
            ]
        )


class SpotEnvBasicDone(SpotEnvBasic):
    """Stops the robot when close to goal pose with low velocity"""

    def __init__(self, dist_tol, ang_tol, vel_tol, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dist_tol = dist_tol
        self.ang_tol = ang_tol
        self.vel_tol = vel_tol

    def is_done(self, obs: np.ndarray) -> bool:
        return (
            np.linalg.norm(obs[:2]) < self.dist_tol
            and np.abs(np.arctan2(obs[3], obs[2])) < self.ang_tol
            and np.linalg.norm(obs[4:]) < self.vel_tol
        )


def change_spot2d_obs_frame(
    obs: np.ndarray, origin: np.ndarray, theta: float
) -> np.ndarray:
    """
    Change the frame of the observation to the given origin and with the x axis tilted by angle theta.
    Parameters:
        obs: [..., 7] array of observations (x, y, cos, sin, vx, vy, w)
        origin: (x,y) origin of the new frame
        theta: angle in radians
    """
    new_obs = np.array(obs)
    new_obs[..., :2] = change_frame_2d(obs[..., :2], origin, theta, degrees=False)
    new_obs[..., 2:4] = rotate_2d_vector(obs[..., 2:4], -theta, degrees=False)
    new_obs[..., 4:6] = rotate_2d_vector(obs[..., 4:6], -theta, degrees=False)
    return new_obs
