from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from alrd.spot_gym_with_arm.model.spot import SpotEnvironmentConfig
from alrd.spot_gym_with_arm.envs.spot2d import (
    Spot2DEnv,
    Spot2DEnvDone,
    Spot2DReward,
    change_spot2d_obs_frame,
)
from alrd.spot_gym_with_arm.envs.simulate2d import Spot2DEnvSim, Spot2DModelSim
from alrd.spot_gym_with_arm.envs.spotgym import SpotGym
from alrd.spot_gym_with_arm.wrappers.operational_wrappers import (
    QueryGoalWrapper,
    QueryStartWrapper,
)
from alrd.spot_gym_with_arm.envs.random_pos import RandomPosInit
from jax import vmap

from opax.utils.replay_buffer import (
    ReplayBuffer,
    Transition,
    ReplayBuffer,
    ReplayBuffer,
)
from opax.models.dynamics_model import DynamicsModel

__all__ = [
    "SpotGym",
]
GOAL = (2.5, 1.8)


def create_spot_env(
    config: SpotEnvironmentConfig,
    cmd_freq: float,
    monitor_freq: float = 30,
    log_dir: str | Path | None = None,
    query_goal: bool = False,
    action_cost: float = 0.0,
    velocity_cost: float = 0.0,
    simulated: bool = False,
    dynamics_model: DynamicsModel | None = None,
    seed: int | None = None,
    random_init_pose: Tuple[float, float, float, float] | None = None,
    done_on_goal_tol: Tuple[float, float, float] | None = None,
):
    """
    Creates and initializes spot environment.
    """
    assert not random_init_pose or not query_goal
    assert not random_init_pose or seed
    if not simulated:
        if done_on_goal_tol is not None:
            # the episode is ended when the robot is at certain distance and angle
            # from the goal with velocity smaller than the one specified
            env = Spot2DEnvDone(
                done_on_goal_tol[0],
                done_on_goal_tol[1],
                done_on_goal_tol[2],
                config,
                cmd_freq,
                monitor_freq,
                log_dir=log_dir,
                action_cost=action_cost,
                velocity_cost=velocity_cost,
            )
        else:
            env = Spot2DEnv(
                config,
                cmd_freq,
                monitor_freq,
                log_dir=log_dir,
                log_str=False,
                action_cost=action_cost,
                velocity_cost=velocity_cost,
            )
    else:
        if dynamics_model is None:
            env = Spot2DEnvSim(
                config,
                cmd_freq,
                action_cost=action_cost,
                velocity_cost=velocity_cost,
            )
        else:
            env = Spot2DModelSim(
                dynamics_model,
                config,
                action_cost=action_cost,
                velocity_cost=velocity_cost,
            )
    if query_goal:
        env = QueryStartWrapper(env)
        env = QueryGoalWrapper(env)
    if random_init_pose:
        env = RandomPosInit(env, seed, random_init_pose[:2], random_init_pose[2:])
    return env


def load_dataset(
    buffer_path: str,
    goal=None,
    action_cost: float = 0.0,
    velocity_cost: float = 0.0,
    normalize: bool = True,
    action_normalize: bool = False,
    learn_deltas: bool = True,
):
    """
    Parameters
        buffer_path: path to input buffer
        goal: goal position (x, y)
        action_cost: action cost used to compute reward when goal is specified
        velocity_cost: velocity cost used to compute reward when goal is specified
        action_normalize: whether to normalize actions
    """
    data = pickle.load(open(buffer_path, "rb"))
    obs_shape = (7,)
    action_shape = (3,)
    assert isinstance(data, ReplayBuffer)
    buffer = ReplayBuffer(
        obs_shape=obs_shape,
        action_shape=action_shape,
        normalize=normalize,
        action_normalize=action_normalize,
        learn_deltas=learn_deltas,
    )
    if goal is not None:
        reward_model = Spot2DReward.create(
            action_coeff=action_cost, velocity_coeff=velocity_cost
        )
        reward_fn = reward_model.predict
    tran = data.get_full_raw_data()
    if goal is not None:
        tran.obs[:] = change_spot2d_obs_frame(tran.obs, goal[:2], goal[2])
        tran.next_obs[:] = change_spot2d_obs_frame(tran.next_obs, goal[:2], goal[2])
        tran.reward[:, 0] = reward_fn(tran.next_obs, tran.action)[:]
    buffer.add(tran)
    return buffer


def load_episodic_dataset(
    buffer_path: str,
    goal=None,
    action_cost: float = 0.0,
    velocity_cost: float = 0.0,
    normalize: bool = True,
    action_normalize: bool = False,
    learn_deltas: bool = True,
    episode_len: Optional[int] = None,
):
    """
    Parameters
        buffer_path: path to input buffer
        usepast: number of past observations to include in sampled observation
        usepastact: whether to include past actions in sampled observation
        goal: goal position (x, y)
        action_cost: action cost used to compute reward when goal is specified
        velocity_cost: velocity cost used to compute reward when goal is specified
        action_normalize: whether to normalize actions
    """
    data = pickle.load(open(buffer_path, "rb"))
    obs_shape = (7,)
    action_shape = (3,)
    # hide_in_obs = [0,1,2,3]
    hide_in_obs = None
    assert isinstance(data, ReplayBuffer)
    if episode_len is None:
        assert isinstance(data, ReplayBuffer)
        num_episodes = data.num_episodes
    else:
        num_episodes = data.size // episode_len
    buffer = ReplayBuffer(
        obs_shape=obs_shape,
        action_shape=action_shape,
        normalize=normalize,
        action_normalize=action_normalize,
        learn_deltas=learn_deltas,
    )
    if goal is not None:
        reward_model = Spot2DReward.create(
            action_coeff=action_cost, velocity_coeff=velocity_cost
        )
        reward_fn = reward_model.predict
    for i in range(num_episodes):
        if episode_len is None:
            tran = data.get_episode(i)
        else:
            tran = Transition(
                obs=data.obs[i * episode_len : (i + 1) * episode_len],
                action=data.action[i * episode_len : (i + 1) * episode_len],
                next_obs=data.next_obs[i * episode_len : (i + 1) * episode_len],
                reward=data.reward[i * episode_len : (i + 1) * episode_len],
                done=data.done[i * episode_len : (i + 1) * episode_len],
            )
        if goal is not None:
            tran.obs[:] = change_spot2d_obs_frame(tran.obs, goal[:2], goal[2])
            tran.next_obs[:] = change_spot2d_obs_frame(tran.next_obs, goal[:2], goal[2])
            tran.reward[:, 0] = reward_fn(tran.next_obs, tran.action)[:]
        buffer.add(tran)

    return buffer


def add_2d_zero_samples(buffer: ReplayBuffer, num_samples: int, reward_model=None):
    """
    Adds zero samples to buffer.
    """
    action_shape = (3,)
    pose = np.random.uniform(
        low=[buffer.obs[:, 0].min(), buffer.obs[:, 1].min(), -np.pi],
        high=[buffer.obs[:, 0].max(), buffer.obs[:, 1].max(), np.pi],
        size=(num_samples, 3),
    )
    obs = np.hstack(
        [
            pose[:, :2],
            np.cos(pose[:, [2]]),
            np.sin(pose[:, [2]]),
            np.zeros((num_samples, 3)),
        ]
    )
    if reward_model is None:
        reward = np.zeros((num_samples, 1))
    else:
        reward = vmap(reward_model.predict)(obs, np.zeros((num_samples, *action_shape)))
    buffer.add(
        Transition(
            obs=obs,
            action=np.zeros((num_samples, *action_shape)),
            next_obs=obs,
            reward=reward,
            done=np.zeros((num_samples, 1)),
        )
    )


def get_first_n(buffer: ReplayBuffer, n: int):
    """Get a buffer with same args as buffer but with only first n transitions"""
    tran = buffer.get_full_raw_data()
    new_tran = Transition(
        obs=tran.obs[:n],
        action=tran.action[:n],
        next_obs=tran.next_obs[:n],
        reward=tran.reward[:n],
        done=tran.done[:n],
    )
    new_buffer = ReplayBuffer(
        obs_shape=buffer.obs_shape,
        action_shape=buffer.action_shape,
        normalize=buffer.normalize,
        action_normalize=buffer.action_normalize,
        learn_deltas=buffer.learn_deltas,
    )
    new_buffer.add(new_tran)
    return new_buffer
