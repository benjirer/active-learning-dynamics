from __future__ import annotations

import logging
import pickle
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple

import gym
import numpy as np
from alrd.spot_gym.model.command import Command
from alrd.spot_gym.model.robot_state import SpotState
from alrd.spot_gym.model.spot import (
    SpotBaseStateMachine,
    SpotEnvironmentConfig,
)
from alrd.utils.utils import get_timestamp_str


class SpotGym(SpotBaseStateMachine, gym.Env, ABC):
    def __init__(
        self,
        config: SpotEnvironmentConfig,
        cmd_freq: float,
        monitor_freq: float = 30,
    ):
        """
        Args:
            cmd_freq: Environment's action frequency. Commands will take at approximately 1/cmd_freq seconds to execute.
            monitor_freq: Environment's desired state monitoring frequency for checking position boundaries.
        """
        super().__init__(config, monitor_freq=monitor_freq)
        self.__cmd_freq = cmd_freq
        self.__should_reset = True
        self.__last_robot_state = None
        self.__default_reset = np.array(
            (config.start_x, config.start_y, config.start_angle)
        )

    @property
    def default_reset(self):
        return self.__default_reset

    def start(self):
        super().start()

    def close(self):
        super().close()
        self._end_episode()

    def stop_robot(self) -> bool:
        """Stops the robot and ends the current episode"""
        if not self.isopen:
            raise RuntimeError("Robot was shutdown, but stop was called.")
        result = self._issue_stop()
        self._end_episode()
        return result

    @property
    def should_reset(self):
        return self.__should_reset

    def _end_episode(self):
        self.__should_reset = True
        self.__last_robot_state = None

    def _step(
        self, cmd: Command
    ) -> Tuple[SpotState | None, float, float | None, bool | None]:
        """
        Apply the command for as long as the command period specified.
        Returns:
            new_state: The new state of the robot after the command is applied. This is None if the state service timed out reading the state.
            cmd_time: The time it took to issue the command + read the state.
            read_time: The time it took to read the state of the robot.
            oob: Whether the robot was out of bounds in the new state returned
        """
        if not self.isopen or self.should_reset:
            if not self.isopen:
                raise RuntimeError("Robot was shutdown, but step was called.")
            else:
                raise RuntimeError("Environment should be reset but step was called.")
        start_cmd = time.time()
        success, result = self._issue_command(cmd, 1 / self.__cmd_freq)
        cmd_time = time.time() - start_cmd
        if not success:
            # command was interrupted
            return None, cmd_time, None, None
        next_state, read_time, oob = result
        return next_state, cmd_time, read_time, oob

    @abstractmethod
    def get_obs_from_state(self, state: SpotState) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_cmd_from_action(
        self, action: np.ndarray, prev_state: np.ndarray
    ) -> Command:
        raise NotImplementedError

    @abstractmethod
    def get_reward(self, action: np.ndarray, next_obs: np.ndarray) -> float:
        raise NotImplementedError

    @abstractmethod
    def is_done(self, obs: np.ndarray) -> bool:
        raise NotImplementedError

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Converts the action to a robot command, applies it and returns the next state as a numpy array
        """
        cmd_time = time.time()
        cmd = self.get_cmd_from_action(action, self.__last_robot_state)
        delta_t_cmd = time.time() - cmd_time
        inner_step_time = time.time()
        next_state, cmd_time, read_time, oob = self._step(cmd)
        delta_t_inner_step = time.time() - inner_step_time
        additional_time = time.time()
        info = {
            "cmd": cmd,
            "last_state": self.__last_robot_state,
            "next_state": next_state,
            "cmd_time": cmd_time,
            "read_time": read_time,
            "oob": oob,
        }
        truncate = False
        if next_state is None:
            # if command was interrupted, return previous state
            truncate = True
            next_state = self.__last_robot_state
        if oob:
            truncate = True
        obs = self.get_obs_from_state(next_state)
        reward = self.get_reward(action, obs)
        done = self.is_done(obs)
        self.__last_robot_state = next_state
        if truncate:
            self._end_episode()
        if done:
            self.stop_robot()

        delta_t_additional_time = time.time() - additional_time

        info["delta_t_cmd"] = delta_t_cmd
        info["delta_t_inner_step"] = delta_t_inner_step
        info["delta_t_additional_time"] = delta_t_additional_time
        info["delta_t_inner_cmd_time"] = cmd_time
        info["delta_t_inner_read_time"] = read_time
        return obs, reward, done, truncate, info

    def _reset(self, pose: np.ndarray) -> Tuple[SpotState, float]:
        """
        Reset the robot to the starting pose.
        Args:
            pose: [x, y, angle] where to reset the robot in global coordinates
        Returns:
            new_state: starting state, None if
        Raises:
            RuntimeError if the robot reset fails
        """
        if not self.isopen:
            raise RuntimeError("Robot is shutdown, but reset was called.")
        if not self.should_reset:
            self._end_episode()
        self.logger.debug("Reset called, stopping robot...")
        success = self._issue_stop()
        if not success:
            self.logger.error("Reset stop command failed")
            raise RuntimeError(
                "Failed to perform environment reset. Robot stop failed."
            )
        self.logger.debug("Robot stopped")
        # reset position
        success, _ = self._issue_reset(*pose)
        self.logger.info("Resetting robot position...")
        if not success:
            self.logger.error("Failed to reset robot position")
            raise RuntimeError(
                "Failed to perform environment reset. Pose reset failed."
            )
        start = time.time()
        new_state = self._read_robot_state()
        read_time = time.time() - start
        self.__should_reset = False
        self.__last_robot_state = new_state
        return new_state, read_time

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Args:
            seed: set the seed for the environment's random number generator.
            options: additional options for the reset, such as:
                pose: [x,y,angle] where the robot should be reset, defaults to the starting pose specified
                    in the configuration
        Raises:
            RuntimeError if the robot reset fails
        """
        super().reset(seed=seed, options=options)
        if options is None:
            options = {}
        pose = options.get("pose", self.__default_reset)
        assert isinstance(pose, np.ndarray) and pose.shape == (
            3,
        ), "Pose must be an array of size 3: (x, y, angle)"
        result = self._reset(pose)
        state, read_time = result
        info = {"next_state": state, "read_time": read_time}
        obs = self.get_obs_from_state(state)
        self.logger.info("Resetting with initial observation {}".format(obs))
        return obs, info
