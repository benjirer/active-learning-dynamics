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
from alrd.spot_gym.envs.record import Session, Episode
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
        log_dir: str | Path | None = None,
        session: Session | None = None,
        log_str: bool = False,
    ):
        """
        Args:
            cmd_freq: Environment's action frequency. Commands will take at approximately 1/cmd_freq seconds to execute.
            monitor_freq: Environment's desired state monitoring frequency for checking position boundaries.
            log_dir: Directory where to save environment logs.cmd
            session: Session object to record episode data.
            log_str: If True, command and state info is logged as a string to a file.
        If log_dir is not None and session is not None, session data will be dumped after each episode.
        """
        assert (
            session is None or log_dir is not None
        ), "If session is not None, log_dir must be specified"
        assert (
            not log_str or log_dir is not None
        ), "If log_str is True, log_dir must be specified"
        super().__init__(config, monitor_freq=monitor_freq)
        self.__cmd_freq = cmd_freq
        self.__should_reset = True
        self.__last_robot_state = None
        self.__current_episode = None
        self.__default_reset = np.array(
            (config.start_x, config.start_y, config.start_angle)
        )
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self.log_file = None
        self.session = session
        self.log_str = log_str
        if log_dir is not None:
            self.logger.addHandler(
                logging.FileHandler(self.log_dir / "spot_gym_with_arm.log")
            )

    @property
    def default_reset(self):
        return self.__default_reset

    def start(self):
        if self.log_dir is not None and not self.log_dir.is_dir():
            self.log_dir.mkdir(exist_ok=False, parents=True)
        super().start()

    def close(self):
        super().close()
        self._end_episode()

    def print_to_file(self, command: Command, state: SpotState, currentTime):
        if self.log_file is None:
            filepath = self.log_dir / ("session-" + get_timestamp_str() + ".txt")
            self.log_file = open(filepath, "w")
        self.log_file.write("time {{\n \tvalue: {:.5f} \n}}\n".format(currentTime))
        self.log_file.write(command.to_str() + "\n")
        self.log_file.write(state.to_str() + "\n")

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
        if self.log_dir is not None:
            if self.log_str and self.log_file is not None:
                self.log_file.close()
                self.log_file = None
            if (
                self.session is not None
                and self.__current_episode is not None
                and len(self.__current_episode) > 0
            ):
                self.session.add_episode(self.__current_episode)
                self.__current_episode = None
                pickle.dump(
                    self.session.asdict(), open(self.log_dir / "record.pkl", "wb")
                )

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
        cmd = self.get_cmd_from_action(action, self.__last_robot_state)
        next_state, cmd_time, read_time, oob = self._step(cmd)
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
        if self.session is not None:
            self.__current_episode.add(cmd, next_state, reward, done)
        if self.log_str:
            self.print_to_file(cmd, self.__last_robot_state, time.time())
        self.__last_robot_state = next_state
        if truncate:
            self._end_episode()
        if done:
            self.stop_robot()
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
        if self.session is not None:
            self.__current_episode = Episode(new_state)
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
