# imports
import time
import logging
import yaml
import pickle
import os
import numpy as np
import cv2
import wandb
import csv

# agents
from alrd.agent.absagent import Agent
from alrd.agent.keyboard import KeyboardAgent
from alrd.agent.xbox_eevel import SpotXboxEEVel
from alrd.agent.xbox_spacemouse import SpotXboxSpacemouse
from alrd.agent.xbox_random_jointpos import SpotXboxRandomJointPos

# environments
from alrd.spot_gym.envs.spot_envs_augmented import SpotEnvBasic, SpotEnvAugmented
from alrd.spot_gym.envs.spot_env_base import SpotEnvBase

# additionals
from gym.wrappers.rescale_action import RescaleAction
from alrd.spot_gym.model.robot_state import SpotState
from alrd.spot_gym.utils.utils import (
    BODY_MAX_VEL,
    BODY_MAX_ANGULAR_VEL,
    ARM_MAX_LINEAR_VEL,
)
from alrd.spot_gym.model.command import Command
from brax.training.types import Transition
import jax.numpy as jnp

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)


# data collection classes
# class to store individual transitions
class TransitionData:
    def __init__(
        self,
        step: int,
        last_obs: np.ndarray,
        action: np.ndarray,
        cmd: Command,
        reward,
        next_obs: np.ndarray,
        terminated: bool,
        truncated: bool,
        reset: bool = False,
    ):
        self.step = step
        self.last_obs = last_obs
        self.action = action
        self.cmd = cmd
        self.reward = reward
        self.next_obs = next_obs
        self.terminated = terminated
        self.truncated = truncated
        self.reset = reset


# class to store individual states, action and delta_t
class StateData:
    def __init__(
        self,
        step: int,
        last_state: SpotState,
        next_state: SpotState,
        action: np.ndarray,
        delta_t: float = 0,
    ):
        self.step = step
        self.last_state = last_state
        self.next_state = next_state
        self.action = action
        self.delta_t = delta_t


# temporary: times of processes
class TimeData:
    def __init__(
        self,
        step: int,
        agent_time: float,
        step_time: float,
        save_time: float,
        cmd_time: float,
        inner_step_time: float,
        additional_time: float,
        inner_cmd_time: float,
        inner_read_time: float,
    ):
        self.step = step
        self.agent_time = agent_time
        self.step_time = step_time
        self.save_time = save_time
        self.cmd_time = cmd_time
        self.inner_step_time = inner_step_time
        self.additional_time = additional_time
        self.inner_cmd_time = inner_cmd_time
        self.inner_read_time = inner_read_time


# class to store all data vectors for an episode
class DataBuffer:
    def __init__(
        self,
        states: list[StateData] = [],
        observations: list[np.ndarray] = [],
        transitions: list[TransitionData] = [],
        times: list[TimeData] = [],
        brax_transitions: list[Transition] = [],
    ):
        self.states = states
        self.observations = observations
        self.transitions = transitions
        self.times = times
        self.brax_transitions = brax_transitions


# class to store all DataBuffers for a session
class SessionBuffer:
    def __init__(self, data_buffers: list[DataBuffer] = []):
        self.data_buffers = data_buffers


# save data function
def save_data(session_buffer: SessionBuffer, session_dir: str):
    # save brax transitions separately
    all_brax_transitions = []
    for data_buffer in session_buffer.data_buffers:
        all_brax_transitions.extend(data_buffer.brax_transitions)
    brax_transitions_path = os.path.join(session_dir, "brax_transitions.pickle")
    with open(brax_transitions_path, "wb") as f:
        pickle.dump(all_brax_transitions, f)

    # save session buffer
    session_path = os.path.join(session_dir, "session_buffer.pickle")
    with open(session_path, "wb") as f:
        pickle.dump(session_buffer, f)


# run episode
def run(
    agent: Agent,
    env: SpotEnvBase,
    num_steps: int = 1000,
    cmd_freq: float = 10,
    collect_data: bool = False,
    data_buffer: DataBuffer = None,
    session_dir: str | None = None,
    action_scale: float = 1.0,
):

    started = False
    step = 0
    recent_state = None
    delta_t = 0
    start_t = time.time()

    while step < num_steps:
        # logger.info("Step %s" % step)
        # if not started, reset the environment
        if not started:
            logger.info("Agent description: %s" % agent.description())
            count = 0
            obs, info = env.reset()
            delta_t = start_t - time.time()
            start_t = time.time()
            if collect_data:
                data_buffer.observations.append(obs)
                data_buffer.states.append(
                    StateData(
                        step,
                        None,
                        info["next_state"],
                        None,
                        delta_t,
                    )
                )
                data_buffer.transitions.append(
                    TransitionData(
                        step,
                        None,
                        None,
                        None,
                        0,
                        obs,
                        False,
                        False,
                        True,
                    )
                )
                data_buffer.times.append(
                    TimeData(
                        step,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    )
                )
                data_buffer.brax_transitions.append(
                    Transition(
                        observation=jnp.array(obs),
                        action=jnp.zeros(9),
                        reward=jnp.array(0),
                        discount=jnp.array(0.99),
                        next_observation=jnp.array(
                            env.get_obs_from_state(info["next_state"])
                        ),
                    )
                )
            if obs is None:
                return
            agent.reset()
            recent_state = info["next_state"]
            started = True

        # get action from agent
        agent_time = time.time()
        action = agent.act(obs, recent_state)
        action = action_scale * action

        delta_t_agent = agent_time - time.time()

        # step the environment
        if action is not None:
            step_time = time.time()
            next_obs, reward, terminated, truncated, info = env.step(action)
            delta_t_step = time.time() - step_time

            delta_t = start_t - time.time()
            start_t = time.time()
            recent_state = info["next_state"]
            save_time = time.time()
            if collect_data:
                data_buffer.observations.append(next_obs)
                data_buffer.transitions.append(
                    TransitionData(
                        step,
                        obs,
                        action,
                        info["cmd"],
                        reward,
                        next_obs,
                        terminated,
                        truncated,
                        False,
                    )
                )
                data_buffer.states.append(
                    StateData(
                        step,
                        info["last_state"],
                        info["next_state"],
                        action,
                        delta_t,
                    )
                )
                data_buffer.brax_transitions.append(
                    Transition(
                        observation=jnp.array(obs),
                        action=jnp.array(action),
                        reward=jnp.array(0),
                        # reward=jnp.array(reward),
                        discount=jnp.array(0.99),
                        next_observation=jnp.array(next_obs),
                    ),
                )
                delta_t_save = time.time() - save_time
                data_buffer.times.append(
                    TimeData(
                        step,
                        delta_t_agent,
                        delta_t_step,
                        delta_t_save,
                        info["delta_t_cmd"],
                        info["delta_t_inner_step"],
                        info["delta_t_additional_time"],
                        info["delta_t_inner_cmd_time"],
                        info["delta_t_inner_read_time"],
                    )
                )
            if next_obs is not None:
                count += 1
                step += 1

        # check if episode is terminated
        if action is None or terminated or truncated:
            started = False
            if count > 0:
                # logger.info("Terminated %s. Truncated %s" % (terminated, truncated))
                return
        else:
            obs = next_obs

    env.stop_robot()


# start experiment
def start_experiment(
    # experiment settings
    num_episodes: int = 1,
    num_steps: int = 100,
    cmd_freq: int = 20,
    collect_data: bool = False,
    data_tag: str = "v5_0",
    action_scale: float = 1.0,
):

    # import real world config
    config = yaml.load(
        open(
            "/home/bhoffman/Documents/MT FS24/active-learning-dynamics/config/crl_spot.yaml",
            "r",
        ),
        Loader=yaml.Loader,
    )

    # set up data collection directory
    session_dir = None
    session_buffer = None

    # set up data collection and save experiment settings
    if collect_data:
        session_buffer = SessionBuffer()
        experiment_id = "test" + time.strftime("%Y%m%d-%H%M%S") + "_" + data_tag
        session_dir = (
            "/home/bhoffman/Documents/MT FS24/active-learning-dynamics/collected_data/"
            + experiment_id
        )
        experiment_settings = [
            "num_episodes: {}".format(num_episodes),
            "num_steps: {}".format(num_steps),
            "cmd_freq: {}".format(cmd_freq),
        ]
        os.makedirs(session_dir, exist_ok=True)
        settings_path = os.path.join(session_dir, "experiment_settings.csv")
        with open(settings_path, mode="w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            for setting in experiment_settings:
                writer.writerow([setting])

    # run episodes
    episode = 0
    while episode < num_episodes:

        # set up data collection for episode
        data_buffer = None
        if collect_data:
            data_buffer = DataBuffer()

        """create env"""
        # note: make sure env and agent are compatible
        # if not download_mode:
        #     env = SpotEnvBasic(
        #         config,
        #         cmd_freq=cmd_freq,
        #         action_cost=0.005,
        #         goal_pos=goal,
        #     )

        env = SpotEnvAugmented(
            config,
            cmd_freq=cmd_freq,
        )

        # env = SpotEEVelEnv(
        #     config,
        #     cmd_freq=cmd_freq,
        #     log_str=False,
        # )
        # env = SpotJointPosEnv(
        #     config,
        #     cmd_freq=cmd_freq,
        #     log_str=False,
        # )

        """create agent"""
        # agent = get_offline_trained_agent(
        #     state_dim=13,
        #     action_dim=6,
        #     goal_dim=3,
        #     goal=goal,
        #     project_name=project_name,
        #     run_id=run_id,
        #     offline_mode=False if download_mode else True,
        # )

        # agent = KeyboardAgent(xy_speed=1, a_speed=1)
        # agent = SpotXboxEEVel(base_speed=1, base_angular=1, ee_speed=1.0)
        agent = SpotXboxSpacemouse(
            base_speed=1.0,
            base_angular=1.0,
            ee_speed=1.5,
            ee_control_mode="augmented",
        )

        # start env
        env.start()
        logger.info("env: %s. obs: %s" % (type(env), env.observation_space.shape))
        # env = RescaleAction(env, min_action=-1, max_action=1)

        # run current episode
        try:
            run(
                agent,
                env,
                num_steps=num_steps,
                cmd_freq=cmd_freq,
                collect_data=collect_data,
                data_buffer=data_buffer,
                session_dir=session_dir,
                action_scale=action_scale,
            )
        except KeyboardInterrupt:
            logger.info("Exiting due to keyboard interrupt")
            env.stop_robot()
            env.close()
            if collect_data:
                session_buffer.data_buffers.append(data_buffer)
                save_data(session_buffer, session_dir)
        except Exception as e:
            logger.error("Exiting due to exception: %s" % e)
            env.stop_robot()
            env.close()
            if collect_data:
                session_buffer.data_buffers.append(data_buffer)
        finally:
            logger.info("Exiting due to finish")
            env.stop_robot()
            env.close()
            if collect_data:
                session_buffer.data_buffers.append(data_buffer)

        episode += 1

    if collect_data:
        save_data(session_buffer, session_dir)


if __name__ == "__main__":

    """============== SETTINGS =============="""
    num_episodes = 1
    num_steps = 50000
    cmd_freq = 10
    collect_data = False
    data_tag = "data_collection"
    action_scale = 1.0

    start_experiment(
        num_episodes=num_episodes,
        num_steps=num_steps,
        cmd_freq=cmd_freq,
        collect_data=collect_data,
        data_tag=data_tag,
        action_scale=action_scale,
    )
