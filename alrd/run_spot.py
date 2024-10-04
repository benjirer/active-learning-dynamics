# imports
import time
import logging
import yaml
import pickle
import os
import numpy as np
import cv2
import wandb

# agents
from alrd.agent.absagent import Agent
from alrd.agent.keyboard import KeyboardAgent
from alrd.agent.xbox_eevel import SpotXboxEEVel
from alrd.agent.xbox_spacemouse import SpotXboxSpacemouse
from alrd.agent.xbox_random_jointpos import SpotXboxRandomJointPos
from alrd.agent.offline_trained import OfflineTrainedAgent

# environments
from alrd.spot_gym.envs.spot_eevel_cart_body import SpotEEVelEnv

# from alrd.spot_gym.envs.spot_eevel_cyl import SpotEEVelEnv
from alrd.spot_gym.envs.spot_jointpos import SpotJointPosEnv
from alrd.spot_gym.envs.spot_basic import SpotBasicEnv
from alrd.spot_gym.envs.spotgym import SpotGym

# additionals
from gym.wrappers.rescale_action import RescaleAction
from alrd.spot_gym.model.robot_state import SpotState
from alrd.spot_gym.utils.utils import (
    BODY_MAX_VEL,
    BODY_MAX_ANGULAR_VEL,
    ARM_MAX_LINEAR_VEL,
)
from alrd.spot_gym.model.command import Command

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
    ):
        self.states = states
        self.observations = observations
        self.transitions = transitions
        self.times = times


# class to store all DataBuffers for a session
class SessionBuffer:
    def __init__(self, data_buffers: list[DataBuffer] = []):
        self.data_buffers = data_buffers


# get offline trained agent
def get_offline_trained_agent(
    state_dim: int,
    action_dim: int,
    goal_dim: int,
) -> Agent:
    offline_mode = True

    project_name = "badass_testing"
    run_id = "i3i1pin4"
    local_dir = "badass_testing"

    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    # fetch learned policy
    if not offline_mode:
        wandb_api = wandb.Api()

        run = wandb_api.run(f"{project_name}/{run_id}")

        # save policy params
        run.file("models/parameters.pkl").download(
            replace=True, root=os.path.join(local_dir)
        )

        # get reward config
        reward_keys = [
            "encode_angle",
            "ctrl_cost_weight",
            "margin_factor",
            "ctrl_diff_weight",
        ]
        reward_config = {}
        for key in reward_keys:
            reward_config[key] = run.config[key]

        # save reward config
        with open(os.path.join(local_dir, "reward_config.yaml"), "w") as file:
            yaml.dump(reward_config, file)

    # get policy params
    policy_params = pickle.load(
        open(os.path.join(local_dir, "models/parameters.pkl"), "rb")
    )

    # get reward config
    reward_config = yaml.load(
        open(os.path.join(local_dir, "reward_config.yaml"), "r"),
        Loader=yaml.Loader,
    )

    # get SAC_KWARGS
    NUM_ENV_STEPS_BETWEEN_UPDATES = 16
    NUM_ENVS = 64
    sac_num_env_steps = 1_000_000
    horizon_len = 50
    SAC_KWARGS = dict(
        num_timesteps=sac_num_env_steps,
        num_evals=20,
        reward_scaling=10,
        episode_length=horizon_len,
        episode_length_eval=2 * horizon_len,
        action_repeat=1,
        discounting=0.99,
        lr_policy=3e-4,
        lr_alpha=3e-4,
        lr_q=3e-4,
        num_envs=NUM_ENVS,
        batch_size=64,
        grad_updates_per_step=NUM_ENV_STEPS_BETWEEN_UPDATES * NUM_ENVS,
        num_env_steps_between_updates=NUM_ENV_STEPS_BETWEEN_UPDATES,
        tau=0.005,
        wd_policy=0,
        wd_q=0,
        wd_alpha=0,
        num_eval_envs=2 * NUM_ENVS,
        max_replay_size=5 * 10**4,
        min_replay_size=2**11,
        policy_hidden_layer_sizes=(64, 64),
        critic_hidden_layer_sizes=(64, 64),
        normalize_observations=True,
        deterministic_eval=True,
        wandb_logging=False,
    )

    agent = OfflineTrainedAgent(
        policy_params=policy_params,
        reward_config=reward_config,
        state_dim=state_dim,
        action_dim=action_dim,
        goal_dim=goal_dim,
        SAC_KWARGS=SAC_KWARGS,
    )
    return agent


# run episode
def run(
    agent: Agent,
    env: SpotGym,
    num_steps: int = 1000,
    cmd_freq: float = 10,
    collect_data: bool = False,
    data_buffer: DataBuffer = None,
    session_dir: str | None = None,
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
            if obs is None:
                return
            agent.reset()
            recent_state = info["next_state"]
            started = True

        # get action from agent
        agent_time = time.time()
        # action = agent.act(obs, recent_state)
        action = agent.act(obs)
        # clip action
        # action[:3] = [0.0, 0.0, 0.0]
        print(action)
        # action = np.clip(action, -0.2, 0.2)
        action = 0.4 * action
        print(action)
        delta_t_agent = agent_time - time.time()
        # logger.info("Action %s" % action)

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
def start_experiment():

    # experiment settings
    num_episodes = 1
    num_steps = 100
    cmd_freq = 20
    collect_data = False
    random_seed = 0

    # random seeds for noise sampling
    sampling_seeds = np.random.default_rng(seed=random_seed).integers(
        0, 2**32, size=num_episodes
    )

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

    if collect_data:
        session_buffer = SessionBuffer()
        tag = "v5_0"
        experiment_id = "test" + time.strftime("%Y%m%d-%H%M%S") + "_" + tag
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
        settings_path = os.path.join(session_dir, "experiment_settings.pickle")
        open(
            settings_path,
            "wb",
        ).write(pickle.dumps(experiment_settings))

    episode = 0
    while episode < num_episodes:

        data_buffer = None
        if collect_data:
            data_buffer = DataBuffer()

        """create env"""
        # note: make sure env and agent are compatible
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
        env = SpotBasicEnv(
            config,
            cmd_freq=cmd_freq,
        )

        """create agent"""
        # agent = KeyboardAgent(xy_speed=1, a_speed=1)
        # agent = SpotXboxEEVel(base_speed=1, base_angular=1, ee_speed=1.0)
        # agent = SpotXboxSpacemouse(
        #     base_speed=1.0,
        #     base_angular=1.0,
        #     ee_speed=1.0,
        #     ee_control_mode="basic",
        # )
        # agent = SpotXboxRandomJointPos(
        #     base_speed=1.0,
        #     base_angular=1.0,
        #     arm_joint_speed=1.0,
        #     cmd_freq=cmd_freq,
        #     steps=num_steps,
        #     random_seed=sampling_seeds[episode],
        # )
        agent = get_offline_trained_agent(
            state_dim=13,
            action_dim=6,
            goal_dim=3,
        )

        # # test agent
        # for _ in range(10):
        #     obs = [
        #         0.0,  # base x
        #         0.0,  # base y
        #         0.0,  # sin theta
        #         0.0,  # cos theta
        #         0.0,  # base x vel
        #         0.0,  # base y vel
        #         0.0,  # base angular vel
        #         2.0,  # ee x
        #         0.0,  # ee y
        #         0.7,  # ee z
        #         0.0,  # ee x vel
        #         0.0,  # ee y vel
        #         0.0,  # ee z vel
        #     ]
        #     action = agent.act(np.array(obs))
        #     print(action)
        # return None

        # start env
        env.start()
        logger.info("env: %s. obs: %s" % (type(env), env.observation_space.shape))
        # env = RescaleAction(env, min_action=-1, max_action=1)

        # run episode
        try:
            run(
                agent,
                env,
                num_steps=num_steps,
                cmd_freq=cmd_freq,
                collect_data=collect_data,
                data_buffer=data_buffer,
                session_dir=session_dir,
            )
        except KeyboardInterrupt:
            logger.info("Exiting due to keyboard interrupt")
            env.stop_robot()
            env.close()
            if collect_data:
                session_buffer.data_buffers.append(data_buffer)
                session_path = os.path.join(session_dir, "session_buffer.pickle")
                open(
                    session_path,
                    "wb",
                ).write(pickle.dumps(session_buffer))
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
        session_path = os.path.join(session_dir, "session_buffer.pickle")
        open(
            session_path,
            "wb",
        ).write(pickle.dumps(session_buffer))


if __name__ == "__main__":
    start_experiment()
