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
from typing import Union

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


# get offline trained agent
def get_offline_trained_agent(
    state_dim: int,
    action_dim: int,
    goal_dim: int,
    goal: np.ndarray,
    project_name: str,
    run_id: str,
    offline_mode,
) -> Agent:

    local_dir = "saved_models/" + project_name + "_" + run_id

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
        goal=goal,
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
    action_scale: float = 1.0,
    num_frame_stack: int = 0,
):

    started = False
    step = 0
    recent_state = None
    delta_t = 0
    start_t = time.time()
    action_buffer = np.zeros(6 * num_frame_stack)

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
                        action=jnp.zeros(6),
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
        # action = agent.act(obs, recent_state)

        action = agent.act(obs, action_buffer)
        action = action_scale * action

        # update action buffer
        if num_frame_stack > 0:
            action_buffer = np.concatenate([action_buffer[6:], action], axis=0)

        # clip for safety
        # action = np.clip(action, -1.0, 1.0)
        # scale acrions to max
        # base vel: 1.6, ang vel: 1.5, ee_vel: 2.5
        # action = np.array(
        #     [
        #         action[0] * 1.6,
        #         action[1] * 1.6,
        #         action[2] * 1.5,
        #         action[3] * 2.0,
        #         action[4] * 2.0,
        #         action[5] * 2.0,
        #     ]
        # )

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
                        reward=agent.get_reward(
                            obs=obs,
                            action=action,
                            next_obs=next_obs,
                        ),
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
    # general
    download_mode: bool = False,
    # experiment settings
    num_episodes: int = 1,
    num_steps: int = 100,
    cmd_freq: int = 20,
    collect_data: bool = False,
    data_tag: str = "v5_0",
    action_scale: float = 1.0,
    # policy settings
    goal: np.array = np.array([0.0, 0.0, 0.7]),
    project_name: str = "jitter_testing",
    run_id: str = "p71lprz0",
    model_type: str = "sim-model",
    data_size: int = 800,
    seed_id: int = 1,
    goal_id: Union[int, str] = 0,
    num_frame_stack: int = 0,
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
        experiment_id = f"{project_name}/{goal_id}/test_{data_tag}"
        session_dir = (
            "/home/bhoffman/Documents/MT FS24/active-learning-dynamics/shape_experiment_data/"
            + experiment_id
        )
        version = 0
        # if directory exists, increment version, else just create new with version 0
        while os.path.exists(session_dir):
            version += 1
            session_dir = (
                "/home/bhoffman/Documents/MT FS24/active-learning-dynamics/shape_experiment_data/"
                + experiment_id
                + "_v"
                + str(version)
            )

        experiment_settings = [
            "num_episodes: {}".format(num_episodes),
            "num_steps: {}".format(num_steps),
            "cmd_freq: {}".format(cmd_freq),
            # "goal: {}".format(goal),
            "project_name: {}".format(project_name),
            "action_scale: {}".format(action_scale),
            "run_id: {}".format(run_id),
            "model_type : {}".format(model_type),
            "data_size: {}".format(data_size),
            "seed_id: {}".format(seed_id),
            "goal_id: {}".format(goal_id),
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
        if not download_mode:
            env = SpotBasicEnv(
                config,
                cmd_freq=cmd_freq,
                action_cost=0.005,
                goal_pos=goal[0],
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
        agent = get_offline_trained_agent(
            state_dim=13,
            action_dim=6,
            goal_dim=3,
            goal=goal,
            project_name=project_name,
            run_id=run_id,
            offline_mode=False if download_mode else True,
        )

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

        # test downloaded policy
        if download_mode:
            for _ in range(10):
                obs = [
                    0.0,  # base x
                    0.0,  # base y
                    0.0,  # sin theta
                    0.0,  # cos theta
                    0.0,  # base x vel
                    0.0,  # base y vel
                    0.0,  # base angular vel
                    2.0,  # ee x
                    0.0,  # ee y
                    0.7,  # ee z
                    0.0,  # ee x vel
                    0.0,  # ee y vel
                    0.0,  # ee z vel
                ]
                action = agent.act(np.array(obs), np.zeros(6 * num_frame_stack))
                print(action)
            return None

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
                num_frame_stack=num_frame_stack,
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

    """============== GOAL TRAJECTORY =============="""
    # import goal trajectory from pickle
    # shape = "heart"
    # shape = "infinity"
    # shape = "infinity_large"
    # shape = "real_traj_0"
    # shape = "real_traj_1"
    # shape = "real_traj_2"

    # shape = "slalom_fast_more_new"
    # shape = "ellipse_large_sparse_new"
    shape = "ellipse_large_sparse"
    goal_file_name = f"/home/bhoffman/Documents/MT FS24/active-learning-dynamics/goal_traj/{shape}_goal_trajectory.pkl"
    with open(goal_file_name, "rb") as f:
        goal_trajectory = pickle.load(f)

    """============== EXPERIMENT CONFIGS =============="""
    """===== SIM-MODEL ====="""
    sim_model_run_configs = {
        # shape goal policy testing v0_pre
        # "6e2362b6": (1000, 2),
        # "y7wdkxaz": (5000, 2),
        # "n6ljjfx7": (13000, 2),
        #
        # shape goal policy testing v0 (faster)
        # "fq3hmij4": (5000, 1),
        #
        # shape goal policy testing v1 (faster)
        # "kbj60y0e": (1000, 2),
        # "8i870q50": (2000, 2),
        # "ranz2azn": (5000, 2),
        #
        # shape_tracing_v2 (old data, higher costs)
        # "9rzpba8k": (2000, 1),
        # "gnm5vx5h": (5000, 1),
        # "5324y6mx": (10000, 1),
        #
        # shape_tracing_v5
        # "ik846ng1": (1000, 2),
        # "87vfy0e3": (4000, 2),
        # "arh30387": (13000, 2),
        #
        # shape_tracing_v7
        # "cdldhhl0": (1000, 2),
        # "yg9x81kd": (4000, 2),
        # "ucx2ixr8": (13000, 2),
        # #
        # "dw2zpnmi": (1000, 3),
        # "npk2vlno": (4000, 3),
        # "ejt9so0p": (13000, 3),
        #
        # shape_tracing_v9
        # "b1y879ru": (1000, 1),
        # "nozm88or": (4000, 1),
        # "vbvslmcc": (8000, 1),
        # #
        # "gaej86dr": (1000, 2),
        # "xk7rephs": (4000, 2),
        # "ijqk8mfs": (8000, 2),
        # #
        # "4pg8teow": (1000, 3),
        # "rvg6ncag": (4000, 3),
        # "tflc8k9d": (8000, 3),
        #
        # shape_tracing_new_data_v19
        # "fb4mnu4y": (2000, 137),
        # "yjkjjjrd": (4000, 137),
        # "56686ady": (8000, 137),
        # "itzsgfze": (10000, 137),
        #
        # shape_tracing_new_data_v21
        "idr3ss4q": (2000, 137),
        "3yj6vpmq": (4000, 137),
        "t0z2lsxm": (6000, 137),
        "31ltns47": (8000, 137),
        #
        "xlu85n5e": (2000, 332),
        "309xhwdy": (4000, 332),
        "142nuhye": (6000, 332),
        "olhim547": (8000, 332),
        #
        "134c27w9": (2000, 417),
        "ravy2yhk": (4000, 417),
        "xeg9x2ep": (6000, 417),
        "gyzh4tw4": (8000, 417),
    }

    exp_config_1 = {
        "run_id": list(sim_model_run_configs.keys()),
        "model_type": "sim-model",
    }

    """===== BNN-SIM-FSVGD ====="""
    bnn_sim_fsvgd_run_configs = {
        # shape goal policy testing vPre
        # "21xg6mx1": (13000, 2),
        #
        # shape goal policy testing v0_pre
        # "d8qdgakj": (1000, 2),
        # "9oitc1lv": (5000, 2),
        # "1sp7yxxx": (13000, 2),
        #
        # shape goal policy testing v0 (faster)
        # "rgjukl9h": (5000, 1),
        #
        # shape goal policy testing v1 (faster)
        # "7uyh9icp": (1000, 2),
        # "oz9hsvo9": (2000, 2),
        # "3qg74im1": (5000, 2),
        #
        # shape_tracing_v2 (old data, higher costs)
        # "4p1phlcy": (2000, 1),
        # "k1q6sdx9": (5000, 1),
        # "90n6i93n": (10000, 1),
        #
        # shape_tracing_v5
        # "rpp9kguo": (1000, 2),
        # "qio99wgo": (4000, 2),
        # "dmpc43qv": (13000, 2),
        #
        # shape_tracing_v7
        # "u2vx0t1a": (1000, 2),
        # "uwi0v5mb": (4000, 2),
        # "1ndhwrac": (13000, 2),
        # #
        # "f7w9z5xw": (1000, 3),
        # "bd4bgdzq": (4000, 3),
        # "xm5hn5bi": (13000, 3),
        #
        # shape_tracing_v9
        # "ir1k7sxx": (1000, 1),
        # "n7vsto4d": (4000, 1),
        # "jrnpnh2e": (8000, 1),
        # #
        # "rmjkshzy": (1000, 2),
        # "usd0rgvm": (4000, 2),
        # "n56lp7fu": (8000, 2),
        # #
        # "hlafbsjp": (1000, 3),
        # "u1e4pwkz": (4000, 3),
        # "qmvn3cn2": (8000, 3),
        #
        # shape_tracing_v12
        # "6jji3xj3": (8000, 1),
        #
        # # shape_tracing_new_data_v15
        # "qvfrf9ds": (4000, 2),
        # "gq2j9xcw": (8000, 2),
        # "uv3iy3sj": (10000, 2),
        #
        # shape_tracing_new_data_v19
        # "qzdn9qme": (2000, 137),
        # "sgybpoq7": (4000, 137),
        # "25lv8p8u": (8000, 137),
        # "rue7mplc": (10000, 137),
        #
        # shape_tracing_new_data_v21
        "k1rhmz06": (2000, 137),
        "n1timrkt": (4000, 137),
        "czwe5hrd": (6000, 137),
        "zt9nx5ju": (8000, 137),
        #
        "q7nx7sbc": (2000, 332),
        "q0jyfhqj": (4000, 332),
        "xi89rl6n": (6000, 332),
        "s1p62tqv": (8000, 332),
        #
        "s97f87i2": (2000, 417),
        "a6svc36w": (4000, 417),
        "svqfw6ks": (6000, 417),
        "dhg4et5i": (8000, 417),
    }

    exp_config_2 = {
        "run_id": list(bnn_sim_fsvgd_run_configs.keys()),
        "model_type": "bnn-sim-fsvgd",
    }

    """===== BNN-FSVGD ====="""

    bnn_fsvgd_run_configs = {
        # shape goal policy testing v0_pre
        # "c0vu1i05": (1000, 2),
        # "pgza5rk5": (5000, 2),
        # "sr1kau8u": (13000, 2),
        #
        # shape goal policy testing v0 (faster)
        # "o7e1vm9i": (5000, 1),
        #
        # shape goal policy testing v1 (faster)
        # "6xplabpt": (1000, 2),
        # "se9q4nvc": (2000, 2),
        # "bw6l5a2x": (5000, 2),
        #
        # shape_tracing_v2 (old data, higher costs)
        # "23z8w0oe": (2000, 1),
        # "xp3gtys7": (5000, 1),
        # "bpehqjkj": (10000, 1),
        #
        # shape_tracing_v5
        # "v27i34qa": (1000, 2),
        # "j1xstb5c": (4000, 2),
        # "6bkqbfxt": (13000, 2),
        #
        # shape_tracing_v7
        # "1i0iuf5x": (1000, 2),
        # "b21bvsky": (4000, 2),
        # "bf1w7p08": (13000, 2),
        # #
        # "1tekzvfj": (1000, 3),
        # "a196vcmq": (4000, 3),
        # "2v6lwsq8": (13000, 3),
        #
        # shape_tracing_v9
        # "84hxv3eb": (1000, 1),
        # "yqgtomfy": (4000, 1),
        # "q0naoo0u": (8000, 1),
        # #
        # "zkn8ocns": (1000, 2),
        # "mqv3ftob": (4000, 2),
        # "ur92ckf7": (8000, 2),
        # #
        # "qpltcy4p": (1000, 3),
        # "11xyv7vl": (4000, 3),
        # "96djmm04": (8000, 3),
        #
        # shape_tracing_new_data_v15
        # "e7utoxvw": (2000, 2),
        # "uv9xjeet": (3000, 2),
        # "38aum9oe": (4000, 2),
        # "028626c0": (8000, 2),
        # "mto6q10t": (10000, 2),
        #
        # shape_tracing_new_data_v19
        # "ck0qn337": (2000, 137),
        # "xkhjog12": (4000, 137),
        # "0k4e4ze6": (8000, 137),
        # "qxaz6cpq": (10000, 137),
        #
        # shape_tracing_new_data_v21
        "96w0vuyu": (2000, 137),
        "pf1dezuk": (4000, 137),
        "8w4margx": (6000, 137),
        "aylbzghf": (8000, 137),
        #
        "81k0o10n": (2000, 332),
        "3hre6c38": (4000, 332),
        "jsozls08": (6000, 332),
        "9t0abj8i": (8000, 332),
        #
        "xdfb6kl5": (2000, 417),
        "0837uj23": (4000, 417),
        "c642ykmp": (6000, 417),
        "ssv81jhk": (8000, 417),
    }

    exp_config_3 = {
        "run_id": list(bnn_fsvgd_run_configs.keys()),
        "model_type": "bnn-fsvgd",
    }

    """============== SETTINGS =============="""
    download_mode = False  # use to download policy from wandb
    num_episodes = 1
    # num_steps = 149
    num_steps = len(goal_trajectory) - 1
    cmd_freq = 15
    collect_data = True
    project_name = "shape_tracing_new_data_v21"
    # project_name = "ee_pos_testing"
    data_tag = project_name

    """============== SET ACTIVE CONFIG =============="""
    active_config_id = 2
    active_run_id = 11
    active_goal_id = shape
    num_frame_stack = 2
    action_scale = 1.0

    """============== BUILD SETTINGS =============="""
    exp_configs = [exp_config_1, exp_config_2, exp_config_3]
    run_configs = [
        sim_model_run_configs,
        bnn_sim_fsvgd_run_configs,
        bnn_fsvgd_run_configs,
    ]
    active_exp_config = exp_configs[active_config_id]
    active_run_config = run_configs[active_config_id]
    run_id = active_exp_config["run_id"][active_run_id]
    model_type = active_exp_config["model_type"]
    goal = goal_trajectory
    data_size = active_run_config[run_id][0]
    seed_id = active_run_config[run_id][1]
    data_tag = (
        f"{data_tag}_{model_type}_{data_size}_{seed_id}_{active_goal_id}_{run_id}_v"
    )

    start_experiment(
        download_mode=download_mode,
        num_episodes=num_episodes,
        num_steps=num_steps,
        cmd_freq=cmd_freq,
        collect_data=False if download_mode else collect_data,
        data_tag=data_tag,
        goal=goal,
        project_name=project_name,
        action_scale=action_scale,
        run_id=run_id,
        model_type=model_type,
        data_size=data_size,
        seed_id=seed_id,
        goal_id=active_goal_id,
        num_frame_stack=num_frame_stack,
    )
