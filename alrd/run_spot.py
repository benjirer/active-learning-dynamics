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
        action = np.clip(action, -1.0, 1.0)

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
    goal_id: int = 0,
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
        experiment_id = "test" + time.strftime("%Y%m%d-%H%M%S") + "_" + data_tag
        session_dir = (
            "/home/bhoffman/Documents/MT FS24/active-learning-dynamics/collected_data/"
            + experiment_id
        )
        experiment_settings = [
            "num_episodes: {}".format(num_episodes),
            "num_steps: {}".format(num_steps),
            "cmd_freq: {}".format(cmd_freq),
            "goal: {}".format(goal),
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
                goal_pos=goal,
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

    """============== GOALs =============="""
    goal_1 = np.array([1.2, -0.2, 0.8])
    goal_2 = np.array([1.4, 0.2, 0.4])
    goal_3 = np.array([1.6, 0.0, 0.2])
    goal_4 = np.array([1.8, -0.4, 0.4])
    goals = [goal_1, goal_2, goal_3, goal_4]

    """============== EXPERIMENT CONFIGS =============="""
    """===== SIM-MODEL ====="""
    sim_model_run_configs = {
        # # v2
        # "lnc8z8pp": (800, 1),
        # "qrgm252s": (2500, 1),
        # "rg9vq53y": (5400, 1),
        # "e8bdn23s": (800, 2),
        # "1yurd56p": (2500, 2),
        # "zd0o0jx8": (5400, 2),
        # "ybtc7l88": (800, 3),
        # "kt9vhd17": (2500, 3),
        # "8sg8lbqq": (5400, 3),
        # #
        # # v4_partial
        # "hck2b2u0": (800, 3),
        # "chn3iu4r": (2000, 3),
        # "yjgeqtmy": (5000, 3),
        # #
        # # v_action_stack_1
        # "55zv3ri3": (800, 1),
        # "31xn9mox": (2000, 1),
        # "lcsku3pl": (5000, 1),
        # #
        # # v_action_stack_2
        # "6fn7dsov": (800, 1),
        # "3zrabx81": (2000, 1),
        # "zhke5a7p": (5000, 1),
        # #
        # v_action_stack_5
        "zlbzwr4q": (5000, 1),  # 0.05, 0.3 -- slow
        "fqiti1id": (5000, 1),  # 0.05, 0.4 -- slow not accurate
        "pr52xbny": (5000, 1),  # 0.02, 0.3 -- shaky
        "lrvplba1": (5000, 1),  # 0.02, 0.4 -- okay but slow
        "v2ifxv4t": (
            5000,
            1,
        ),  # 0.05, 0.35, new (weights changed from 2.0 to 1.5 and 0.5, bounds changed to 0.05) -- very shakey
    }

    exp_config_1 = {
        "run_id": list(sim_model_run_configs.keys()),
        "model_type": "sim-model",
    }

    """===== BNN-SIM-FSVGD ====="""
    bnn_sim_fsvgd_run_configs = {
        # # v2
        # "pjf0qaum": (800, 1),
        # "flxhy0yy": (2500, 1),
        # "rkrv365l": (5400, 1),
        # "tl210l8f": (800, 2),
        # "wj6jdjh5": (2500, 2),
        # "p5wgr7rm": (5400, 2),
        # "wudh8u7u": (800, 3),
        # "c4o3eb4k": (2500, 3),
        # "rfl97xto": (5400, 3),
        # #
        # # v4_partial
        # "sq0k6akn": (800, 3),
        # "k5kepn4q": (2000, 3),
        # "891g63gq": (5000, 3),
        # #
        # # v_action_stack_1
        # "v64vrzpw": (800, 1),
        # "bggled25": (2000, 1),
        # "xcmnhhfq": (5000, 1),
        # #
        # v_action_stack_1.2
        # "zpdk97km": (5000, 1),
        # "9g7whijl": (5000, 1),
        # "8n60b1jt": (5000, 1),
        # #
        # # v_action_stack_2
        # "0vaw5ltx": (800, 1),
        # "zo86kfgj": (2000, 1),
        # "waklhuyc": (5000, 1),
        # #
        # # v_action_stack_3
        # "hcmip3gc": (5000, 1),
        # "dj131cg6": (5000, 1),
        # #
        # # v_action_stack_5
        # "p7kd475o": (
        #     5000,
        #     1,
        # ),  # 0.05, 0.3 -- pretty good, not super slow and quite accurate
        # "lhfwewuf": (5000, 1),  # 0.05, 0.4 -- also good but slightly slower maybe
        # "u9i1vo0n": (5000, 1),  # 0.02, 0.3 -- not as accurate
        # "i37so3f7": (5000, 1),  # 0.02, 0.4 -- too slow
        # "gboojd12": (
        #     5000,
        #     1,
        # ),  # 0.05, 0.35, new (weights changed from 2.0 to 1.5 and 0.5, bounds changed to 0.05) -- unstable
        # #
        # v_action_stack_6
        "93jklpn7": (
            5000,
            1,
        ),  # 0.05, 0.3, (weighst back to 0.05 and 0.3, bound back to 0.1, new margin at 5.0) -- very nice
    }

    exp_config_2 = {
        "run_id": list(bnn_sim_fsvgd_run_configs.keys()),
        "model_type": "bnn-sim-fsvgd",
    }

    """===== BNN-FSVGD ====="""

    bnn_fsvgd_run_configs = {
        # # v2
        # "n2okbiym": (800, 1),
        # "0awx6i93": (2500, 1),
        # "jr0ybogk": (5400, 1),
        # "6mzchm1o": (800, 2),
        # "gdooo4u2": (2500, 2),
        # "99rq9ysq": (5400, 2),
        # "kagasm3e": (800, 3),
        # "fhe93mwq": (2500, 3),
        # "160s663n": (5400, 3),
        # #
        # # v4_partial
        # "hjbw63y8": (800, 3),
        # "0dslu87b": (2000, 3),
        # "hq7f5yzu": (5000, 3),
        # #
        # # v_action_stack_1
        # "hjbw63y8": (800, 1),
        # "0dslu87b": (2000, 1),
        # "hq7f5yzu": (5000, 1),
        # #
        # # v_action_stack_2
        # "ccx8cxmw": (800, 1),
        # "h6ci16iv": (2000, 1),
        # "06hno299": (5000, 1),
        # #
        # v_action_stack_5
        "biteqxmk": (5000, 1),  # 0.05, 0.3 -- not bad tbh
        "ji4pagqe": (5000, 1),  # 0.05, 0.4 -- too slow and innacurate
        "btaozmx5": (5000, 1),  # 0.02, 0.3 -- ok
        "gwvr88cm": (5000, 1),  # 0.02, 0.4 -- a bit slow
        "0fjeh9h1": (
            5000,
            1,
        ),  # 0.05, 0.35, new (weights changed from 2.0 to 1.5 and 0.5, bounds changed to 0.05) --
    }

    exp_config_3 = {
        "run_id": list(bnn_fsvgd_run_configs.keys()),
        "model_type": "bnn-fsvgd",
    }

    """============== SETTINGS =============="""
    download_mode = False  # use to download policy from wandb
    num_episodes = 1
    num_steps = 50
    cmd_freq = 10
    collect_data = True
    project_name = "action_stack_testing_v6"
    data_tag = project_name

    """============== SET ACTIVE CONFIG =============="""
    active_config_id = 1
    active_run_id = 0
    active_goal_id = 3
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
    goal = goals[active_goal_id]
    data_size = active_run_config[run_id][0]
    seed_id = active_run_config[run_id][1]
    data_tag = f"{data_tag}_{run_id}_{model_type}_{data_size}_{seed_id}_{active_goal_id}__{action_scale}"

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
