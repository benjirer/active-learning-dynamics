# imports
import time
import logging
import yaml
import pickle
import os
import numpy as np

# agents
from alrd.agent.absagent import Agent
from alrd.agent.keyboard import KeyboardAgent
from alrd.agent.xbox import SpotXbox2D
from alrd.agent.spacebox import SpotSpaceBox
from alrd.agent.randomxbox import SpotRandomXbox

# environments
from alrd.spot_gym.envs.spot_eevel import SpotEEVelEnv
from alrd.spot_gym.envs.spot_jointpos import SpotJointPosEnv
from alrd.spot_gym.envs.spotgym import SpotGym

# additionals
from gym.wrappers.rescale_action import RescaleAction
from alrd.spot_gym.model.robot_state import SpotState

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)


# data collection classes
# class to store individual transitions
class TransitionData:
    def __init__(self, step, obs, action, cmd, reward, next_obs, terminated, truncated):
        self.step = step
        self.obs = obs
        self.action = action
        self.cmd = cmd
        self.reward = reward
        self.next_obs = next_obs
        self.terminated = terminated
        self.truncated = truncated


# class to store individual states
class StateData:
    def __init__(self, step: int, last_state: SpotState, next_state: SpotState):
        self.step = step
        self.last_state = last_state
        self.next_state = next_state


# class to store all data vectors for an episode
class DataBuffer:
    def __init__(
        self,
        states: list[StateData] = [],
        observations: list[np.ndarray] = [],
        transitions: list[TransitionData] = [],
    ):
        self.states = states
        self.observations = observations
        self.transitions = transitions


# class to store all DataBuffers for a session
class SessionBuffer:
    def __init__(self, data_buffers: list[DataBuffer] = []):
        self.data_buffers = data_buffers


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

    while step < num_steps:
        logger.info("Step %s" % step)
        # if not started, reset the environment
        if not started:
            logger.info("Agent description: %s" % agent.description())
            count = 0
            obs, info = env.reset()
            if collect_data:
                data_buffer.observations.append(obs)
            if obs is None:
                return
            agent.reset()
            started = True

        # get action from agent
        action = agent.act(obs)
        logger.info("Action %s" % action)

        # step the environment
        if action is not None:
            next_obs, reward, terminated, truncated, info = env.step(action)
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
                    )
                )
                data_buffer.states.append(
                    StateData(
                        step,
                        info["last_state"],
                        info["next_state"],
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
    num_steps = 1000
    cmd_freq = 10
    collect_data = True
    random_seed = 0

    # random seeds for noise sampling
    sampling_seeds = np.random.default_rng(seed=random_seed).integers(
        0, 2**32, size=num_episodes
    )

    # import real world config
    config = yaml.load(
        open(
            "/home/bhoffman/Documents/MT FS24/active-learning-dynamics/config/test.yaml",
            "r",
        ),
        Loader=yaml.Loader,
    )

    # set up data collection directory
    session_dir = None
    session_buffer = None

    if collect_data:
        session_buffer = SessionBuffer()
        experiment_id = "test" + time.strftime("%Y%m%d-%H%M%S")
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

        # note: make sure env and agent are compatible
        # create env
        env = SpotEEVelEnv(
            config,
            cmd_freq=cmd_freq,
            log_str=False,
        )

        # create agent
        # agent = KeyboardAgent(xy_speed=1, a_speed=1)
        agent = SpotXbox2D(base_speed=1, base_angular=1, arm_speed=0.5)
        # agent = SpotSpaceBox(base_speed=1.0, base_angular=1.0, ee_speed=0.5)
        # agent = SpotRandomXbox(
        #     base_speed=1.0,
        #     base_angular=1.0,
        #     arm_speed=1.0,
        #     cmd_freq=cmd_freq,
        #     steps=num_steps,
        #     random_seed=sampling_seeds[episode],
        # )

        # start env
        env.start()
        logger.info("env: %s. obs: %s" % (type(env), env.observation_space.shape))
        env = RescaleAction(env, min_action=-1, max_action=1)

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
