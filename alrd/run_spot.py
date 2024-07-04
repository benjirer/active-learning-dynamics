# imports
import time
import logging
import yaml
import pickle
import os
from alrd.agent import Agent
from alrd.agent.keyboard import KeyboardAgent
from alrd.agent.xbox import SpotXbox2D
from alrd.agent.spacebox import SpotSpaceBox
from alrd.agent.randomxbox import SpotRandomXbox
from alrd.spot_gym import Spot2DEnv
from alrd.spot_gym.envs.spotgym import SpotGym
from gym.wrappers.rescale_action import RescaleAction

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)


# data collection classes
# class to store all data vectors (states and transitions)
class DataBuffer:
    def __init__(self, states, transitions):
        self.states = states
        self.transitions = transitions


# class to store individual transitions
class transition:
    def __init__(
        self, step, obs, action, given_commands, reward, next_obs, terminated, truncated
    ):
        self.step = step
        self.obs = obs
        self.action = action
        self.given_commands = given_commands
        self.reward = reward
        self.next_obs = next_obs
        self.terminated = terminated
        self.truncated = truncated


def run(
    agent: Agent,
    env: SpotGym,
    num_steps: int = 1000,
    cmd_freq: float = 10,
    collect_data: bool = False,
    data_buffer: DataBuffer = None,
    experiment_dir: str | None = None,
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
                data_buffer.states.append(obs)
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
                data_buffer.states.append(next_obs)
                data_buffer.transitions.append(
                    transition(
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


def start_experiment():

    # experiment settings
    num_episodes = 1
    num_steps = 200
    cmd_freq = 10
    collect_data = True

    # import real world config
    config = yaml.load(
        open(
            "/home/bhoffman/Documents/MT FS24/active-learning-dynamics/config/test.yaml",
            "r",
        ),
        Loader=yaml.Loader,
    )

    # set up data collection directory
    experiment_dir = None
    data_buffer = None
    if collect_data:
        data_buffer = DataBuffer([], [])
        experiment_id = "test" + time.strftime("%Y%m%d-%H%M%S")
        experiment_dir = (
            "/home/bhoffman/Documents/MT FS24/active-learning-dynamics/collected_data/"
            + experiment_id
        )
        experiment_settings = [
            "num_episodes: {}".format(num_episodes),
            "num_steps: {}".format(num_steps),
            "cmd_freq: {}".format(cmd_freq),
        ]
        os.makedirs(experiment_dir, exist_ok=True)
        settings_path = os.path.join(experiment_dir, "experiment_settings.pickle")
        open(
            settings_path,
            "wb",
        ).write(pickle.dumps(experiment_settings))

    episode = 0
    while episode < num_episodes:

        if collect_data:
            transitions_path = os.path.join(experiment_dir, "transitions.pickle")
            states_path = os.path.join(experiment_dir, "states.pickle")

        # create env
        env = Spot2DEnv(
            config,
            cmd_freq=cmd_freq,
            log_str=False,
        )

        # create agent
        # agent = KeyboardAgent(xy_speed=1, a_speed=1)
        # agent = SpotXbox2D(base_speed=1, base_angular=1, arm_speed=1)
        # agent = SpotSpaceBox(base_speed=1.0, base_angular=1.0, arm_speed=1.0)
        agent = SpotRandomXbox(
            base_speed=1.0,
            base_angular=1.0,
            arm_speed=1.0,
            cmd_freq=cmd_freq,
            steps=num_steps,
        )

        # start env
        env.start()
        logger.info("env: %s. obs: %s" % (type(env), env.observation_space.shape))
        env = RescaleAction(env, min_action=-1, max_action=1)

        # run
        try:
            run(
                agent,
                env,
                num_steps=num_steps,
                cmd_freq=cmd_freq,
                collect_data=collect_data,
                data_buffer=data_buffer,
                experiment_dir=experiment_dir,
            )
        except KeyboardInterrupt:
            logger.info("Exiting due to keyboard interrupt")
            env.stop_robot()
            env.close()
        except Exception as e:
            logger.error("Exiting due to exception: %s" % e)
            env.stop_robot()
            env.close()
        finally:
            logger.info("Exiting due to finish")
            env.stop_robot()
            env.close()

        if collect_data:
            open(
                transitions_path,
                "wb",
            ).write(pickle.dumps(data_buffer.transitions))
            open(
                states_path,
                "wb",
            ).write(pickle.dumps(data_buffer.states))

        episode += 1


if __name__ == "__main__":
    start_experiment()
