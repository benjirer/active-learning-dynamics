# imports
import time
import logging
import yaml
from alrd.agent import Agent
from alrd.agent.keyboard import KeyboardAgent
from alrd.agent.xbox import SpotXbox2D
from alrd.agent.spacebox import SpotSpaceBox
from alrd.agent.randomxbox import SpotRandomXbox
from alrd.spot_gym_with_arm import Spot2DEnv
from alrd.spot_gym_with_arm.envs.spotgym import SpotGym
from gym.wrappers.rescale_action import RescaleAction

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)


def run(agent: Agent, env: SpotGym, num_steps: int = 1000, cmd_freq: float = 10):
    started = False
    step = 0

    while True:
        print(step)

        # if not started, reset the environment
        if not started:
            print(agent.description())
            count = 0
            obs, info = env.reset()
            if obs is None:
                return
            agent.reset()
            started = True

        # get action from agent
        action = agent.act(obs)
        print("Action taken:", action)

        # step the environment
        if action is not None:
            next_obs, reward, terminated, truncated, info = env.step(action)
            if next_obs is not None:
                count += 1
                step += 1

        # check if episode is terminated
        if action is None or terminated or truncated:
            started = False
            if count > 0:
                print("Terminated %s. Truncated %s" % (terminated, truncated))
                return
        else:
            obs = next_obs
    env.stop_robot()


def start():

    # import config
    config = yaml.load(
        open(
            "/home/bhoffman/Documents/MT FS24/active-learning-dynamics/config/test.yaml",
            "r",
        ),
        Loader=yaml.Loader,
    )

    cmd_freq = 10

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
    agent = SpotRandomXbox(base_speed=1.0, base_angular=1.0, arm_speed=1.0)

    # start env
    env.start()
    print("env", type(env), "obs", env.observation_space.shape)
    env = RescaleAction(env, min_action=-1, max_action=1)

    # run
    try:
        run(agent, env, num_steps=200, cmd_freq=cmd_freq)
    except KeyboardInterrupt:
        print("Exiting due to keyboard interrupt")
        env.stop_robot()
        env.close()
    finally:
        print("Exiting due to finish")
        env.stop_robot()
        env.close()


if __name__ == "__main__":

    start()
