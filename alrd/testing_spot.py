import numpy as np
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from alrd.spot_gym_with_arm.envs.spot2d import Spot2DReward, SpotEnvironmentConfig
from alrd.spot_gym_with_arm.envs.simulate2d import Spot2DEnvSim
from alrd.spot_gym_with_arm.envs.spot2d import MIN_X, MIN_Y, MAX_X, MAX_Y
import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")


class CustomSpot2DEnvSim(Spot2DEnvSim):
    def __init__(
        self, config: SpotEnvironmentConfig, freq, render_mode="human", **kwargs
    ):
        super().__init__(config, freq, **kwargs)
        self.render_mode = render_mode
        self.fig, self.ax = plt.subplots()
        self.agent_patch = patches.Circle((0, 0), 0.1, color="blue", label="Agent")
        self.goal_patch = patches.Circle((0, 0), 0.1, color="red", label="Goal")
        self.ax.add_patch(self.agent_patch)
        self.ax.add_patch(self.goal_patch)
        self.ax.set_xlim([MIN_X, MAX_X])
        self.ax.set_ylim([MIN_Y, MAX_Y])
        plt.legend()

    def _update_state(self, action):
        super()._update_state(action)  # Call the updated dynamics from the parent class

    def render(self, mode="human"):
        if mode != "human":
            raise NotImplementedError("Render mode not supported: {}".format(mode))
        # Update the robot's base position
        self.agent_patch.center = (self.state[0], self.state[1])
        plt.draw()
        plt.pause(0.001)


class WandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(WandbCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        wandb.log(
            {
                "reward": self.locals["rewards"],
                "episode_length": self.locals["infos"][0]
                .get("episode", {})
                .get("l", 0),
                "distance": self.locals["infos"][0].get("dist", 0),
                "angle": self.locals["infos"][0].get("angle", 0),
            }
        )
        return True


if __name__ == "__main__":
    wandb.init(project="spot2d-rl")

    config = SpotEnvironmentConfig(
        hostname="localhost",
        start_x=0,
        start_y=0,
        start_angle=0,
        min_x=-5,
        max_x=5,
        min_y=-5,
        max_y=5,
    )
    env = CustomSpot2DEnvSim(config, freq=10, action_cost=0.1, velocity_cost=0.1)
    env = Monitor(env)
    env = make_vec_env(lambda: env, n_envs=1)

    try:
        model = PPO.load("ppo_spot2d")
    except FileNotFoundError:
        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=100000, callback=WandbCallback())
        model.save("ppo_spot2d")

    obs = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
