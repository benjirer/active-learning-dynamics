import numpy as np
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from alrd.environment.spot.spot2d import Spot2DReward, SpotEnvironmentConfig
from alrd.environment.spot.simulate2d import Spot2DEnvSim
from alrd.environment.spot.spot2d import MIN_X, MIN_Y, MAX_X, MAX_Y
import matplotlib.patches as patches

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


class CustomSpot2DEnvSim(Spot2DEnvSim):
    def __init__(
        self, config: SpotEnvironmentConfig, freq, render_mode="human", **kwargs
    ):
        super().__init__(config, freq, **kwargs)
        self.render_mode = render_mode
        self.reward = Spot2DReward.create(
            goal_pos=np.array(
                [
                    self.config.start_x - 1,
                    self.config.start_y + 1,
                    self.config.start_angle,
                ]
            ),
            angle_coeff=0.1,
            action_coeff=self.action_cost,
            velocity_coeff=self.velocity_cost,
        )

        self.fig, self.ax = plt.subplots()  # Initialize the plot here
        self.agent_patch = patches.Circle((0, 0), 0.1, color="blue", label="Agent")
        self.goal_patch = patches.Circle((0, 0), 0.1, color="red", label="Goal")
        self.ax.add_patch(self.agent_patch)
        self.ax.add_patch(self.goal_patch)
        self.ax.set_xlim([MIN_X, MAX_X])
        self.ax.set_ylim([MIN_Y, MAX_Y])
        plt.legend()

    def step(self, action):
        self._update_state(action)
        obs = self._get_obs()
        reward = self.reward.predict(obs, action)
        self.last_reward = reward
        done = not self.is_in_bounds(self.state)
        info = {
            "dist": np.linalg.norm(obs[:2]).item(),
            "angle": np.abs(np.arctan2(obs[3], obs[2])).item(),
            "episode": {"l": self._episode_length, "r": self.last_reward},
        }
        truncated = not self.is_in_bounds(self.state)
        self._episode_length += 1
        return obs, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        self._episode_length = 0
        return super().reset(seed, options)

    def render(self, mode="human"):
        if mode != "human":
            raise NotImplementedError("Render mode not supported: {}".format(mode))

        self.agent_patch.center = (self.state[0], self.state[1])
        self.goal_patch.center = (self.goal_frame.x, self.goal_frame.y)
        plt.draw()
        plt.pause(0.001)


# Custom callback for wandb logging
class WandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(WandbCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        wandb.log(
            {
                "reward": self.locals["rewards"],
                "episode_length": (
                    self.locals["infos"][0].get("episode")["l"]
                    if "episode" in self.locals["infos"][0]
                    else 0
                ),
                "distance": self.locals["infos"][0].get("dist", 0),
                "angle": self.locals["infos"][0].get("angle", 0),
            }
        )
        return True


if __name__ == "__main__":
    # Initialize wandb
    wandb.init(project="spot2d-rl")

    # Create environment
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

    # Create vectorized environment for stable-baselines3
    env = make_vec_env(lambda: env, n_envs=1)

    # load model if exists (check if model exists before loading it, else learn)
    try:
        model = PPO.load("ppo_spot2d")
    except:
        # Define the PPO model
        model = PPO("MlpPolicy", env, verbose=1)

        # Train the model
        model.learn(total_timesteps=100000, callback=WandbCallback())

        # Save the model
        model.save("ppo_spot2d")

    # Evaluate the model
    obs = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
