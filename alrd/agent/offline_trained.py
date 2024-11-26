from alrd.agent.absagent import Agent, AgentReset
from sim_transfer.rl.spot_rl_on_offline_data import RLFromOfflineData
from sim_transfer.sims.envs import SpotEnvReward
import jax.numpy as jnp
import numpy as np


class OfflineTrainedAgent(AgentReset):
    def __init__(
        self,
        policy_params,
        reward_config: dict,
        state_dim: int,
        action_dim: int,
        goal_dim: int,
        goal: np.ndarray,
        SAC_KWARGS,
    ) -> None:
        self.rl_from_offline_data = RLFromOfflineData(
            sac_kwargs=SAC_KWARGS,
            x_train=jnp.zeros((10, state_dim + goal_dim + action_dim)),
            y_train=jnp.zeros((10, state_dim)),
            x_test=jnp.zeros((10, state_dim + goal_dim + action_dim)),
            y_test=jnp.zeros((10, state_dim)),
            spot_reward_kwargs=reward_config,
        )
        self.policy = self.rl_from_offline_data.prepare_policy(params=policy_params)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.goal = goal
        self.goal_idx = 0

        self.reward = SpotEnvReward(
            encode_angle=reward_config["encode_angle"],
            ctrl_cost_weight=reward_config["ctrl_cost_weight"],
            margin_factor=reward_config["margin_factor"],
        )

    def act(self, obs: np.ndarray, action_buffer: np.ndarray) -> np.ndarray:
        # add goal to obs
        goal = self.goal[self.goal_idx]
        # set x compnent to x = 1.3
        goal[0] = 1.4
        obs_goal_distance = np.linalg.norm(obs[7:10] - goal)

        # print(f"obs_goal_distance: {obs_goal_distance}")
        obs = np.concatenate((obs, goal), axis=-1)
        # print(f"obs: {obs}")

        # add action buffer to obs
        obs = np.concatenate((obs, action_buffer), axis=-1)

        action = self.policy(obs)

        print(f"GOAL: {goal}")
        print(f"DISTANCE TO GOAL: {obs_goal_distance}")
        print(f"ACTION: {action}")

        self.goal_idx += 1
        return np.array(action)

    def get_reward(
        self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray
    ) -> float:
        # add goal to obs
        goal = self.goal[self.goal_idx]
        obs = np.concatenate((obs, goal), axis=-1)
        next_obs = np.concatenate((next_obs, goal), axis=-1)
        return self.reward(jnp.array(obs), jnp.array(action), jnp.array(next_obs))

    def description(self):
        return """Using offline learned policy"""
