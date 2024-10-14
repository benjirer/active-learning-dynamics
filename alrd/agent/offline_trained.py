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
        self.policy = self.rl_from_offline_data.prepare_policy_bnn(params=policy_params)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.goal = goal
        self.reached_counter = 0
        self.reached = False

        self.reward = SpotEnvReward(
            encode_angle=reward_config["encode_angle"],
            ctrl_cost_weight=reward_config["ctrl_cost_weight"],
            margin_factor=reward_config["margin_factor"],
        )

    def act(self, obs: np.ndarray) -> np.ndarray:
        # add goal to obs
        goal = self.goal
        obs_goal_distance = np.linalg.norm(obs[7:10] - goal)

        # print(f"obs_goal_distance: {obs_goal_distance}")
        obs = np.concatenate((obs, goal), axis=-1)
        # print(f"obs: {obs}")
        action = self.policy(obs)

        print(f"DISTANCE TO GOAL: {obs_goal_distance}")

        force_stop = False
        if force_stop:
            if obs_goal_distance < 0.1 or self.reached:
                if not self.reached:
                    print("GOAL TEMP REACHED")
                self.reached_counter += 1
                if self.reached_counter > 2:
                    self.reached = True
                if self.reached:
                    action = np.zeros(self.action_dim)
                    print("GOAL FINALLY REACHED")
            else:
                self.reached_counter = 0
        return np.array(action)

    def get_reward(
        self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray
    ) -> float:
        # add goal to obs
        goal = self.goal
        obs = np.concatenate((obs, goal), axis=-1)
        next_obs = np.concatenate((next_obs, goal), axis=-1)
        return self.reward(jnp.array(obs), jnp.array(action), jnp.array(next_obs))

    def description(self):
        return """Using offline learned policy"""
