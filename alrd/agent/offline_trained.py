from alrd.agent.absagent import Agent, AgentReset
from sim_transfer.rl.spot_sim_rl_on_offline_data import RLFromOfflineData
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
        self.reached_counter = 0
        self.reached = False

    def act(self, obs: np.ndarray) -> np.ndarray:
        # add goal to obs
        goal = self.goal
        obs_goal_distance = np.linalg.norm(obs[7:10] - goal)

        # print(f"obs_goal_distance: {obs_goal_distance}")
        obs = np.concatenate((obs, goal), axis=-1)
        # print(f"obs: {obs}")
        action = self.policy(obs)

        print(f"DISTANCE TO GOAL: {obs_goal_distance}")

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

    def description(self):
        return """Using offline learned policy"""
