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
        self.reached = False

    def act(self, obs: np.ndarray) -> np.ndarray:
        # add goal to obs
        goal = np.array([1.2, -0.8, 0.9])
        obs_goal_distance = np.linalg.norm(obs[7:10] - goal)

        # print(f"obs_goal_distance: {obs_goal_distance}")
        obs = np.concatenate((obs, goal), axis=-1)
        print(f"obs: {obs}")
        action = self.policy(obs)

        print(obs_goal_distance)

        if obs_goal_distance < 0.1 or self.reached:
            self.reached = True
            action = np.zeros(self.action_dim)
        return np.array(action)

    def description(self):
        return """Using offline learned policy"""
