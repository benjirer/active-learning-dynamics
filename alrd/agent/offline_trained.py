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

    def act(self, obs: np.ndarray) -> np.ndarray:
        # add goal to obs
        goal = np.array([1.0, -0.04, 0.5])
        obs_goal_distance = np.linalg.norm(obs[7:10] - goal)
        # print(f"obs_goal_distance: {obs_goal_distance}")
        obs = np.concatenate((obs, goal), axis=-1)
        print(f"obs: {obs}")

        return np.array(self.policy(obs))

    def description(self):
        return """Using offline learned policy"""
