import numpy as np


class SpotSimulator:
    def __init__(self, b: np.array = None):
        """
        Args:
            b (np.array): The fitted transition parameters.
        """

    def step(self, current_state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """
        Predict the next state given the current state and action.

        Args:
            current_state (np.ndarray): The current state of the system.
            action (np.ndarray): The action taken.

        State space:
            base_x              - frame: vision
            base_y              - frame: vision
            heading             - frame: vision
            base_vx             - frame: vision
            base_vy             - frame: vision
            base_vrot           - frame: vision
            ee_x                - frame: vision
            ee_y                - frame: vision
            ee_z                - frame: vision
            ee_vx               - frame: vision
            ee_vy               - frame: vision
            ee_vz               - frame: vision

        Action space:
            base_vx             - frame: body
            base_vy             - frame: body
            base_vrot           - frame: body
            ee_vx               - frame: body
            ee_vy               - frame: body
            ee_vz               - frame: body


        Returns:
            np.ndarray: The predicted next state.
        """
        current_state = np.array(current_state, dtype=np.float32)
        action = np.array(action, dtype=np.float32)

        theta_t = current_state[2]
        cos_t = np.cos(theta_t)
        sin_t = np.sin(theta_t)

        delta_t = 1.0 / 10.0

        next_state = np.zeros_like(current_state)

        self.b = np.zeros_like(action)

        """
        Base
        """

        # base actions in world frame
        base_vx_action_world = cos_t * action[0] - sin_t * action[1]
        base_vy_action_world = sin_t * action[0] + cos_t * action[1]
        vtheta_action_world = action[2]

        # add weighted velocities
        base_vx_action_world = (
            self.b[0] * current_state[3] + (1 - self.b[0]) * base_vx_action_world
        )
        base_vy_action_world = (
            self.b[1] * current_state[4] + (1 - self.b[1]) * base_vy_action_world
        )

        vtheta_action_world = (
            self.b[2] * current_state[5] + (1 - self.b[2]) * vtheta_action_world
        )

        # base velocities
        next_state[3] = base_vx_action_world
        next_state[4] = base_vy_action_world
        next_state[5] = vtheta_action_world

        # base positions
        next_state[0] = current_state[0] + delta_t * next_state[3]
        next_state[1] = current_state[1] + delta_t * next_state[4]
        next_state[2] = (current_state[2] + delta_t * next_state[5] + np.pi) % (
            2 * np.pi
        ) - np.pi

        """
        End effector
        """
        # end effector actions in world frame
        ee_vx_action_world = cos_t * action[3] - sin_t * action[4]
        ee_vy_action_world = sin_t * action[3] + cos_t * action[4]
        ee_vz_action_world = action[5]

        # add weighted velocities
        ee_vx_action_world = (
            self.b[3] * current_state[9] + (1 - self.b[3]) * ee_vx_action_world
        )
        ee_vy_action_world = (
            self.b[4] * current_state[10] + (1 - self.b[4]) * ee_vy_action_world
        )
        ee_vz_action_world = (
            self.b[5] * current_state[11] + (1 - self.b[5]) * ee_vz_action_world
        )

        # base angular velocity induces linear velocity at end effector
        base_xy_world = np.array([current_state[0], current_state[1]])
        ee_xy_world = np.array([current_state[6], current_state[7]])
        distance_ee_base = np.linalg.norm(ee_xy_world - base_xy_world)
        vel_ee_from_rot_magnitude = distance_ee_base * next_state[5]
        alpha = np.arctan2(
            ee_xy_world[0] - base_xy_world[0], ee_xy_world[1] - base_xy_world[1]
        )
        vel_ee_from_rot_x_world = -vel_ee_from_rot_magnitude * np.cos(alpha)
        vel_ee_from_rot_y_world = vel_ee_from_rot_magnitude * np.sin(alpha)

        # end effector velocities
        next_state[9] = ee_vx_action_world + next_state[3] + vel_ee_from_rot_x_world
        next_state[10] = ee_vy_action_world + next_state[4] + vel_ee_from_rot_y_world
        next_state[11] = ee_vz_action_world

        # end effector positions
        next_state[6] = current_state[6] + delta_t * next_state[9]
        next_state[7] = current_state[7] + delta_t * next_state[10]
        next_state[8] = max(current_state[8] + delta_t * next_state[11], 0.0)

        return next_state
