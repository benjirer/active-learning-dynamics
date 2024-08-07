import numpy as np


class SpotSimulator:
    def __init__(self, b: dict = None, method: str = "normal"):
        """
        Args:
            b (dict): The fitted transition parameters.
        """
        if b is not None:
            self.b = b
        else:
            self.b = {
                "base_vx_scale": 1.0,
                "base_vy_scale": 1.0,
                "vtheta_scale": 1.0,
                "ee_vx_scale": 1.0,
                "ee_vy_scale": 1.0,
                "ee_vz_scale": 1.0,
                "delta_t": 0.1,
            }

        self.method = method

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

        base_vx_scale = self.b.get("base_vx_scale", 1.0)
        base_vy_scale = self.b.get("base_vy_scale", 1.0)
        vtheta_scale = self.b.get("vtheta_scale", 1.0)
        ee_vx_scale = self.b.get("ee_vx_scale", 1.0)
        ee_vy_scale = self.b.get("ee_vy_scale", 1.0)
        ee_vz_scale = self.b.get("ee_vz_scale", 1.0)
        delta_t = self.b.get("delta_t", 0.1)

        # A = np.array(
        #     [
        #         [1, 0, 0, delta_t / 2, 0, 0, 0, 0, 0, 0, 0, 0],
        #         [0, 1, 0, 0, delta_t / 2, 0, 0, 0, 0, 0, 0, 0],
        #         [0, 0, 1, 0, 0, delta_t / 2, 0, 0, 0, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 0, 1, 0, 0, delta_t / 2, 0, 0],
        #         [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, delta_t / 2, 0],
        #         [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, delta_t / 2],
        #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #     ]
        # )

        # B = np.array(
        #     [
        #         [
        #             delta_t / 2 * np.cos(theta_t) * base_vx_scale,
        #             -delta_t / 2 * np.sin(theta_t) * base_vy_scale,
        #             0,
        #             0,
        #             0,
        #             0,
        #         ],
        #         [
        #             delta_t / 2 * np.sin(theta_t) * base_vx_scale,
        #             delta_t / 2 * np.cos(theta_t) * base_vy_scale,
        #             0,
        #             0,
        #             0,
        #             0,
        #         ],
        #         [0, 0, delta_t / 2 * vtheta_scale, 0, 0, 0],
        #         [
        #             np.cos(theta_t) * base_vx_scale,
        #             -np.sin(theta_t) * base_vy_scale,
        #             0,
        #             0,
        #             0,
        #             0,
        #         ],
        #         [
        #             np.sin(theta_t) * base_vx_scale,
        #             np.cos(theta_t) * base_vy_scale,
        #             0,
        #             0,
        #             0,
        #             0,
        #         ],
        #         [0, 0, vtheta_scale, 0, 0, 0],
        #         [
        #             0,
        #             0,
        #             0,
        #             delta_t / 2 * np.cos(theta_t) * ee_vx_scale,
        #             -delta_t / 2 * np.sin(theta_t) * ee_vy_scale,
        #             0,
        #         ],
        #         [
        #             0,
        #             0,
        #             0,
        #             delta_t / 2 * np.sin(theta_t) * ee_vx_scale,
        #             delta_t / 2 * np.cos(theta_t) * ee_vy_scale,
        #             0,
        #         ],
        #         [0, 0, 0, 0, 0, delta_t / 2 * ee_vz_scale],
        #         [
        #             0,
        #             0,
        #             0,
        #             np.cos(theta_t) * ee_vx_scale,
        #             -np.sin(theta_t) * ee_vy_scale,
        #             0,
        #         ],
        #         [
        #             0,
        #             0,
        #             0,
        #             np.sin(theta_t) * ee_vx_scale,
        #             np.cos(theta_t) * ee_vy_scale,
        #             0,
        #         ],
        #         [0, 0, 0, 0, 0, ee_vz_scale],
        #     ]
        # )

        # Compute the next state
        # next_state = A @ current_state + B @ action

        # average velocities from previous and input
        if self.method == "averaged":
            next_state = np.zeros_like(current_state)
            next_state[0] = current_state[0] + delta_t / 2 * (
                current_state[3]
                + base_vx_scale * np.cos(theta_t) * action[0]
                - base_vy_scale * np.sin(theta_t) * action[1]
            )
            next_state[1] = current_state[1] + delta_t / 2 * (
                current_state[4]
                + base_vx_scale * np.sin(theta_t) * action[0]
                + base_vy_scale * np.cos(theta_t) * action[1]
            )
            next_state[2] = current_state[2] + delta_t / 2 * (
                current_state[5] + vtheta_scale * action[2]
            )
            # normalize to be within -pi to pi
            next_state[2] = (next_state[2] + np.pi) % (2 * np.pi) - np.pi

            next_state[3] = (
                base_vx_scale * np.cos(theta_t) * action[0]
                - base_vy_scale * np.sin(theta_t) * action[1]
            )
            next_state[4] = (
                base_vx_scale * np.sin(theta_t) * action[0]
                + base_vy_scale * np.cos(theta_t) * action[1]
            )
            next_state[5] = vtheta_scale * action[2]

            next_state[6] = current_state[6] + delta_t / 2 * (
                current_state[9]
                + ee_vx_scale * np.cos(theta_t) * action[3]
                - ee_vy_scale * np.sin(theta_t) * action[4]
            )
            next_state[7] = current_state[7] + delta_t / 2 * (
                current_state[10]
                + ee_vx_scale * np.sin(theta_t) * action[3]
                + ee_vy_scale * np.cos(theta_t) * action[4]
            )
            next_state[8] = current_state[8] + delta_t / 2 * (
                current_state[11] + ee_vz_scale * action[5]
            )

            next_state[9] = (
                ee_vx_scale * np.cos(theta_t) * action[3]
                - ee_vy_scale * np.sin(theta_t) * action[4]
            )
            next_state[10] = (
                ee_vx_scale * np.sin(theta_t) * action[3]
                + ee_vy_scale * np.cos(theta_t) * action[4]
            )
            next_state[11] = ee_vz_scale * action[5]

        # use input velocity only
        elif self.method == "normal":
            next_state = np.zeros_like(current_state)

            # BASE
            # base x position
            next_state[0] = current_state[0] + delta_t * (
                base_vx_scale * np.cos(theta_t) * action[0]
                - base_vy_scale * np.sin(theta_t) * action[1]
            )

            # base y position
            next_state[1] = current_state[1] + delta_t * (
                base_vx_scale * np.sin(theta_t) * action[0]
                + base_vy_scale * np.cos(theta_t) * action[1]
            )

            # base heading angle (yaw)
            next_state[2] = current_state[2] + delta_t * vtheta_scale * action[2]
            # normalize to be within -pi to pi
            next_state[2] = (next_state[2] + np.pi) % (2 * np.pi) - np.pi

            # base x velocity
            next_state[3] = (
                base_vx_scale * np.cos(theta_t) * action[0]
                - base_vy_scale * np.sin(theta_t) * action[1]
            )

            # base y velocity
            next_state[4] = (
                base_vx_scale * np.sin(theta_t) * action[0]
                + base_vy_scale * np.cos(theta_t) * action[1]
            )

            # base heading (yaw) angular velocity
            next_state[5] = vtheta_scale * action[2]

            # END EFFECTOR
            # end effector x position
            next_state[6] = (
                current_state[6]
                + delta_t
                * (
                    ee_vx_scale * np.cos(theta_t) * action[3]
                    - ee_vy_scale * np.sin(theta_t) * action[4]
                )
                + delta_t * next_state[3]
            )

            # end effector y position
            next_state[7] = (
                current_state[7]
                + delta_t
                * (
                    ee_vx_scale * np.sin(theta_t) * action[3]
                    + ee_vy_scale * np.cos(theta_t) * action[4]
                )
                + delta_t * next_state[4]
            )

            # end effector z position
            next_state[8] = current_state[8] + delta_t * ee_vz_scale * action[5]

            # velocity components of body rotation
            body_xyz_w = np.array([current_state[0], current_state[1], 0])
            ee_xyz_w = np.array([current_state[6], current_state[7], 0])
            r_xyz_w = ee_xyz_w - body_xyz_w
            rot_xyz_w = np.array([0, 0, action[2]])
            vel_ee_from_rot = np.cross(rot_xyz_w, r_xyz_w)
            vel_ee_from_rot_x = vel_ee_from_rot[0]
            vel_ee_from_rot_y = vel_ee_from_rot[1]

            # end effector x velocity
            next_state[9] = (
                (
                    ee_vx_scale * np.cos(theta_t) * action[3]
                    - ee_vy_scale * np.sin(theta_t) * action[4]
                )
                + next_state[3]
                + vel_ee_from_rot_x
            )

            # end effector y velocity
            next_state[10] = (
                (
                    ee_vx_scale * np.sin(theta_t) * action[3]
                    + ee_vy_scale * np.cos(theta_t) * action[4]
                )
                + next_state[4]
                + vel_ee_from_rot_y
            )

            # end effector z velocity
            next_state[11] = ee_vz_scale * action[5]

        else:
            raise ValueError("Invalid method. Must be either 'normal' or 'averaged'.")

        return next_state
