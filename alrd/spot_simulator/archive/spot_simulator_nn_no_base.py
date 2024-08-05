import numpy as np
import tensorflow as tf
import pickle

from alrd.spot_gym.utils.spot_arm_ik import SpotArmIK


class SpotSimulatorNNNoBase:
    def __init__(self, model_path: str, norm_params_path: str):
        """
        Args:
            model_path (str): The path to the saved TensorFlow model.
        """
        self.model = tf.keras.models.load_model(
            model_path, custom_objects={"mse": tf.keras.losses.MeanSquaredError()}
        )
        with open(norm_params_path, "rb") as f:
            self.norm_params = pickle.load(f)
        self.spot_arm_ik = SpotArmIK()

    def normalize(self, data, mean, std):
        return (data - mean) / std

    def denormalize(self, data, mean, std):
        return data * std + mean

    def step(self, current_state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """
        Predict the next state given the current state and action using a neural network model.
        Use IK to calculate joint positions from end effector positions for each state.

        Args:
            current_state (np.ndarray): The current state of the system.
            action (np.ndarray): The action taken.

        State space:
            ee_x                - frame: body
            ee_y                - frame: body
            ee_z                - frame: body
            ee_rx               - frame: body
            ee_ry               - frame: body
            ee_rz               - frame: body
            ee_vx               - frame: body
            ee_vy               - frame: body
            ee_vz               - frame: body
            ee_vrx              - frame: body
            ee_vry              - frame: body
            ee_vrz              - frame: body
            arm_joint_positions - frame: none

        Action space:
            base_vx             - frame: body
            base_vy             - frame: body
            base_vrot           - frame: body
            ee_vx               - frame: body
            ee_vy               - frame: body
            ee_vz               - frame: body
            ee_vrx              - frame: body
            ee_vry              - frame: body
            ee_vrz              - frame: body

        Returns:
            np.ndarray: The predicted next state.
        """
        # current_state = np.array(current_state, dtype=np.float32)
        # current_state_no_joints = current_state[:12]
        # current_joint_state = current_state[12:]

        current_state_no_joints = current_state
        action = np.array(action, dtype=np.float32)

        # Normalize input features
        current_state_no_joints = self.normalize(
            current_state_no_joints,
            self.norm_params["inputs_states_mean"],
            self.norm_params["inputs_states_std"],
        )
        action = self.normalize(
            action,
            self.norm_params["inputs_actions_mean"],
            self.norm_params["inputs_actions_std"],
        )

        input_features = np.concatenate([current_state_no_joints, action])
        input_features = np.expand_dims(input_features, axis=0)

        predicted_state_no_joints = self.model.predict(input_features)[0]

        # Denormalize output features
        predicted_state_no_joints = self.denormalize(
            predicted_state_no_joints,
            self.norm_params["outputs_mean"],
            self.norm_params["outputs_std"],
        )

        # # Calculate joint positions from end effector positions
        # ee_x, ee_y, ee_z = predicted_state_no_joints[0:3]
        # ee_target = np.array([ee_x, ee_y, ee_z])

        # joint_positions = self.spot_arm_ik.calculate_ik(ee_target, current_joint_state)

        # next_state = np.concatenate((predicted_state_no_joints, joint_positions))

        next_state = predicted_state_no_joints

        return next_state
