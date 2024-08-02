import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from scipy.spatial.transform import Rotation as R
from typing import Tuple
from alrd.run_spot import SessionBuffer, DataBuffer, TransitionData, StateData, TimeData
from alrd.spot_gym.model.robot_state import SpotState


def load_data(file_path: str):
    """
    Load and parse data from a pickle file.

    State space:
        base_x              - frame: vision
        base_y              - frame: vision
        heading             - frame: vision
        base_vx             - frame: vision
        base_vy             - frame: vision
        base_vrot           - frame: vision
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

    Args:
        file_path (str): The path to pickle file.

    Returns:
        Tuple of numpy arrays for previous states, actions, and next states.
    """
    with open(file_path, "rb") as file:
        data = pickle.load(file)

    states_data = data.data_buffers[0].states
    states_data = states_data[1:]  # skip first state for ensuring continuity
    previous_states = []
    actions = []
    next_states = []

    for state in states_data:
        # Parsing previous state
        previous_state = state.last_state
        x, y, _, qx, qy, qz, qw = previous_state.pose_of_body_in_vision
        angle = R.from_quat([qx, qy, qz, qw]).as_euler("xyz", degrees=False)[2]
        vx, vy, _, _, _, w = previous_state.velocity_of_body_in_vision

        hand_x, hand_y, hand_z, hand_qx, hand_qy, hand_qz, hand_qw = (
            previous_state.pose_of_hand_in_body
        )
        hand_rx, hand_ry, hand_rz = R.from_quat(
            [hand_qx, hand_qy, hand_qz, hand_qw]
        ).as_euler("xyz", degrees=False)
        hand_vx, hand_vy, hand_vz, hand_vrx, hand_vry, hand_vrz = (
            previous_state.velocity_of_hand_in_body
        )

        prev_state_list = [
            x,
            y,
            angle,
            vx,
            vy,
            w,
            hand_x,
            hand_y,
            hand_z,
            hand_rx,
            hand_ry,
            hand_rz,
            hand_vx,
            hand_vy,
            hand_vz,
            hand_vrx,
            hand_vry,
            hand_vrz,
        ]
        previous_states.append(np.array(prev_state_list, dtype=np.float32))

        # Parsing next state
        next_state = state.next_state
        x, y, _, qx, qy, qz, qw = next_state.pose_of_body_in_vision
        angle = R.from_quat([qx, qy, qz, qw]).as_euler("xyz", degrees=False)[2]
        vx, vy, _, _, _, w = next_state.velocity_of_body_in_vision

        hand_x, hand_y, hand_z, hand_qx, hand_qy, hand_qz, hand_qw = (
            next_state.pose_of_hand_in_body
        )
        hand_rx, hand_ry, hand_rz = R.from_quat(
            [hand_qx, hand_qy, hand_qz, hand_qw]
        ).as_euler("xyz", degrees=False)
        hand_vx, hand_vy, hand_vz, hand_vrx, hand_vry, hand_vrz = (
            next_state.velocity_of_hand_in_body
        )

        next_state_list = [
            x,
            y,
            angle,
            vx,
            vy,
            w,
            hand_x,
            hand_y,
            hand_z,
            hand_rx,
            hand_ry,
            hand_rz,
            hand_vx,
            hand_vy,
            hand_vz,
            hand_vrx,
            hand_vry,
            hand_vrz,
        ]
        next_states.append(np.array(next_state_list, dtype=np.float32))

    # Parsing action
    actions = [np.array(s.action, dtype=np.float32) for s in states_data]

    # Convert lists of arrays to single numpy arrays
    previous_states = np.stack(previous_states)
    actions = np.stack(actions)
    next_states = np.stack(next_states)

    return previous_states, actions, next_states


def train_model(previous_states, actions, next_states) -> np.ndarray:
    """
    Train a model to predict the next state given the previous state and action.
    """
    b = tf.Variable(
        tf.random.normal((previous_states.shape[1], actions.shape[1])), dtype=tf.float32
    )

    # Training parameters
    learning_rate = 0.01
    epochs = 100
    multistep_horizon = 5
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Loss function
    def compute_loss(prev_state, actions, next_states, b, horizon=1):
        loss = 0
        current_state = prev_state[0:1]
        for step in range(horizon):
            next_state = next_states[step : step + 1]
            action = actions[step : step + 1]
            predicted_next_state = current_state + tf.matmul(
                action, b, transpose_b=True
            )
            loss += tf.reduce_mean(tf.square(predicted_next_state - next_state))
            current_state = predicted_next_state
        return loss / horizon

    # Training loop
    for epoch in range(epochs):
        for i in range(len(previous_states) - multistep_horizon):
            idx = np.random.randint(0, len(previous_states))
            prev_state_sample = previous_states[idx : idx + 1]
            action_samples = actions[idx : idx + multistep_horizon]
            next_state_samples = next_states[idx : idx + multistep_horizon]

            with tf.GradientTape() as tape:
                loss = compute_loss(
                    prev_state_sample,
                    action_samples,
                    next_state_samples,
                    b,
                    horizon=multistep_horizon,
                )
            gradients = tape.gradient(loss, [b])
            optimizer.apply_gradients(zip(gradients, [b]))

            if epoch % 100 == 0 and i == 0:
                print(f"Epoch {epoch}, Sample Loss: {loss.numpy()}")

    print("Training complete.")
    return b.numpy()


if __name__ == "__main__":
    file_path = "/home/bhoffman/Documents/MT FS24/active-learning-dynamics/collected_data/test20240730-174534/session_buffer.pickle"
    previous_states, actions, next_states = load_data(file_path)
    b = train_model(previous_states, actions, next_states)

    # Save transition parameter b
    timestamp = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
    np.save(
        "/home/bhoffman/Documents/MT FS24/active-learning-dynamics/alrd/spot_simulator/transition_parameters/b_"
        + timestamp,
        b,
    )
