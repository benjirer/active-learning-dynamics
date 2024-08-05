import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from scipy.spatial.transform import Rotation as R
from typing import Tuple

from alrd.run_spot import SessionBuffer, DataBuffer, TransitionData, StateData, TimeData
from alrd.spot_gym.model.robot_state import SpotState
from utils import normalize_data, load_data_set


def train_model(previous_states, actions, next_states) -> np.ndarray:

    b = tf.Variable(
        tf.random.normal((previous_states.shape[1], actions.shape[1])), dtype=tf.float32
    )

    # Training parameters
    learning_rate = 0.01
    epochs = 100
    multistep_horizon = 5
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # normalize data
    previous_states, prev_mean, prev_std = normalize_data(previous_states)
    actions, act_mean, act_std = normalize_data(actions)
    next_states, next_mean, next_std = normalize_data(next_states)

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

    norm_params = {
        "inputs_states_mean": prev_mean,
        "inputs_states_std": prev_std,
        "inputs_actions_mean": act_mean,
        "inputs_actions_std": act_std,
        "outputs_mean": next_mean,
        "outputs_std": next_std,
    }

    return b.numpy(), norm_params


if __name__ == "__main__":
    file_path = "/home/bhoffman/Documents/MT FS24/active-learning-dynamics/collected_data/test20240730-174534/session_buffer.pickle"
    previous_states, actions, next_states = load_data_set(file_path)
    b, norm_params = train_model(previous_states, actions, next_states)

    parameter_path = "/home/bhoffman/Documents/MT FS24/active-learning-dynamics/alrd/spot_simulator/tansition_parameters/"
    timestamp = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
    np.save(
        f"{parameter_path}b_{timestamp}.npy",
        b,
    )
    with open(f"{parameter_path}norm_params_{timestamp}.pkl", "wb") as f:
        pickle.dump(norm_params, f)
