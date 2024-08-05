import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from scipy.spatial.transform import Rotation as R
import pickle
import pandas as pd

from alrd.run_spot import SessionBuffer, DataBuffer, TransitionData, StateData, TimeData
from alrd.spot_gym.model.robot_state import SpotState
from utils import normalize_data, load_data


def train_model(previous_states, actions, next_states) -> tf.keras.Model:

    # split data
    states_idx = previous_states.shape[1]
    print(states_idx)
    inputs = np.concatenate((previous_states, actions), axis=1)
    outputs = next_states
    inputs_train, inputs_test, outputs_train, outputs_test = train_test_split(
        inputs, outputs, test_size=0.2, random_state=42
    )

    # normalize input data (for inputs: seperately for actions and states)
    inputs_train_states = inputs_train[:, :states_idx]
    inputs_train_actions = inputs_train[:, states_idx:]

    inputs_train_states, inputs_states_mean, inputs_states_std = normalize_data(
        inputs_train_states
    )
    inputs_train_actions, inputs_actions_mean, inputs_actions_std = normalize_data(
        inputs_train_actions
    )
    inputs_train = np.concatenate((inputs_train_states, inputs_train_actions), axis=1)

    # normalize output data
    outputs_train, outputs_mean, outputs_std = normalize_data(outputs_train)

    input_dim = inputs_train.shape[1]
    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(input_dim,)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(outputs_train.shape[1]),
        ]
    )

    model.compile(optimizer="adam", loss="mse")

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        min_delta=0.0001,
        mode="min",
        verbose=1,
        restore_best_weights=True,
    )

    # fit
    model.fit(
        inputs_train,
        outputs_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        verbose=1,
        callbacks=[early_stopping],
    )

    # normalize test data
    inputs_test_states = inputs_test[:, :states_idx]
    inputs_test_actions = inputs_test[:, states_idx:]

    inputs_test_states = (inputs_test_states - inputs_states_mean) / inputs_states_std
    inputs_test_actions = (
        inputs_test_actions - inputs_actions_mean
    ) / inputs_actions_std
    inputs_test = np.concatenate((inputs_test_states, inputs_test_actions), axis=1)

    outputs_test = (outputs_test - outputs_mean) / outputs_std

    test_loss = model.evaluate(inputs_test, outputs_test)

    print(f"Test Loss: {test_loss}")

    norm_params = {
        "inputs_states_mean": inputs_states_mean,
        "inputs_states_std": inputs_states_std,
        "inputs_actions_mean": inputs_actions_mean,
        "inputs_actions_std": inputs_actions_std,
        "outputs_mean": outputs_mean,
        "outputs_std": outputs_std,
    }

    print(norm_params)

    return model, norm_params


if __name__ == "__main__":
    file_path = "/home/bhoffman/Documents/MT FS24/active-learning-dynamics/collected_data/test20240730-174534/session_buffer.pickle"
    previous_states, actions, next_states = load_data(file_path)
    model, norm_params = train_model(previous_states, actions, next_states)

    model_path = "/home/bhoffman/Documents/MT FS24/active-learning-dynamics/alrd/spot_simulator/models/"
    timestamp = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
    tf.keras.models.save_model(model, f"{model_path}model_{timestamp}.keras")
    with open(f"{model_path}norm_params_{timestamp}.pkl", "wb") as f:
        pickle.dump(norm_params, f)
