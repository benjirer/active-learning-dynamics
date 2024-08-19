import numpy as np
import tensorflow as tf
from alrd.utils.data_utils import load_data_set
from alrd.spot_gym.model.robot_state import SpotState
from alrd.spot_simulator.spot_simulator import SpotSimulator

from alrd.run_spot import SessionBuffer, DataBuffer, TransitionData, StateData, TimeData
from alrd.spot_gym.model.robot_state import SpotState


class SpotSimulatorModel(tf.Module):
    def __init__(self):
        # initialize randomly
        self.logits_b = tf.Variable(tf.ones(6) * 0.5, dtype=tf.float32)
        # self.b = tf.Variable(tf.random.normal([6]), dtype=tf.float32)

    @property
    def b(self):
        return tf.sigmoid(self.logits_b)

    @tf.function
    def step(self, current_state, action):
        theta_t = current_state[2]
        cos_t = tf.cos(theta_t)
        sin_t = tf.sin(theta_t)

        delta_t = 1.0 / 10.0

        base_vx_action_world = cos_t * action[0] - sin_t * action[1]
        base_vy_action_world = sin_t * action[0] + cos_t * action[1]
        vtheta_action_world = action[2]

        base_vx_action_world = (
            self.b[0] * current_state[3] + (1 - self.b[0]) * base_vx_action_world
        )
        base_vy_action_world = (
            self.b[1] * current_state[4] + (1 - self.b[1]) * base_vy_action_world
        )
        vtheta_action_world = (
            self.b[2] * current_state[5] + (1 - self.b[2]) * vtheta_action_world
        )

        next_state = tf.TensorArray(dtype=tf.float32, size=12)
        next_state = next_state.write(
            0, current_state[0] + delta_t * base_vx_action_world
        )
        next_state = next_state.write(
            1, current_state[1] + delta_t * base_vy_action_world
        )
        next_state = next_state.write(
            2,
            (current_state[2] + delta_t * vtheta_action_world + np.pi) % (2 * np.pi)
            - np.pi,
        )
        next_state = next_state.write(3, base_vx_action_world)
        next_state = next_state.write(4, base_vy_action_world)
        next_state = next_state.write(5, vtheta_action_world)

        ee_vx_action_world = cos_t * action[3] - sin_t * action[4]
        ee_vy_action_world = sin_t * action[3] + cos_t * action[4]
        ee_vz_action_world = action[5]

        ee_vx_action_world = (
            self.b[3] * current_state[9] + (1 - self.b[3]) * ee_vx_action_world
        )
        ee_vy_action_world = (
            self.b[4] * current_state[10] + (1 - self.b[4]) * ee_vy_action_world
        )
        ee_vz_action_world = (
            self.b[5] * current_state[11] + (1 - self.b[5]) * ee_vz_action_world
        )

        base_xy_world = tf.stack([current_state[0], current_state[1]])
        ee_xy_world = tf.stack([current_state[6], current_state[7]])
        distance_ee_base = tf.norm(ee_xy_world - base_xy_world)
        vel_ee_from_rot_magnitude = distance_ee_base * vtheta_action_world
        alpha = tf.atan2(
            ee_xy_world[0] - base_xy_world[0], ee_xy_world[1] - base_xy_world[1]
        )
        vel_ee_from_rot_x_world = -vel_ee_from_rot_magnitude * tf.cos(alpha)
        vel_ee_from_rot_y_world = vel_ee_from_rot_magnitude * tf.sin(alpha)

        ee_vx_final = (
            ee_vx_action_world + base_vx_action_world + vel_ee_from_rot_x_world
        )
        ee_vy_final = (
            ee_vy_action_world + base_vy_action_world + vel_ee_from_rot_y_world
        )
        ee_vz_final = ee_vz_action_world

        next_state = next_state.write(6, current_state[6] + delta_t * ee_vx_final)
        next_state = next_state.write(7, current_state[7] + delta_t * ee_vy_final)
        next_state = next_state.write(
            8, tf.maximum(current_state[8] + delta_t * ee_vz_final, 0.0)
        )
        next_state = next_state.write(9, ee_vx_final)
        next_state = next_state.write(10, ee_vy_final)
        next_state = next_state.write(11, ee_vz_final)

        return next_state.stack()


def loss_function(
    model, current_states, actions, next_states, variances, batch_size=32, num_steps=5
):
    batch_loss = 0.0
    for i in tf.range(0, batch_size):
        predicted_state = current_states[i]
        step_loss = 0.0
        num_steps_curr = min(num_steps, len(actions) - i)
        for step in tf.range(num_steps_curr):
            predicted_state = model.step(predicted_state, actions[i + step])
            step_loss += tf.reduce_sum(
                ((predicted_state - next_states[i + step]) ** 2) / variances
            )
        batch_loss += step_loss / tf.cast(num_steps_curr, tf.float32)
    return batch_loss / batch_size


def compute_variances(states):
    return np.var(states, axis=0)


def train(
    file_path: str,
    num_epochs: int = 100,
    learning_rate: float = 0.01,
    batch_size: int = 32,
):
    spot_simulator_model = SpotSimulatorModel()
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

    # load data
    previous_states, actions, next_states = load_data_set(
        file_path=file_path,
    )

    previous_states = tf.convert_to_tensor(previous_states, dtype=tf.float32)
    actions = tf.convert_to_tensor(actions, dtype=tf.float32)
    next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)

    for epoch in range(num_epochs):

        i = 0
        while i in range(len(previous_states) - batch_size):

            batch_previous_states = previous_states[i : i + batch_size]
            batch_actions = actions[i : i + batch_size]
            batch_next_states = next_states[i : i + batch_size]
            batch_variances = compute_variances(
                np.vstack((batch_previous_states, batch_next_states))
            )

            with tf.GradientTape() as tape:
                loss = loss_function(
                    spot_simulator_model,
                    batch_previous_states,
                    batch_actions,
                    batch_next_states,
                    batch_variances,
                    batch_size=batch_size,
                )
            gradients = tape.gradient(loss, [spot_simulator_model.logits_b])
            optimizer.apply_gradients(zip(gradients, [spot_simulator_model.logits_b]))

            if i % 20 == 0:
                print(f"Step {i}, Loss: {loss.numpy()}")

            i += batch_size

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch}, Loss: {loss.numpy()}, Alpha: {spot_simulator_model.b.numpy()}"
            )

    optimized_b = spot_simulator_model.b.numpy()
    print("Optimized b:", list(optimized_b))
    return optimized_b


if __name__ == "__main__":

    file_path = "/home/bhoffman/Documents/MT FS24/active-learning-dynamics/collected_data/test20240806-135621/session_buffer.pickle"
    b = train(file_path)