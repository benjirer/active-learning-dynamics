import tensorflow as tf
import numpy as np


class SpotSimulator:
    def __init__(self, params):
        self.params = params

    def step(self, current_state, action):
        theta_t = current_state[2]
        delta_t_half = self.params["delta_t"] / 2.0

        A = tf.constant(
            [
                [1.0, 0.0, 0.0, delta_t_half, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, delta_t_half, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, delta_t_half, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, delta_t_half, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, delta_t_half, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, delta_t_half],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=tf.float32,
        )

        B = tf.stack(
            [
                [
                    delta_t / 2.0 * tf.cos(theta_t) * self.params["base_vx_scale"],
                    -delta_t / 2.0 * tf.sin(theta_t) * self.params["base_vy_scale"],
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    delta_t / 2.0 * tf.sin(theta_t) * self.params["base_vx_scale"],
                    delta_t / 2.0 * tf.cos(theta_t) * self.params["base_vy_scale"],
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [0.0, 0.0, delta_t / 2.0 * self.params["vtheta_scale"], 0.0, 0.0, 0.0],
                [
                    tf.cos(theta_t) * self.params["base_vx_scale"],
                    -tf.sin(theta_t) * self.params["base_vy_scale"],
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    tf.sin(theta_t) * self.params["base_vx_scale"],
                    tf.cos(theta_t) * self.params["base_vy_scale"],
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [0.0, 0.0, self.params["vtheta_scale"], 0.0, 0.0, 0.0],
                [
                    0.0,
                    0.0,
                    0.0,
                    delta_t / 2.0 * tf.cos(theta_t) * self.params["ee_vx_scale"],
                    -delta_t / 2.0 * tf.sin(theta_t) * self.params["ee_vy_scale"],
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    delta_t / 2.0 * tf.sin(theta_t) * self.params["ee_vx_scale"],
                    delta_t / 2.0 * tf.cos(theta_t) * self.params["ee_vy_scale"],
                    0.0,
                ],
                [0.0, 0.0, 0.0, 0.0, 0.0, delta_t / 2.0 * self.params["ee_vz_scale"]],
                [
                    0.0,
                    0.0,
                    0.0,
                    tf.cos(theta_t) * self.params["ee_vx_scale"],
                    -tf.sin(theta_t) * self.params["ee_vy_scale"],
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    tf.sin(theta_t) * self.params["ee_vx_scale"],
                    tf.cos(theta_t) * self.params["ee_vy_scale"],
                    0.0,
                ],
                [0.0, 0.0, 0.0, 0.0, 0.0, self.params["ee_vz_scale"]],
            ],
            axis=0,
        )

        next_state = tf.matmul(A, tf.expand_dims(current_state, -1)) + tf.matmul(
            B, tf.expand_dims(action, -1)
        )
        return tf.squeeze(next_state)


class SpotSimTrainer:
    def __init__(self, params):
        self.simulator = SpotSimulator(params)
        self.optimizer = tf.optimizers.Adam(learning_rate=0.001)
        self.params = params

    def train_step(self, states, actions, next_states, horizon=5):
        with tf.GradientTape() as tape:
            total_loss = 0
            for i in range(horizon):
                if i == 0:
                    predicted_states = self.simulator.step(states[0], actions[0])
                else:
                    predicted_states = self.simulator.step(
                        predicted_states[0], actions[0]
                    )

                true_states = next_states[:, i, :]
                loss = tf.reduce_mean(tf.square(predicted_states - true_states))
                total_loss += loss

            total_loss /= horizon

        gradients = tape.gradient(total_loss, list(self.params.values()))
        self.optimizer.apply_gradients(zip(gradients, list(self.params.values())))
        return total_loss

    def train(self, states, actions, next_states, num_epochs):
        for epoch in range(num_epochs):
            cumulative_loss = 0
            for st, at, next_st in zip(states, actions, next_states):
                loss = self.train_step(
                    st[None, :], at[None, :], next_st[None, None, :], horizon=5
                )
                cumulative_loss += loss.numpy()

            average_loss = cumulative_loss / len(states)
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}, Average Loss: {average_loss}")

        return {key: var.numpy() for key, var in self.params.items()}


def load_data_set():
    return np.random.randn(100, 12), np.random.randn(100, 6), np.random.randn(100, 12)


if __name__ == "__main__":
    # Load your dataset here
    previous_states, actions, next_states = load_data_set()

    # Initialize parameters
    params = {
        "base_vx_scale": tf.Variable(1.0, dtype=tf.float32),
        "base_vy_scale": tf.Variable(1.0, dtype=tf.float32),
        "vtheta_scale": tf.Variable(1.0, dtype=tf.float32),
        "ee_vx_scale": tf.Variable(1.0, dtype=tf.float32),
        "ee_vy_scale": tf.Variable(1.0, dtype=tf.float32),
        "ee_vz_scale": tf.Variable(1.0, dtype=tf.float32),
        "delta_t": tf.Variable(0.1, dtype=tf.float32),
    }

    trainer = SpotSimTrainer(params)
    optimized_params = trainer.train(
        previous_states, actions, next_states, num_epochs=100
    )
    print("Optimized Parameters:", optimized_params)
