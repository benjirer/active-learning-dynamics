import numpy as np
import pandas as pd
import tensorflow as tf

# TODO: write data parser
# data
data = {
    "previous_state": [np.random.rand(10) for _ in range(100)],
    "action": [np.random.rand(3) for _ in range(100)],
    "next_state": [np.random.rand(10) for _ in range(100)],
}

df = pd.DataFrame(data)
previous_state = np.vstack(df["previous_state"].values).astype(np.float32)
action = np.vstack(df["action"].values).astype(np.float32)
next_state = np.vstack(df["next_state"].values).astype(np.float32)

# initialize b
b = tf.Variable(
    tf.random.normal((previous_state.shape[1], action.shape[1])), dtype=tf.float32
)

# hyperparameters
learning_rate = 0.01
epochs = 1000
multistep_horizon = 5


# loss computation
def compute_loss(previous_state, action, next_state, b, horizon=1):
    loss = 0
    current_state = previous_state
    for step in range(horizon):
        predicted_next_state = current_state + tf.matmul(action, b, transpose_b=True)
        loss += tf.reduce_mean(tf.square(predicted_next_state - next_state))
        current_state = predicted_next_state
    return loss / horizon


# optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# training loop
for epoch in range(epochs):
    for i in range(len(previous_state)):
        idx = np.random.randint(0, len(previous_state))
        prev_state_sample = previous_state[idx : idx + 1]
        action_sample = action[idx : idx + 1]
        next_state_sample = next_state[idx : idx + 1]

        with tf.GradientTape() as tape:
            # compute loss
            loss = compute_loss(
                prev_state_sample,
                action_sample,
                next_state_sample,
                b,
                horizon=multistep_horizon,
            )

        # compute gradients
        gradients = tape.gradient(loss, [b])

        # apply gradients
        optimizer.apply_gradients(zip(gradients, [b]))

    # calculate loss
    if epoch % 100 == 0:
        loss = compute_loss(
            previous_state, action, next_state, b, horizon=multistep_horizon
        )
        print(f"Epoch {epoch}, Loss: {loss.numpy()}")

print("Training complete.")
print("Final parameter b:", b.numpy())
