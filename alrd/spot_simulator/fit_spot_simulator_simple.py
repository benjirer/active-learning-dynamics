import numpy as np
from alrd.spot_simulator.utils import load_data_set
from alrd.run_spot import SessionBuffer, DataBuffer, TransitionData, StateData, TimeData
from alrd.spot_gym.model.robot_state import SpotState
from scipy.optimize import minimize
from alrd.spot_simulator.spot_simulator import SpotSimulator


def compute_variances(states):
    return np.var(states, axis=0)


def loss_function(
    b,
    simulator,
    current_states,
    actions,
    next_states,
    variances,
    batch_size=10,
    num_steps=5,
):
    total_loss = 0
    num_batches = (len(current_states) - num_steps) // batch_size
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = batch_start + batch_size
        batch_loss = 0
        for i in range(batch_start, batch_end):
            predicted_state = current_states[i]
            step_loss = 0
            for step in range(num_steps):
                predicted_state = simulator.step(predicted_state, actions[i + step], b)
                step_loss += np.sum(
                    ((predicted_state - next_states[i + step]) ** 2) / variances
                )
            batch_loss += step_loss / num_steps
        total_loss += batch_loss / batch_size
    return total_loss / num_batches


previous_states, actions, next_states = load_data_set(
    file_path="/home/bhoffman/Documents/MT FS24/active-learning-dynamics/collected_data/test20240806-135621/session_buffer.pickle"
)

variances = compute_variances(np.vstack((previous_states, next_states)))
spot_simulator = SpotSimulator()
initial_b = np.ones(6) * 0.5
bounds = [(0, 1)] * 6

result = minimize(
    loss_function,
    initial_b,
    args=(spot_simulator, previous_states, actions, next_states, variances),
    bounds=bounds,
    method="L-BFGS-B",
)

optimized_b = list(result.x)
print("Optimized b:", optimized_b)
