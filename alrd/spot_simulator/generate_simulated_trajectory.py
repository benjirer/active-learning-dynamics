import numpy as np
import pandas as pd
from typing import Tuple, List
import pickle
from scipy.spatial.transform import Rotation as R
import re

from alrd.run_spot import SessionBuffer, DataBuffer, TransitionData, StateData, TimeData
from alrd.spot_gym.model.robot_state import SpotState

from alrd.spot_simulator.spot_simulator import SpotSimulator
from alrd.utils.data_utils import load_data, load_data_set


def generate_trajectory(
    actions,
    initial_state,
    steps: int = 100,
    params: List[float] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
) -> List[np.ndarray]:
    """
    Generates a simulated trajectory using SpotSimulator class.

    Args:
        actions (): The actions taken (loaded from collected data).
        initial_state: The initial state of the robot (loaded from collected data).

    Returns:
        List[np.ndarray]: The simulated trajectory as a list of states.
    """

    # initialize
    simulator = SpotSimulator()
    actions = actions
    trajectory = [initial_state]

    # simulate trajectory
    for step, action in enumerate(actions, start=1):
        next_state = simulator.step(
            trajectory[-1],
            action,
            b=params,
        )
        trajectory.append(next_state)
        print("Step:", step)

    return trajectory


# load data
session_path = "/home/bhoffman/Documents/MT FS24/active-learning-dynamics/collected_data/test20240815-151559/session_buffer.pickle"
previous_states, actions, next_states = load_data_set(file_path=session_path)
steps = len(actions)

# set params
# params = [0.84279954, 0.8125736, 0.052791923, 0.455076, 0.053945437, 0.31142652]
# params = [
#     0.6817563492785863,
#     0.6778495899839831,
#     0.6224323484049439,
#     0.0,
#     0.0,
#     0.3785519045980369,
# ]  # scipy
# params = [
#     0.84279954,
#     0.8125736,
#     0.052791923,
#     0.455076,
#     0.053945437,
#     0.31142652,
# ]  # tf

# generate and save trajectory
trajectory = generate_trajectory(actions, previous_states[0], steps)
timestamp = re.search(r"test(\d{8}-\d{6})/", session_path).group(1)
output_path = f"/home/bhoffman/Documents/MT FS24/active-learning-dynamics/alrd/spot_simulator/generated_trajectories/trajectory_{timestamp}_{pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')}.pickle"
with open(output_path, "wb") as file:
    pickle.dump(trajectory, file)
print(f"Trajectory saved to {output_path}")
