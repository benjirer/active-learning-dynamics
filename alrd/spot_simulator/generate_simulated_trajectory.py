import numpy as np
import pandas as pd
from typing import Tuple, List
import pickle
from scipy.spatial.transform import Rotation as R

from alrd.run_spot import SessionBuffer, DataBuffer, TransitionData, StateData, TimeData
from alrd.spot_gym.model.robot_state import SpotState

from alrd.spot_simulator.spot_simulator import SpotSimulator
from alrd.spot_simulator.utils import load_data, load_data_set


def generate_trajectory(
    actions,
    initial_state,
    steps: int = 100,
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
            [0.84279954, 0.8125736, 0.052791923, 0.455076, 0.053945437, 0.31142652],
        )
        trajectory.append(next_state)
        print("Step:", step)

    return trajectory


if __name__ == "__main__":

    # load data
    session_path = "/home/bhoffman/Documents/MT FS24/active-learning-dynamics/collected_data/test20240806-135621/session_buffer.pickle"
    previous_states, actions, next_states = load_data_set(file_path=session_path)
    steps = len(actions)

    # # load parameters from pickle file
    # with open(
    #     "/home/bhoffman/Documents/MT FS24/active-learning-dynamics/alrd/spot_simulator/learned_parameters/parameters_1723130964.053997.pickle",
    #     "rb",
    # ) as file:
    #     b = pickle.load(file)

    # generate and save trajectory
    trajectory = generate_trajectory(actions, previous_states[0], steps)
    output_path = f"/home/bhoffman/Documents/MT FS24/active-learning-dynamics/alrd/spot_simulator/generated_trajectories/trajectory_{pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')}.pickle"
    with open(output_path, "wb") as file:
        pickle.dump(trajectory, file)
    print(f"Trajectory saved to {output_path}")
