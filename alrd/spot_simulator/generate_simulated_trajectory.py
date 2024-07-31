import numpy as np
from alrd.spot_simulator.spot_simulator import SpotSimulator
import pickle
from scipy.spatial.transform import Rotation as R
from alrd.run_spot import SessionBuffer, DataBuffer, TransitionData, StateData, TimeData
from alrd.spot_gym.model.robot_state import SpotState


def generate_trajectory(b: np.ndarray, actions, initial_state, steps: int = 100):
    """
    Generates a simulated trajectory using SpotSimulator class.

    Args:
        b (np.ndarray): The transition parameter.
        actions: The actions taken (loaded from collected data).
        initial_state: The initial state of the robot (loaded from collected data).

    Returns:
        List[np.ndarray]: The simulated trajectory.
    """

    # Initialize the simulator
    simulator = SpotSimulator(b)
    actions = actions[:steps]

    # Initialize the trajectory
    trajectory = [initial_state]

    # Simulate the trajectory
    for step, action in enumerate(actions, start=1):
        next_state = simulator.step(trajectory[-1], action)
        trajectory.append(next_state)
        print("Step:", step)

    return trajectory


def load_data(file_path: str):
    """
    Load and parse data from a pickle file to get initial state and actions.

    Args:
        file_path (str): The path to pickle file.

    Returns:
        Tuple of numpy arrays for initial state and actions.
    """
    with open(file_path, "rb") as file:
        data = pickle.load(file)

    states_data = data.data_buffers[0].states

    # parse initial state
    initial_state_data = states_data[0]
    initial_state_pre = initial_state_data.next_state
    x, y, _, qx, qy, qz, qw = initial_state_pre.pose_of_body_in_vision
    angle = R.from_quat([qx, qy, qz, qw]).as_euler("xyz", degrees=False)[2]
    vx, vy, _, _, _, w = initial_state_pre.velocity_of_body_in_vision

    hand_x, hand_y, hand_z, hand_qx, hand_qy, hand_qz, hand_qw = (
        initial_state_pre.pose_of_hand_in_body
    )
    hand_rx, hand_ry, hand_rz = R.from_quat(
        [hand_qx, hand_qy, hand_qz, hand_qw]
    ).as_euler("xyz", degrees=False)
    hand_vx, hand_vy, hand_vz, hand_vrx, hand_vry, hand_vrz = (
        initial_state_pre.velocity_of_hand_in_body
    )

    sh0, sh1, el0, el1, wr0, wr1 = initial_state_pre.arm_joint_positions

    initial_state_list = [
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
        sh0,
        sh1,
        el0,
        el1,
        wr0,
        wr1,
    ]
    initial_state = np.array(initial_state_list, dtype=np.float32)

    # parse actions (skip first state)
    actions = [np.array(s.action, dtype=np.float32) for s in states_data[1:]]

    return initial_state, actions


if __name__ == "__main__":
    # Load transition parameter b
    b = np.load(
        "/home/bhoffman/Documents/MT FS24/active-learning-dynamics/alrd/spot_simulator/transition_parameters/b_20240731-113113.npy"
    )

    # Actions and initial state
    initial_state, actions = load_data(
        "/home/bhoffman/Documents/MT FS24/active-learning-dynamics/collected_data/test20240730-174534/session_buffer.pickle"
    )

    # Generate trajectory
    steps = 100
    trajectory = generate_trajectory(b, actions, initial_state, steps)

    # Save trajectory
    with open("trajectory.pickle", "wb") as file:
        pickle.dump(trajectory, file)
