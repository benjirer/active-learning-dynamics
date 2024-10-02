import numpy as np
import pickle
from scipy.spatial.transform import Rotation as R
from typing import Tuple
import pandas as pd

from alrd.run_spot import SessionBuffer, DataBuffer, TransitionData, StateData, TimeData
from alrd.spot_gym.model.robot_state import SpotState


def project_angle(theta: np.array) -> np.array:
    """
    Project angles to the range [-pi, pi].

    Args:
        theta (np.ndarray): The angle to project.

    Returns:
        The projected angles.
    """
    return (theta + np.pi) % (2 * np.pi) - np.pi


def load_data(
    file_path: str,
    which_data: str,
    as_euler: bool = True,
    with_wrist: bool = False,
    with_joints: bool = False,
    skip_first: bool = True,
    start_idx: int = 0,
    end_idx: int = None,
) -> np.ndarray:
    """
    Load and parse a single data vector from a pickle file.

    Args:
        file_path (str): The path to the pickle file.
        which_state (str): Either "previous_state" or "next_state" or "action".
        as_euler (bool): Whether to return the orientation as Euler angles.
        with_wrist (bool): Whether to include wrist positions and velocity in the state space.
        with_joints (bool): Whether to include joint positions in the state space.
        skip_first (bool): Whether to skip the first state.
        start_idx (int): The index of the first state to include.
        end_idx (int): The index of the last state to include.

    Returns:
        state_vector (np.ndarray): The parsed data vector.

    """

    # load data set
    with open(file_path, "rb") as file:
        data = pickle.load(file)
    states_data = data.data_buffers[0].states
    state_vector = []

    # apply filters
    if end_idx is not None:
        states_data = states_data[start_idx:end_idx]
    if start_idx == 0 and skip_first:
        states_data = states_data[1:]

    # parse selected data vector
    for state in states_data:
        if which_data == "previous_state":
            state_data = state.last_state
        elif which_data == "next_state":
            state_data = state.next_state
        elif which_data == "action":
            state_data = state.action
            state_vector.append(np.array(state_data, dtype=np.float32))
            continue
        else:
            raise ValueError(
                "which_data must be either 'previous_state', 'next_state', or 'action'."
            )

        # base state
        body_vector = []
        x, y, z, qx, qy, qz, qw = state_data.pose_of_body_in_vision
        vx, vy, _, _, _, w = state_data.velocity_of_body_in_vision

        # ee state
        ee_state = []
        ee_x, ee_y, ee_z, ee_qx, ee_qy, ee_qz, ee_qw = state_data.pose_of_hand_in_vision
        ee_vx, ee_vy, ee_vz, ee_vrx, ee_vry, ee_vrz = (
            state_data.velocity_of_hand_in_vision
        )

        # add positions
        body_vector.extend([x, y])
        ee_state.extend([ee_x, ee_y, ee_z])

        # add orientations
        if as_euler:
            body_heading = R.from_quat([qx, qy, qz, qw]).as_euler("xyz", degrees=False)[
                2
            ]
            body_heading = project_angle(body_heading)
            ee_rx, ee_ry, ee_rz = R.from_quat([ee_qx, ee_qy, ee_qz, ee_qw]).as_euler(
                "xyz", degrees=False
            )

            body_vector.extend([body_heading])
            if with_wrist:
                ee_state.extend([ee_rx, ee_ry, ee_rz])
        else:
            body_vector.extend([qx, qy, qz, qw])
            if with_wrist:
                ee_state.extend([ee_qx, ee_qy, ee_qz, ee_qw])

        # add velocities
        body_vector.extend([vx, vy, w])
        if with_wrist:
            ee_state.extend([ee_vx, ee_vy, ee_vz, ee_vrx, ee_vry, ee_vrz])
        else:
            ee_state.extend([ee_vx, ee_vy, ee_vz])

        # concat
        state_vector.append(np.array(body_vector + ee_state, dtype=np.float32))
        if with_joints:
            arm_joint_positions = state_data.arm_joint_positions
            state_vector[-1] = np.concatenate([state_vector[-1], arm_joint_positions])

    return np.stack(state_vector)


def load_data_set(
    file_path: str,
    as_euler: bool = True,
    with_wrist: bool = False,
    with_joints: bool = False,
    skip_first: bool = True,
    start_idx: int = 0,
    end_idx: int = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and parse data set from a pickle file.
    Note: The actions are determined by what is used during data collection.

    Args:
        file_path (str): The path to the pickle file.
        as_euler (bool): Whether to return the orientation as Euler angles.
        with_wrist (bool): Whether to include wrist positions and velocity in the state space.
        with_joints (bool): Whether to include joint positions in the state space.
        skip_first (bool): Whether to skip the first state.
        start_idx (int): The index of the first state to include.
        end_idx (int): The index of the last state to include.

    Returns:
        Tuple of numpy arrays for previous states, actions, and next states.
    """
    previous_states = load_data(
        file_path,
        which_data="previous_state",
        as_euler=as_euler,
        with_wrist=with_wrist,
        with_joints=with_joints,
        skip_first=skip_first,
        start_idx=start_idx,
        end_idx=end_idx,
    )
    actions = load_data(
        file_path,
        which_data="action",
        as_euler=as_euler,
        with_wrist=with_wrist,
        with_joints=with_joints,
        skip_first=skip_first,
        start_idx=start_idx,
        end_idx=end_idx,
    )
    next_states = load_data(
        file_path,
        which_data="next_state",
        as_euler=as_euler,
        with_wrist=with_wrist,
        with_joints=with_joints,
        skip_first=skip_first,
        start_idx=start_idx,
        end_idx=end_idx,
    )

    assert (
        previous_states.shape[0] == actions.shape[0] == next_states.shape[0]
    ), "Lengths of previous_states, actions, and next_states must be equal."

    return previous_states, actions, next_states
