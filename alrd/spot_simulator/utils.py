import numpy as np
import pickle
from scipy.spatial.transform import Rotation as R

from alrd.run_spot import SessionBuffer, DataBuffer, TransitionData, StateData, TimeData
from alrd.spot_gym.model.robot_state import SpotState


def normalize_data(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / std, mean, std


def load_data(
    file_path: str,
    which_data: str,
    as_euler: bool = False,
    with_base: bool = False,
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
        with_base (bool): Whether to include base velocities in the action space.
        with_joints (bool): Whether to include joint positions in the state space.
        skip_first (bool): Whether to skip the first state.
        start_idx (int): The index of the first state to include.
        end_idx (int): The index of the last state to include.


    Returns:
        Numpy array for the requested data.

    """

    # load data set
    with open(file_path, "rb") as file:
        data = pickle.load(file)

    states_data = data.data_buffers[0].states
    if end_idx is not None:
        states_data = states_data[start_idx:end_idx]
    else:
        if skip_first:
            states_data = states_data[1:]
    data_vector = []

    # parse selected data vector
    for state in states_data:

        if which_data == "previous_state":
            state_data = state.last_state
        elif which_data == "next_state":
            state_data = state.next_state
        elif which_data == "action":
            state_data = state.action
            data_vector.append(np.array(state_data, dtype=np.float32))
            return np.stack(data_vector)
        else:
            raise ValueError(
                "which_data must be either 'previous_state', 'next_state', or 'action'."
            )

        # base state
        body_vector = []
        x, y, z, qx, qy, qz, qw = state_data.pose_of_body_in_vision
        vx, vy, _, _, _, w = state_data.velocity_of_body_in_vision

        # hand state
        hand_state = []
        hand_x, hand_y, hand_z, hand_qx, hand_qy, hand_qz, hand_qw = (
            state_data.pose_of_hand_in_body
        )
        hand_vx, hand_vy, hand_vz, hand_vrx, hand_vry, hand_vrz = (
            state_data.velocity_of_hand_in_body
        )

        # add positions
        body_vector.extend([x, y])
        hand_state.extend([hand_x, hand_y, hand_z])

        # add orientations
        if as_euler:
            body_heading = R.from_quat([qx, qy, qz, qw]).as_euler("xyz", degrees=False)[
                2
            ]
            hand_rx, hand_ry, hand_rz = R.from_quat(
                [hand_qx, hand_qy, hand_qz, hand_qw]
            ).as_euler("xyz", degrees=False)

            body_vector.extend([body_heading])
            hand_state.extend([hand_rx, hand_ry, hand_rz])
        else:
            body_vector.extend([qx, qy, qz, qw])
            hand_state.extend([hand_qx, hand_qy, hand_qz, hand_qw])

        # add velocities
        body_vector.extend([vx, vy, w])
        hand_state.extend([hand_vx, hand_vy, hand_vz, hand_vrx, hand_vry, hand_vrz])

        # concat
        if with_base:
            data_vector.append(np.array(body_vector + hand_state, dtype=np.float32))
        else:
            data_vector.append(np.array(hand_state, dtype=np.float32))

        if with_joints:
            arm_joint_positions = state_data.arm_joint_positions
            data_vector[-1] = np.concatenate([data_vector[-1], arm_joint_positions])

    return np.stack(data_vector)


def load_data_set(
    file_path: str,
    as_euler: bool = False,
    with_base: bool = False,
    with_joints: bool = False,
    skip_first: bool = True,
    start_idx: int = 0,
    end_idx: int = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and parse data set from a pickle file.

    State space:
        ee_x                - frame: body
        ee_y                - frame: body
        ee_z                - frame: body
        ee_qx               - frame: body
        ee_qy               - frame: body
        ee_qz               - frame: body
        ee_qw               - frame: body
        ee_vx               - frame: body
        ee_vy               - frame: body
        ee_vz               - frame: body
        ee_vrx              - frame: body
        ee_vry              - frame: body
        ee_vrz              - frame: body

    Action space:
        base_vx             - frame: body
        base_vy             - frame: body
        base_vrot           - frame: body
        ee_vx               - frame: body
        ee_vy               - frame: body
        ee_vz               - frame: body
        ee_vrx              - frame: body
        ee_vry              - frame: body
        ee_vrz              - frame: body

    Args:
        file_path (str): The path to the pickle file.
        as_euler (bool): Whether to return the orientation as Euler angles.
        with_base (bool): Whether to include base velocities in the action space.
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
        with_base=with_base,
        with_joints=with_joints,
        skip_first=skip_first,
        start_idx=start_idx,
        end_idx=end_idx,
    )
    actions = load_data(
        file_path,
        which_data="action",
        as_euler=as_euler,
        with_base=with_base,
        with_joints=with_joints,
        skip_first=skip_first,
        start_idx=start_idx,
        end_idx=end_idx,
    )
    next_states = load_data(
        file_path,
        which_data="next_state",
        as_euler=as_euler,
        with_base=with_base,
        with_joints=with_joints,
        skip_first=skip_first,
        start_idx=start_idx,
        end_idx=end_idx,
    )

    return previous_states, actions, next_states
