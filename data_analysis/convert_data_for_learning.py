import pickle
import pandas as pd
import jax.numpy as jnp
import re

from alrd.utils.data_utils import load_data_set
from alrd.run_spot import SessionBuffer, DataBuffer, TransitionData, StateData, TimeData
from alrd.spot_gym.model.robot_state import SpotState


def convert_for_learning(
    file_path: str,
    as_euler: bool = True,
    with_wrist: bool = False,
    with_joints: bool = False,
    skip_first: bool = True,
    start_idx: int = 0,
    end_idx: int = None,
    format: str = "numpy",
):
    """
    Export a data set to pickle file as a tuple of numpy arrays for previous_states, actions, and next_states.

    Args:
        file_path (str): The path to the pickle file.
        as_euler (bool): Whether to return the orientation as Euler angles.
        with_wrist (bool): Whether to include wrist positions and velocity in the state space.
        with_joints (bool): Whether to include joint positions in the state space.
        skip_first (bool): Whether to skip the first state.
        start_idx (int): The index of the first state to include.
        end_idx (int): The index of the last state to include.
        format (str): The format of the data set. Either "numpy" or "jax".
    """
    data_set = load_data_set(
        file_path,
        as_euler=as_euler,
        with_wrist=with_wrist,
        with_joints=with_joints,
        skip_first=skip_first,
        start_idx=start_idx,
        end_idx=end_idx,
    )

    # convert to jnp.ndarray if format is "jax"
    if format == "jax":
        previous_states, actions, next_states = data_set
        data_set = (
            jnp.array(previous_states),
            jnp.array(actions),
            jnp.array(next_states),
        )

    return data_set


# load and convert data
session_path = "/home/bhoffman/Documents/MT FS24/active-learning-dynamics/collected_data/test20240806-135621/session_buffer.pickle"
format = "jax"
data_set_converted = convert_for_learning(
    file_path=session_path,
    as_euler=True,
    with_wrist=False,
    with_joints=False,
    skip_first=True,
    start_idx=400,
    end_idx=900,
    format=format,
)

# export converted data
output_path = "/home/bhoffman/Documents/MT FS24/active-learning-dynamics/data_analysis/learning_data/"
timestamp = re.search(r"test(\d{8}-\d{6})/", session_path).group(1)
output_path = f"{output_path}dataset_learn_{format}_{timestamp}_{pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')}.pickle"
with open(output_path, "wb") as file:
    pickle.dump(data_set_converted, file)
print(f"Data set exported to {output_path}")
