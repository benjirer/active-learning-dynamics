import pickle
import pandas as pd
import jax.numpy as jnp
import re
from brax.training.types import Transition

from alrd.utils.data_utils import load_data_set
from alrd.run_spot import SessionBuffer, DataBuffer, TransitionData, StateData, TimeData
from alrd.spot_gym.model.robot_state import SpotState


def convert_for_learning(
    file_path: str,
    as_euler: bool = True,
    with_wrist: bool = False,
    with_joints: bool = False,
    skip_first: bool = True,
    encode_angles: bool = True,
    start_idx: int = 0,
    end_idx: int = None,
    format: str = "numpy",
    use_vision: bool = False,
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
        use_vision (bool): Whether to use vision or odom sensor.
    """
    data_set = load_data_set(
        file_path,
        as_euler=as_euler,
        with_wrist=with_wrist,
        with_joints=with_joints,
        encode_angles=encode_angles,
        skip_first=skip_first,
        start_idx=start_idx,
        end_idx=end_idx,
        use_vision=use_vision,
    )

    # convert to jnp.ndarray if format is "jax"
    if format == "jax":
        previous_states, actions, next_states = data_set
        data_set = (
            jnp.array(previous_states),
            jnp.array(actions),
            jnp.array(next_states),
        )
    elif format == "brax":
        previous_states, actions, next_states = data_set

        brax_transitions = []
        for previous_state, action, next_state in zip(
            previous_states, actions, next_states
        ):
            brax_transitions.append(
                Transition(
                    observation=jnp.array(previous_state),
                    action=jnp.array(action),
                    reward=jnp.array(0.0),
                    discount=jnp.array(0.99),
                    next_observation=jnp.array(next_state),
                )
            )
        data_set = brax_transitions

    return data_set


# load and convert data
# session_path = "/home/bhoffman/Documents/MT FS24/active-learning-dynamics/collected_data/test20240806-135621/session_buffer.pickle"
# session_path = "/home/bhoffman/Documents/MT FS24/active-learning-dynamics/collected_data/test20240815-151559/session_buffer.pickle"
# session_path = "/home/bhoffman/Documents/MT FS24/active-learning-dynamics/collected_data/test20240819-141455all/session_buffer.pickle"
# session_path = "/home/bhoffman/Documents/MT FS24/active-learning-dynamics/collected_data/test20240819-142443all_easy/session_buffer.pickle"
# session_path = "/home/bhoffman/Documents/MT FS24/active-learning-dynamics/collected_data/test20240830-111841_v1_1/session_buffer.pickle"
# session_path = "/home/bhoffman/Documents/MT FS24/active-learning-dynamics/collected_data/test20240830-112105_v1_2/session_buffer.pickle"
# session_path = "/home/bhoffman/Documents/MT FS24/active-learning-dynamics/collected_data/test20240830-112255_v1_3/session_buffer.pickle"

# session_path = "/home/bhoffman/Documents/MT FS24/active-learning-dynamics/collected_data/test20240903-131900_v2_0/session_buffer.pickle"
# session_path = "/home/bhoffman/Documents/MT FS24/active-learning-dynamics/collected_data/test20240903-132044_v2_1/session_buffer.pickle"
# session_path = "/home/bhoffman/Documents/MT FS24/active-learning-dynamics/collected_data/test20240903-132303_v2_2/session_buffer.pickle"
# session_path = "/home/bhoffman/Documents/MT FS24/active-learning-dynamics/collected_data/test20240903-132514_v2_3/session_buffer.pickle"

# session_path = "/home/bhoffman/Documents/MT FS24/active-learning-dynamics/collected_data/test20240904-153813_v3_1/session_buffer.pickle"
# session_path = "/home/bhoffman/Documents/MT FS24/active-learning-dynamics/collected_data/test20240904-154043_v3_2/session_buffer.pickle"
# session_path = "/home/bhoffman/Documents/MT FS24/active-learning-dynamics/collected_data/test20240904-154353_v3_3/session_buffer.pickle"
# session_path = "/home/bhoffman/Documents/MT FS24/active-learning-dynamics/collected_data/test20240904-155015_v3_4/session_buffer.pickle"

# session_path = "/home/bhoffman/Documents/MT FS24/active-learning-dynamics/collected_data/test20240906-171355_vonly_arm/session_buffer.pickle"
# session_path = "/home/bhoffman/Documents/MT FS24/active-learning-dynamics/collected_data/test20240906-171626_vonly_base_linear/session_buffer.pickle"
# session_path = "/home/bhoffman/Documents/MT FS24/active-learning-dynamics/collected_data/test20240906-171829_vonly_rotation/session_buffer.pickle"

# session_path = "/home/bhoffman/Documents/MT FS24/active-learning-dynamics/collected_data/test20240909-142029_v4_1/session_buffer.pickle"
# session_path = "/home/bhoffman/Documents/MT FS24/active-learning-dynamics/collected_data/test20240909-142230_v4_2/session_buffer.pickle"
# session_path = "/home/bhoffman/Documents/MT FS24/active-learning-dynamics/collected_data/test20240909-142535_v4_3/session_buffer.pickle"
# session_path = "/home/bhoffman/Documents/MT FS24/active-learning-dynamics/collected_data/test20240909-142945_v4_4/session_buffer.pickle"

# session_path = "/home/bhoffman/Documents/MT FS24/active-learning-dynamics/collected_data/test20241003-113919_vArm_rot_cw/session_buffer.pickle"

# session_path = "/home/bhoffman/Documents/MT FS24/active-learning-dynamics/collected_data/test20241104-174844_data_collection_new_v0/session_buffer.pickle"
# session_path = "/home/bhoffman/Documents/MT FS24/active-learning-dynamics/collected_data/test20241104-175451_data_collection_new_v1/session_buffer.pickle"
# session_path = "/home/bhoffman/Documents/MT FS24/active-learning-dynamics/collected_data/test20241104-180148_data_collection_new_v2/session_buffer.pickle"
# session_path = "/home/bhoffman/Documents/MT FS24/active-learning-dynamics/collected_data/test20241104-180443_data_collection_new_v3/session_buffer.pickle"
# session_path = "/home/bhoffman/Documents/MT FS24/active-learning-dynamics/collected_data/test20241104-180949_data_collection_new_v4/session_buffer.pickle"
# session_path = "/home/bhoffman/Documents/MT FS24/active-learning-dynamics/collected_data/test20241104-181100_data_collection_new_v4/session_buffer.pickle"
# session_path = "/home/bhoffman/Documents/MT FS24/active-learning-dynamics/collected_data/test20241104-181628_data_collection_new_v5/session_buffer.pickle"

session_paths = [
    # "/home/bhoffman/Documents/MT FS24/active-learning-dynamics/collected_data/test20241106-141659_data_collection_v0/session_buffer.pickle",
    # "/home/bhoffman/Documents/MT FS24/active-learning-dynamics/collected_data/test20241106-142210_data_collection_v1/session_buffer.pickle",
    # "/home/bhoffman/Documents/MT FS24/active-learning-dynamics/collected_data/test20241106-142743_data_collection_v2/session_buffer.pickle",
    # "/home/bhoffman/Documents/MT FS24/active-learning-dynamics/collected_data/test20241106-143309_data_collection_v3/session_buffer.pickle",
    # "/home/bhoffman/Documents/MT FS24/active-learning-dynamics/collected_data/test20241106-143832_data_collection_v4/session_buffer.pickle",
    # "/home/bhoffman/Documents/MT FS24/active-learning-dynamics/collected_data/test20241106-144336_data_collection_v5/session_buffer.pickle",
    # "/home/bhoffman/Documents/MT FS24/active-learning-dynamics/collected_data/test20241106-144831_data_collection_v6/session_buffer.pickle",
    # "/home/bhoffman/Documents/MT FS24/active-learning-dynamics/collected_data/test20241106-145352_data_collection_v7/session_buffer.pickle",
    # "/home/bhoffman/Documents/MT FS24/active-learning-dynamics/collected_data/test20241106-150012_data_collection_v8/session_buffer.pickle",
    "/home/bhoffman/Documents/MT FS24/active-learning-dynamics/collected_data/test20241113-170438_ee_ori_data_v0/session_buffer.pickle"
]

for session_path in session_paths:
    format = "brax"
    use_vision = False
    data_set_converted = convert_for_learning(
        file_path=session_path,
        as_euler=True,
        with_wrist=True,
        with_joints=False,
        skip_first=True,
        end_idx=None,
        encode_angles=True,
        format=format,
        use_vision=use_vision,
    )

    # export converted data
    output_path = "/home/bhoffman/Documents/MT_FS24/simulation_transfer/data/recordings_spot_ee_v2/"
    version = re.search(r"test(\d+)-(\d+)_ee_ori_data_v(\d+)", session_path).group(3)
    file_name = (
        f"brax_transitions_v{version}.pickle"
        if use_vision
        else f"brax_transitions_ee_ori_testing_odom_v{version}.pickle"
    )

    output_path = output_path + file_name
    with open(output_path, "wb") as file:
        pickle.dump(data_set_converted, file)
    print(f"Data set exported to {output_path}")
