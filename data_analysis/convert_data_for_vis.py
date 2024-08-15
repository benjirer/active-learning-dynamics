import pickle
import matplotlib.pyplot as plt
import numpy as np
import re
import json
from scipy.spatial.transform import Rotation as R

from alrd.run_spot import SessionBuffer, DataBuffer, TransitionData, StateData, TimeData
from alrd.spot_gym.model.robot_state import SpotState


def convert_for_vis(file_path):
    """
    Converts collected data from pickle file to json format for the visualization.

    Args:
        file_path (str): The path to the pickle file.
    """

    # convert to y-up helper
    def convert_quat(quat):
        x, y, z, w = quat
        return [y, z, x, w]

    # load data
    with open(file_path, "rb") as file:
        data = pickle.load(file)

    # extract data
    states_data = data.data_buffers[0].states

    # extract states
    states = []
    for state in states_data:
        next_state = state.next_state
        base_pose = next_state.pose_of_body_in_vision
        quat_pre = base_pose[3:]
        quat = convert_quat(quat_pre)
        arm_joint_positions = next_state.arm_joint_positions
        state = {
            "basePosition": {"x": base_pose[1], "y": base_pose[2], "z": base_pose[0]},
            "baseOrientation": {
                "w": quat[3],
                "x": quat[0],
                "y": quat[1],
                "z": quat[2],
            },
            "jointStates": [
                {
                    "jointPos": arm_joint_positions[0],  # sh0
                },
                {
                    "jointPos": 0.0,  # hip
                },
                {
                    "jointPos": 0.0,  # hip
                },
                {
                    "jointPos": 0.0,  # hip
                },
                {
                    "jointPos": 0.0,  # hip
                },
                {
                    "jointPos": arm_joint_positions[1],  # sh1
                },
                {
                    "jointPos": 0.0,  # leg
                },
                {
                    "jointPos": 0.0,  # leg
                },
                {
                    "jointPos": 0.0,  # leg
                },
                {
                    "jointPos": 0.0,  # leg
                },
                {
                    "jointPos": arm_joint_positions[2],  # el0
                },
                {
                    "jointPos": 0.0,  # knee
                },
                {
                    "jointPos": 0.0,  # knee
                },
                {
                    "jointPos": 0.0,  # knee
                },
                {
                    "jointPos": 0.0,  # knee
                },
                {
                    "jointPos": arm_joint_positions[3],  # el1
                },
                {
                    "jointPos": arm_joint_positions[4],  # wr0
                },
                {
                    "jointPos": arm_joint_positions[5],  # wr1
                },
                {
                    "jointPos": 0.0,  # gripper
                },
            ],
        }

        states.append(state)

    return states


# load and convert data
# session_path = "/home/bhoffman/Documents/MT FS24/active-learning-dynamics/collected_data/test20240806-135621/session_buffer.pickle"
session_path = "/home/bhoffman/Documents/MT FS24/active-learning-dynamics/collected_data/test20240807-154634/session_buffer.pickle"
states = convert_for_vis(session_path)

# export converted data
output_path = "/home/bhoffman/Documents/MT FS24/spot_visualizer/data/collected data/"
timestamp = re.search(r"test(\d{8}-\d{6})/", session_path).group(1)
output_path = f"{output_path}dataset_vis_{timestamp}_{pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')}.pickle"
with open(output_path, "w") as file:
    file.write(json.dumps(states, indent=4))
print(f"Data set exported to {output_path}")
