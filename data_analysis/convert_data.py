import pickle
import matplotlib.pyplot as plt
import numpy as np
from alrd.run_spot import SessionBuffer, DataBuffer, TransitionData, StateData
from alrd.spot_gym.model.robot_state import SpotState
import json
from scipy.spatial.transform import Rotation as R


def convert_quat(quat):
    x, y, z, w = quat
    q = R.from_quat([x, y, z, w])
    q_rot = R.from_euler("x", -np.pi / 2, degrees=False)
    q_new = q * q_rot
    x_new, y_new, z_new, w_new = q_new.as_quat()
    return [x_new, y_new, z_new, w_new]


def convert_data(file_path):
    """
    Converts data from pickel file to json file for the visualization.
    """

    # load data
    with open(file_path, "rb") as file:
        data = pickle.load(file)

    # extract data
    states_data = data.data_buffers[0].states

    # extract states
    states = []
    for state in states_data:
        next_state = state.next_state
        # base_pose = convert_pose(last_state.pose_of_body_in_odom)
        base_pose = next_state.pose_of_body_in_vision
        quat = base_pose[3:]
        # quat = convert_quat(quat_pre)
        arm_joint_positions = next_state.arm_joint_positions
        state = {
            "basePosition": {"x": base_pose[0], "y": base_pose[2], "z": -base_pose[1]},
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


file_path = "/home/bhoffman/Documents/MT FS24/active-learning-dynamics/collected_data/test20240730-154452/session_buffer.pickle"
states = convert_data(file_path)
states_json = json.dumps(states, indent=4)
with open(
    "/home/bhoffman/Documents/MT FS24/spot_visualizer/data/collected data/states.json",
    "w",
) as file:
    file.write(states_json)
