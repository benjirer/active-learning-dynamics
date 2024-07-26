import pickle
import matplotlib.pyplot as plt
import numpy as np
from alrd.run_spot import SessionBuffer, DataBuffer, TransitionData, StateData
from alrd.spot_gym.model.robot_state import SpotState
import json
from scipy.spatial.transform import Rotation as R


def z_up_to_y_up_pose(pose):
    x, y, z, qw, qx, qz, qy = pose
    R_z_to_y = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    R_z_90 = R.from_euler("z", 90, degrees=True).as_matrix()

    position_zup = np.array([x, y, z])
    position_yup = R_z_to_y @ position_zup

    quaternion_zup = [qw, qx, qy, qz]
    r_yup = R.from_matrix(
        R_z_90 @ (R_z_to_y @ R.from_quat(quaternion_zup).as_matrix() @ R_z_to_y.T)
    )
    quaternion_yup = r_yup.as_quat()

    return np.concatenate(
        (
            position_yup,
            [
                quaternion_yup[3],
                quaternion_yup[0],
                quaternion_yup[2],
                quaternion_yup[1],
            ],
        )
    )


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
        last_state = state.last_state
        base_pose = z_up_to_y_up_pose(last_state.pose_of_body_in_odom)
        arm_joint_positions = last_state.arm_joint_positions
        state = {
            "basePosition": {"x": base_pose[0], "y": base_pose[1], "z": base_pose[2]},
            "baseOrientation": {
                "w": base_pose[3],
                "x": base_pose[4],
                "y": base_pose[5],
                "z": base_pose[6],
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


file_path = "/home/bhoffman/Documents/MT FS24/active-learning-dynamics/collected_data/test20240725-170625/session_buffer.pickle"
states = convert_data(file_path)
states_json = json.dumps(states, indent=4)
with open(
    "/home/bhoffman/Documents/MT FS24/spot_visualizer/data/collected data/states.json",
    "w",
) as file:
    file.write(states_json)
