import pickle
import matplotlib.pyplot as plt
import numpy as np
import re
import json
import pandas as pd
from scipy.spatial.transform import Rotation as R
import ikpy.chain

from alrd.run_spot import SessionBuffer, DataBuffer, TransitionData, StateData, TimeData
from alrd.spot_gym.model.robot_state import SpotState


def create_vis_state_for_measurement(base_pos, quat, arm_joint_positions):
    return {
        "basePosition": {"x": base_pos[1], "y": base_pos[2], "z": base_pos[0]},
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


def create_vis_state_for_policy(base_pos, quat, arm_joint_positions):
    return {
        "basePosition": {"x": base_pos[1], "y": base_pos[2], "z": base_pos[0]},
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


def convert_measured_to_vis(file_path):
    """
    Converts measured trajectory from pickle file to json format for the visualization.

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
        state = create_vis_state_for_measurement(base_pose, quat, arm_joint_positions)

        states.append(state)

    return states


def convert_policy_to_vis(file_path):
    """
    Converts plicy-generated trajectory from pickle file to json format for the visualization.

    Args:
        file_path (str): The path to the pickle file.
    """
    import numpy as np

    def world_to_body_frame(base_pos, theta, ee_pos):
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        ee_x_local = cos_theta * (ee_pos[0] - base_pos[0]) + sin_theta * (
            ee_pos[1] - base_pos[1]
        )
        ee_y_local = -sin_theta * (ee_pos[0] - base_pos[0]) + cos_theta * (
            ee_pos[1] - base_pos[1]
        )
        ee_z_local = ee_pos[2]
        ee_body = [ee_x_local, ee_y_local, ee_z_local]
        return ee_body

    # convert to y-up helper
    def convert_quat(quat):
        x, y, z, w = quat
        return [y, z, x, w]

    def convert_heading_to_quat(heading):
        return R.from_euler("z", heading).as_quat().tolist()

    # load data
    with open(file_path, "rb") as file:
        states_data_all = pickle.load(file)

    states_data = states_data_all[20]

    # extract states
    states = []
    use_ik = True
    current_arm_joint_states = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    for step, state in enumerate(states_data):
        print(f"Processing step {step}/{len(states_data)}")
        base_pos = np.concatenate((state[0:2], [0.445]))
        quat = convert_quat(convert_heading_to_quat(state[2]))
        if use_ik:
            ee_target_world = np.array(state[3:6])
            ee_target_body = world_to_body_frame(base_pos, state[2], ee_target_world)
            ee_target_body = np.array(ee_target_body)
            # current_arm_joint_states = ik.calculate_ik(
            #     ee_target_body, current_arm_joint_states
            # )
            # deg to rad
            # current_arm_joint_states = np.deg2rad(current_arm_joint_states)
            ik_chain = ikpy.chain.Chain.from_urdf_file(
                "/home/bhoffman/Documents/MT FS24/active-learning-dynamics/alrd/spot_gym/utils/spot_urdf_model/spot_with_arm.urdf",
                base_elements=["base"],
                last_link_vector=[0.19557, 0, 0],
                active_links_mask=[
                    True,
                    True,
                    True,
                    True,
                    True,
                    False,
                    False,
                    False,
                    False,
                ],
            )
            current_arm_joint_states_pre = ik_chain.inverse_kinematics(ee_target_body)
            # between 0 and 2pi
            current_arm_joint_states_pre = np.mod(
                current_arm_joint_states_pre, 2 * np.pi
            )
            current_arm_joint_states = current_arm_joint_states_pre[1:7]
        else:
            current_arm_joint_states = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        state = create_vis_state_for_policy(base_pos, quat, current_arm_joint_states)
        states.append(state)

    return states


# load and convert data
# session_path = "/home/bhoffman/Documents/MT FS24/active-learning-dynamics/collected_data/test20240806-135621/session_buffer.pickle"
# session_path = "/home/bhoffman/Documents/MT FS24/active-learning-dynamics/collected_data/test20240807-154634/session_buffer.pickle"
# session_path = "/home/bhoffman/Documents/MT FS24/active-learning-dynamics/collected_data/test20240815-151559/session_buffer.pickle"

session_path = "/home/bhoffman/Documents/MT_FS24/simulation_transfer/results/policies_traj/trajectories_all.pkl"

# source = "measured"
source = "policy"

if source == "measured":
    states = convert_measured_to_vis(session_path)
elif source == "policy":
    states = convert_policy_to_vis(session_path)

# export converted data
output_path = "/home/bhoffman/Documents/MT FS24/spot_visualizer/data/collected data/"
if source == "measured":
    test_id = re.search(r"test\d{8}-\d{6}_v\d{1}_\d{1}", session_path).group(0)
elif source == "policy":
    test_id = "test_policy_new"
output_path = f"{output_path}dataset_vis_{source}_{test_id}.json"

with open(output_path, "w") as file:
    file.write(json.dumps(states, indent=4))
print(f"Data set exported to {output_path}")
