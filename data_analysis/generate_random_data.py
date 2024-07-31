import json
import random


def generate_random_states(num_states, num_joints=6):
    states = []
    base_x, base_y, base_z = 0.0, 0.0, 0.0
    base_w, base_x_rot, base_y_rot, base_z_rot = 1.0, 0.0, 0.0, 0.0
    joint_positions = [0.0] * num_joints

    for i in range(num_states):
        # Small random variations
        # base_x += round(random.uniform(0.01, 0.1), 2)
        # base_z_rot += round(random.uniform(0.01, 0.02), 2)
        # base_z_rot = min(base_z_rot, 1.0)  # Ensure base_z_rot does not exceed 1
        # base_w = round((1 - (base_z_rot**2)) ** 0.5, 4)

        # keep base position and orientation constant
        joint_positions[18] += round(random.uniform(0.00, 0.01), 2)
        state = {
            "basePosition": {"x": round(base_x, 2), "y": base_y, "z": base_z},
            "baseOrientation": {
                "w": round(base_w, 4),
                "x": base_x_rot,
                "y": base_y_rot,
                "z": round(base_z_rot, 2),
            },
            "jointStates": [
                {
                    "jointPos": round(joint_positions[j], 2),
                }
                for j in range(num_joints)
            ],
        }
        states.append(state)

    return states


num_states = 10000
num_joints = 19
states = generate_random_states(num_states, num_joints)
states_json = json.dumps(states, indent=4)
with open(
    "/home/bhoffman/Documents/MT_FS24/spot_visualizer/data/collected data/states.json",
    "w",
) as file:
    file.write(states_json)
