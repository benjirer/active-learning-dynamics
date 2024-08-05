import numpy as np
from alrd.spot_gym.utils.spot_arm_ik import SpotArmIK


class SpotSimulator:
    def __init__(self, b: np.ndarray):
        """
        Args:
            b (np.ndarray): The fitted transition parameters.
        """
        self.b = np.array(b, dtype=np.float32)
        self.spot_arm_ik = SpotArmIK()

    def step(self, current_state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """
        Predict the next state given the current state and action.
        Use IK to calulculate joint positions from end effector positions for each state.

        Args:
            current_state (np.ndarray): The current state of the system.
            action (np.ndarray): The action taken.

        State space:
            base_x              - frame: vision
            base_y              - frame: vision
            heading             - frame: vision
            base_vx             - frame: vision
            base_vy             - frame: vision
            base_vrot           - frame: vision
            ee_x                - frame: body
            ee_y                - frame: body
            ee_z                - frame: body
            ee_rx               - frame: body
            ee_ry               - frame: body
            ee_rz               - frame: body
            ee_vx               - frame: body
            ee_vy               - frame: body
            ee_vz               - frame: body
            ee_vrx              - frame: body
            ee_vry              - frame: body
            ee_vrz              - frame: body
            arm_joint_positions - frame: none

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


        Returns:
            np.ndarray: The predicted next state.
        """
        current_state = np.array(current_state, dtype=np.float32)
        current_state_no_joints = current_state[:18]
        action = np.array(action, dtype=np.float32)

        next_state_no_joints = current_state_no_joints + np.dot(action, self.b.T)

        # Calculate joint positions from end effector positions
        ee_x, ee_y, ee_z = next_state_no_joints[6:9]
        ee_target = np.array([ee_x, ee_y, ee_z])

        joint_positions = self.spot_arm_ik.calculate_ik(ee_target, current_state[18:])

        next_state = np.concatenate((next_state_no_joints, joint_positions))

        return next_state


if __name__ == "__main__":
    # Load transition parameter b
    b = np.load(
        "/home/bhoffman/Documents/MT FS24/active-learning-dynamics/alrd/spot_simulator/transition_parameters/b_20240731-113113.npy"
    )

    # Initialize simulator
    simulator = SpotSimulator(b)

    # Predict next state
    current_state = np.random.rand(1, 10)
    action = np.random.rand(1, 3)
    next_state = simulator.step(current_state, action)
    print("Predicted next state:", next_state)
