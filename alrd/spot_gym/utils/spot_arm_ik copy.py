import torch
import numpy as np
from alrd.spot_gym.utils.spot_arm_fk import SpotArmFK


class SpotArmIK:
    def __init__(self):
        self.spot_arm_fk = SpotArmFK()

        # arm assets
        self.joint_q: np.array = None
        self.joint_qd: np.array = None
        self.current_ee_pose: np.array = None

        self.jacobian: np.array = None

        # configs
        self.max_iter = 100
        self.step_size = 0.1
        self.tolerance = 0.1  # [m]

    def calculate_fk(self, q):
        ee_position = self.spot_arm_fk.get_ee_position(q.detach().numpy())
        ee_position_torch = torch.from_numpy(ee_position.full().flatten())
        return ee_position_torch

    def calculate_ik(
        self,
        ee_target: np.array,
        current_arm_joint_states: np.array,
    ):
        ee_target_torch = torch.from_numpy(ee_target)
        q = torch.from_numpy(current_arm_joint_states)

        for i in range(self.max_iter):
            current_position_torch = self.calculate_fk(q)
            error = ee_target_torch - current_position_torch

            jacobian = torch.autograd.functional.jacobian(self.calculate_fk, q)
            print(jacobian)

            jacobian_pinv = torch.linalg.pinv(jacobian)

            q_update = torch.matmul(jacobian_pinv, error)
            q = q + self.step_size * q_update

            if torch.norm(error) < self.tolerance:
                break

        return q.numpy()


if __name__ == "__main__":
    # test ik
    spot_arm_ik = SpotArmIK()
    ee_target = np.array([0.5, 0.5, 0.5])
    current_arm_joint_states = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    q = spot_arm_ik.calculate_ik(ee_target, current_arm_joint_states)
    print(q)
