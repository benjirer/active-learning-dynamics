import torch
import numpy as np
from alrd.spot_gym.utils.spot_arm_fk import SpotArmFK


class SpotArmIK:
    def __init__(self):
        self.spot_arm_fk = SpotArmFK()

        # arm assets
        self.joint_q: np.array = np.zeros(6)
        self.current_ee_pose: np.array = np.zeros(3)
        self.ee_pose_b_target: np.array = np.zeros(3)

        self.jacobian: np.ndarray = np.zeros((3, 6))

        # configs
        self.ik_max_iter = 10
        self.ik_step_size = 0.4
        self.ik_tol = 0.01

    def calculate_ik(
        self,
        ee_target: np.array,
        current_arm_joint_states: np.array,
    ):

        # get joint positions
        joint_pos = torch.from_numpy(current_arm_joint_states)

        # set as class assets
        self.joint_q = joint_pos.numpy()
        self.ee_pose_b_target = ee_target

        # compute Jacobian
        max_iter = self.ik_max_iter
        for i in range(max_iter):
            curr_q = torch.from_numpy(self.joint_q).clone()

            # compute error
            self.compute_fk()
            curr_ee_pose = torch.from_numpy(self.current_ee_pose)

            curr_ee_pose_error = torch.from_numpy(self.ee_pose_b_target) - curr_ee_pose

            # calculate jacobian using finite difference based on current q and qd
            self.jacobian[:] = 0.0
            self.compute_fd_jacobian(curr_q)

            # calculate q delta
            dpdqj_compact = torch.from_numpy(self.jacobian)
            grad = torch.matmul(dpdqj_compact.transpose(0, 1), curr_ee_pose_error)
            delta_q = torch.linalg.solve(
                torch.matmul(dpdqj_compact.transpose(0, 1), dpdqj_compact)
                + 1e-6 * torch.eye(6),
                grad,
            ).squeeze(-1)

            # update q
            curr_q += self.ik_step_size * delta_q.flatten()
            self.joint_q = curr_q.numpy()

            if torch.all(torch.norm(curr_ee_pose_error, dim=-1) < self.ik_tol):
                break

        return self.joint_q

    def compute_fk(self):
        current_ee_pose_pre = self.spot_arm_fk.get_ee_position(self.joint_q)
        self.current_ee_pose = current_ee_pose_pre.full().flatten()

    def compute_fd_jacobian(self, curr_q, eps=1e-4):
        # use auto differentiation
        for i in range(6):  # all 6 joints
            q = curr_q.clone()

            q[i] += eps
            self.joint_q = q.numpy()
            self.compute_fk()
            f_plus = torch.from_numpy(self.current_ee_pose).clone()
            q[i] -= 2 * eps
            self.joint_q = q.numpy()
            self.compute_fk()
            f_minus = torch.from_numpy(self.current_ee_pose).clone()

            self.jacobian[:, i] = (f_plus - f_minus) / (2 * eps)

        # restore q
        self.joint_q = curr_q.numpy()


if __name__ == "__main__":
    spot_arm_ik = SpotArmIK()
    ee_target = np.array([0.8, 0.1, 0.1])
    current_arm_joint_states = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    ik_joint_q = spot_arm_ik.calculate_ik(ee_target, current_arm_joint_states)

    print(ik_joint_q)

    fk_check = spot_arm_ik.spot_arm_fk.get_ee_position(ik_joint_q)
    print(fk_check)
