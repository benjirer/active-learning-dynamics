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
        self.ik_step_size = 0.5

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
            curr_q_pre = torch.from_numpy(self.joint_q)
            curr_q = curr_q_pre.clone()

            # compute error
            self.compute_fk()
            curr_ee_pose = [torch.from_numpy(self.current_ee_pose)]

            curr_ee_pose_error = torch.from_numpy(self.ee_pose_b_target) - curr_ee_pose

            # calculate jacobian using finite difference based on current q and qd
            self.jacobian[:] = 0.0
            self.compute_fd_jacobian(curr_q)

            # calculate q delta
            dpdqj_compact = self.jacobian.view(self.num_envs, -1, self.num_joints)
            curr_ee_pose_error_compact = curr_ee_pose_error.view(self.num_envs, -1)
            grad = torch.matmul(
                dpdqj_compact.transpose(1, 2), curr_ee_pose_error_compact.unsqueeze(-1)
            )
            delta_q = torch.linalg.solve(
                torch.matmul(dpdqj_compact.transpose(1, 2), dpdqj_compact)
                + 1e-6 * torch.eye(self.num_joints, device=self.device),
                grad,
            ).squeeze(-1)

            # update q
            curr_q[self.ghost_joint_idx] += self.ik_step_size * delta_q.flatten()
            self.ghost.joint_q.assign(wp.from_torch(curr_q))

            if torch.all(torch.norm(curr_ee_pose_error, dim=-1) < self.cfg.ik_tol):
                break

        joint_pos_ = curr_q[self.ghost_joint_idx].view(self.num_envs, -1)
        self.nominal_joint_pos = torch.cat(
            [
                joint_pos_[:, 0:1],
                joint_pos_[:, 3:4],
                joint_pos_[:, 6:7],
                joint_pos_[:, 9:10],
                joint_pos_[:, 1:2],
                joint_pos_[:, 4:5],
                joint_pos_[:, 7:8],
                joint_pos_[:, 10:11],
                joint_pos_[:, 2:3],
                joint_pos_[:, 5:6],
                joint_pos_[:, 8:9],
                joint_pos_[:, 11:12],
            ],
            dim=1,
        )

        # TODO: What does this do?

        for i in range(self.num_feet):
            self.ee_pose_w_ik[:, i, :] = math_utils.quat_rotate(
                math_utils.yaw_quat(asset.data.root_quat_w), curr_ee_pose[:, i, :]
            )
        self.ee_pose_w_ik[:, :, 0:2] += asset.data.root_pos_w[:, 0:2].unsqueeze(1)

    def compute_fk(self):
        self.current_ee_pose = self.spot_arm_fk.get_ee_position(self.joint_q)

    def compute_fd_jacobian(self, curr_q, eps=1e-4):
        # use auto differentiation
        for i in range(6):  # all 6 joints
            q = curr_q.clone()

            q[i] += eps
            self.joint_q.assign(q.numpy())
            self.compute_fk()
            f_plus = [torch.from_numpy(self.current_ee_pose).clone()]
            q[i] -= 2 * eps
            self.joint_q.assign(q.numpy())
            self.compute_fk()
            f_minus = [torch.from_numpy(self.current_ee_pose).clone()]

            self.jacobian[:, i] = (f_plus - f_minus) / (2 * eps)

        # restore q
        self.joint_q.assign(curr_q.numpy())
