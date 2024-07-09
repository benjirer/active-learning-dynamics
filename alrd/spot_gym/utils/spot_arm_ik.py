import torch
import numpy as np
from alrd.spot_gym.utils.spot_arm_fk import SpotArmFK


class SpotArmIK:
    def __init__(self):
        spot_arm_fk = SpotArmFK()
        self.forward_kinematics = spot_arm_fk.get_T_fk()

        # arm assets
        self.joint_q: np.array = None
        self.joint_qd: np.array = None
        self.current_ee_pose: np.array = None

        self.jacobian: np.array = None

        # configs
        self.ik_max_iter = 10
        self.ik_step_size = 0.5

    def calculate_ik(
        self,
        ee_target: np.array,
        current_arm_joint_states: np.array,
    ):
        # save current
        # asset = self.scene[self.asset_name]

        # if self.cfg.base_impedance_mode == "none":
        #     # sync the base pose, no tracking at all
        #     root_pos = asset.data.root_pos_w.clone()
        #     root_pos[:, :2] = 0.0
        #     rp_quat = math_utils.roll_pitch_quat(asset.data.root_quat_w)
        #     root_quat = torch.cat([rp_quat[:, 1:4], rp_quat[:, 0:1]], dim=1)

        # elif self.cfg.base_impedance_mode == "hard":
        #     # hard tracking of base pose
        #     root_pos = asset.data.root_pos_w.clone()
        #     root_pos[:, :2] = 0.0
        #     root_pos[:, 2] = self.cfg.base_height_target

        #     rp_quat = math_utils.quat_from_euler_xy(
        #         self.base_roll_target, self.base_pitch_target
        #     )
        #     root_quat = torch.cat([rp_quat[:, 1:4], rp_quat[:, 0:1]], dim=1)

        # elif self.cfg.base_impedance_mode == "soft":
        #     raise NotImplementedError(
        #         "Soft base impedance mode is not implemented yet."
        #     )

        # else:
        #     raise ValueError(
        #         f"Unknown base impedance mode: {self.cfg.base_impedance_mode}"
        #     )

        # joint_pos = torch.cat(
        #     [
        #         asset.data.joint_pos[:, 3:4],
        #         asset.data.joint_pos[:, 7:8],
        #         asset.data.joint_pos[:, 11:12],
        #         asset.data.joint_pos[:, 2:3],
        #         asset.data.joint_pos[:, 6:7],
        #         asset.data.joint_pos[:, 10:11],
        #         asset.data.joint_pos[:, 1:2],
        #         asset.data.joint_pos[:, 5:6],
        #         asset.data.joint_pos[:, 9:10],
        #         asset.data.joint_pos[:, 0:1],
        #         asset.data.joint_pos[:, 4:5],
        #         asset.data.joint_pos[:, 8:9],
        #     ],
        #     dim=1,
        # )

        # get joint positions
        joint_pos = torch.from_numpy(current_arm_joint_states)

        # set as class assets
        self.joint_q = joint_pos.numpy()
        self.joint_qd = np.zeros_like(self.joint_qd)

        # compute ghost Jacobian
        max_iter = self.ik_max_iter
        for i in range(max_iter):
            curr_q_pre = torch.from_numpy(self.joint_q)
            curr_q_pre.requires_grad_()
            curr_q = curr_q_pre.clone()

            # compute error
            self.compute_fk()
            # TODO: is this a pose or just a position?
            curr_ee_pose = [
                wp.to_torch(self.ghost_ee_pos_wp[i]).unsqueeze(1)
                for i in range(self.ghost_num_feet)
            ]

            curr_ee_pose = torch.cat(curr_ee_pose, dim=1)
            curr_ee_pose_error = self.ee_pose_b_target - curr_ee_pose

            # calculate jacobian using finite difference based on current q and qd
            self.jacobian[:] = 0.0
            self.compute_fd_jacobian(curr_q)

            # calculate q delta
            # TODO: what is envs and device?
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
        # wp.sim.eval_fk(
        #     self.ghost, self.ghost.joint_q, self.ghost.joint_qd, None, self.ghost_state
        # )
        # wp.launch(
        #     compute_ee_position,
        #     dim=self.num_envs,
        #     inputs=[self.ghost_state.body_q, self.ghost_num_links],
        #     outputs=[
        #         self.ghost_ee_pos_wp[0],
        #         self.ghost_ee_pos_wp[1],
        #         self.ghost_ee_pos_wp[2],
        #         self.ghost_ee_pos_wp[3],
        #     ],
        #     device=self.device,
        # )

        self.current_ee_pose = self.forward_kinematics(self.joint_q)

    def compute_fd_jacobian(self, curr_q, eps=1e-4):
        # use auto differentiation
        for i in range(3):  # hip, thigh, calf
            q = curr_q.clone()

            q[self.ghost_ik_df_idx + i] += eps
            self.ghost.joint_q.assign(wp.from_torch(q))
            self.compute_fk()
            f_plus = [
                wp.to_torch(self.ghost_ee_pos_wp[i]).clone()
                for i in range(self.ghost_num_feet)
            ]
            q[self.ghost_ik_df_idx + i] -= 2 * eps
            self.ghost.joint_q.assign(wp.from_torch(q))
            self.compute_fk()
            f_minus = [
                wp.to_torch(self.ghost_ee_pos_wp[i]).clone()
                for i in range(self.ghost_num_feet)
            ]

            for j in range(self.ghost_num_feet):
                self.jacobian[:, j, :, 3 * j + i] = (f_plus[j] - f_minus[j]) / (2 * eps)

        # restore q
        self.ghost.joint_q.assign(wp.from_torch(curr_q))
