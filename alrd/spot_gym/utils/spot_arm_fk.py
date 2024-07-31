# TODO: test implementation

import casadi as cs
import numpy as np
from urdf2casadi import urdfparser as u2c


class SpotArmFK:
    def __init__(
        self,
        urdf_path: str = "/home/bhoffman/Documents/MT_FS24/active-learning-dynamics/alrd/spot_gym/model/spot_urdf_model/spot_with_arm.urdf",
        links: list = [
            "base",
            "arm0.link_sh0",
            "arm0.link_sh1",
            "arm0.link_el0",
            "arm0.link_el1",
            "arm0.link_wr0",
            "arm0.link_wr1",
        ],
    ):
        self.urdf_path = urdf_path
        self.robot_parser = u2c.URDFparser()
        self.robot_parser.from_file(urdf_path)
        self.links = links

    def get_T_fk(self, base_link_index: int = 0, end_link_index: int = 6):
        assert base_link_index >= 0, "base_link_index must be at least 0"
        assert end_link_index < len(
            self.links
        ), "end_link_index must be less than len(links)"
        assert (
            base_link_index < end_link_index
        ), "base_link_index must be less than end_link_index"

        fk_dict = self.robot_parser.get_forward_kinematics(
            self.links[base_link_index], self.links[end_link_index]
        )
        forward_kinematics = fk_dict["T_fk"]
        return forward_kinematics

    def get_ee_position(
        self, joint_states: list, base_link_index: int = 0, end_link_index: int = 6
    ):
        assert (
            len(joint_states) == end_link_index - base_link_index
        ), "Must provide {} joint states".format(end_link_index - base_link_index)

        q = cs.vertcat(*joint_states)
        forward_kinematics = self.get_T_fk()
        ee_position_pre = forward_kinematics(q)[:3, 3]

        # add final transformation aknowledging the wrist to hand offset
        # dimensions from: https://dev.bostondynamics.com/docs/concepts/arm/arm_concepts
        hand_offset = np.array([0.19557, 0, 0])
        ee_position = ee_position_pre + hand_offset
        return ee_position


if __name__ == "__main__":
    spot_arm_fk = SpotArmFK()
    joint_states = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    t = spot_arm_fk.get_T_fk()
    print(t)
    ee_position = spot_arm_fk.get_ee_position(joint_states)
    print(ee_position)
