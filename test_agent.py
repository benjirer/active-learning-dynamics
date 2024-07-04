import casadi as cs
import numpy as np
from urdf2casadi import urdfparser as u2c

urdf_path = "/home/bhoffman/Desktop/hab_spot_arm/urdf/hab_spot_arm.urdf"

links = [
    "arm0.link_sh0",
    "arm0.link_sh1",
    "arm0.link_el0",
    "arm0.link_el1",
    "arm0.link_wr0",
    "arm0.link_wr1",
]

joint_states = [0, 0, 0, 0, 0, 0]  # sh0  # sh1  # el0  # el1  # wr0  # wr1

robot_parser = u2c.URDFparser()
robot_parser.from_file(urdf_path)

# loop over all links and plot each time in the same 3d plot

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

for i in range(len(links) - 1):
    root_link = links[i]
    end_link = links[i + 1]
    fk_dict = robot_parser.get_forward_kinematics(root_link, end_link)
    forward_kinematics = fk_dict["T_fk"]
    curr_joint_states = joint_states[i : i + 1]
    ef_pos_as_quaternions = forward_kinematics(curr_joint_states)
    print(ef_pos_as_quaternions)

plt.show()
