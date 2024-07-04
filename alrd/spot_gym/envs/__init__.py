from alrd.spot_gym.envs.spot2d import Spot2DEnv
from alrd.spot_gym.model.robot_state import (
    SpotState,
    KinematicState,
    modify_2d_state,
)
from alrd.spot_gym.model.mobility_command import MobilityCommand
from alrd.spot_gym.envs.record import Session
from opax.models.bayesian_dynamics_model import BayesianDynamicsModel
from opax.utils.replay_buffer import Transition, ReplayBuffer
import numpy as np
from typing import List
from scipy.spatial.transform import Rotation as R
from alrd.utils.utils import Frame2D
