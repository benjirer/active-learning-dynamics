from alrd.environment.env import VelocityControlEnv
from alrd.ui import KeyboardListener
from alrd.agent.absagent import Agent
import numpy as np
from alrd.utils import rotate_2d_vector

class KeyboardAgent(Agent):
    def __init__(self, xy_speed, a_speed, noangle=False) -> None:
        self.listener = KeyboardListener()
        self.xy_speed = xy_speed
        self.a_speed = a_speed
        self.cmds = {
            'w': (1, 0, 0),
            'a': (0, -1, 0),
            's': (-1, 0, 0),
            'd': (0, 1, 0),
            'q': (0, 0, -1),
            'e': (0, 0, 1)
        }
        self.noangle = noangle

    def act(self, obs):
        pressed = list(self.listener.which_pressed(self.cmds.keys()))
        action = np.zeros(3)
        for key in pressed:
            action += self.cmds[key]
        norm = np.linalg.norm(action[:2])
        if norm > 1e-5:
            action[:2] = self.xy_speed * action[:2] / norm
        action[2] = self.a_speed * action[2]
        if self.noangle:
            action = action[:2]
        return action

class KeyboardGPAgent(Agent):
    def __init__(self, gp_agent, xy_speed, a_speed) -> None:
        super().__init__()
        self.gp_agent = gp_agent
        self.kb_agent = KeyboardAgent(xy_speed, a_speed)

    def act(self, obs):
        pressed = list(self.kb_agent.listener.which_pressed(['r']))
        if len(pressed) > 0:
            action = self.gp_agent.act(obs)
        else:
            action = self.kb_agent.act(obs)
        return action