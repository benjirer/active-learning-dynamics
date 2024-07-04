# test random sample generation from randomxbox agent
from alrd.agent.randomxbox import SpotRandomXbox
import matplotlib.pyplot as plt

agent = SpotRandomXbox()
arm_dq_series = agent.arm_dq_series
plt.plot(arm_dq_series)
plt.show()
