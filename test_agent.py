from alrd.agent.keyboard import KeyboardAgent
from alrd.agent.xbox import SpotXbox2D


# Initialize the KeyboardAgent with example parameters
xy_speed = 1.0  # Example speed for x and y movements
a_speed = 0.1  # Example speed for angular movements
noangle = False  # Whether to disable angle movements

agent = SpotXbox2D(base_speed=xy_speed, base_angular=a_speed)

# Print the description of controls
# print(agent.description())

# Assuming you have some kind of loop to continually check for actions
try:
    while True:
        # This would be your observation from the environment
        obs = None

        # Get the action from the agent
        action = agent.act(obs)

        # Here you would normally pass the action to your environment
        # For example:
        # env.step(action)

        # Print the action for demonstration purposes
        print("Action taken:", action)
except KeyboardInterrupt:
    print("Exiting...")
