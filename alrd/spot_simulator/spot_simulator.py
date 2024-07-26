import numpy as np


class SpotSimulator:
    def __init__(self, b: np.ndarray):
        """
        Args:
            b (np.ndarray): The learned parameters for the state transition model.
        """
        self.b = np.array(b, dtype=np.float32)

    def step(self, current_state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """
        Predict the next state given the current state and action.

        Args:
            current_state (np.ndarray): The current state of the system.
            action (np.ndarray): The action taken.

        Returns:
            np.ndarray: The predicted next state.
        """
        current_state = np.array(current_state, dtype=np.float32)
        action = np.array(action, dtype=np.float32)

        next_state = current_state + np.dot(action, self.b.T)
        return next_state


# Example usage
if __name__ == "__main__":
    # Example parameter b (assuming it has been trained already)
    b = np.random.rand(10, 3)  # Replace with the actual trained b

    # Initialize the simulator
    simulator = SpotSimulator(b)

    # Example current state and action
    current_state = np.random.rand(1, 10)  # Replace with actual current state
    action = np.random.rand(1, 3)  # Replace with actual action

    # Predict the next state
    next_state = simulator.step(current_state, action)
    print("Predicted next state:", next_state)
