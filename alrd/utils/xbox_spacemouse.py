import threading
import pyspacemouse
import numpy as np
from typing import Tuple
from alrd.utils.xbox.xbox_joystick_factory import XboxJoystickFactory


class XboxSpaceMouse:
    """
    This class provides an interface to the SpaceMouse and Xbox controller.
    It continuously reads the SpaceMouse and Xbox state and provides
    a "get_action" method to get the latest action and button state.
    """

    def __init__(self):
        pyspacemouse.open()
        self.joy = XboxJoystickFactory.get_joystick()

        self.state_lock = threading.Lock()
        self.latest_data = {
            "spacemouse_actions": np.zeros(6),
            "spacemouse_buttons": [0, 0],
            "xbox_actions": [0, 0, 0, 0, False, False],
        }
        # Start a thread to continuously read the SpaceMouse state
        self.thread = threading.Thread(target=self._read_controller)
        self.thread.daemon = True
        self.thread.start()

    def _read_controller(self):
        while True:
            spacemouse_state = pyspacemouse.read()
            xbox_state = (
                self.joy.left_x(),
                self.joy.left_y(),
                self.joy.right_x(),
                self.joy.right_y(),
                self.joy.left_trigger(),
                self.joy.right_trigger(),
            )
            with self.state_lock:
                self.latest_data["spacemouse_actions"] = np.array(
                    [
                        -spacemouse_state.y,
                        spacemouse_state.x,
                        spacemouse_state.z,
                        -spacemouse_state.roll,
                        -spacemouse_state.pitch,
                        -spacemouse_state.yaw,
                    ]
                )  # spacemouse axis matched with robot base frame
                self.latest_data["spacemouse_buttons"] = spacemouse_state.buttons
                self.latest_data["xbox_actions"] = xbox_state

    def get_action(self) -> Tuple[np.ndarray, list]:
        """Returns the latest action and button state of the SpaceMouse."""
        with self.state_lock:
            return (
                self.latest_data["spacemouse_actions"],
                self.latest_data["spacemouse_buttons"],
                self.latest_data["xbox_actions"],
            )


if __name__ == "__main__":

    controller = XboxSpaceMouse()

    try:
        while True:
            spacemouse_actions, spacemouse_buttons, xbox_actions = (
                controller.get_action()
            )
            print(
                f"Spacemouse Action: {spacemouse_actions}, Spacemouse Buttons: {spacemouse_buttons} Xbox Actions: {xbox_actions}"
            )
    except KeyboardInterrupt:
        print("done")
