from typing import List

import pyautogui

from gestures import Gesture

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

_GESTURE_KEY_MAP = {
    Gesture.MOVE_LEFT: "left",
    Gesture.MOVE_RIGHT: "right",
    Gesture.JUMP: "up",
    Gesture.DUCK: "down",
    Gesture.ROLL: "space",
}


class InputController:
    def dispatch(self, gestures: List[Gesture]) -> None:
        for gesture in gestures:
            key = _GESTURE_KEY_MAP.get(gesture)
            if key is not None:
                pyautogui.press(key)
