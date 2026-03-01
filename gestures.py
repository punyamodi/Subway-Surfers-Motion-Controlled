import time
from enum import Enum, auto
from math import hypot
from typing import List, Optional

from config import Config
from detector import PoseLandmarks


class Gesture(Enum):
    MOVE_LEFT = auto()
    MOVE_RIGHT = auto()
    JUMP = auto()
    DUCK = auto()
    ROLL = auto()


class HorizontalZone(Enum):
    LEFT = 0
    CENTER = 1
    RIGHT = 2


class GestureEngine:
    def __init__(self, config: Config):
        self._config = config
        self._previous_zone = HorizontalZone.CENTER
        self._previous_shoulder_y: float = 0.0
        self._last_shoulder_sample_time: float = 0.0
        self._last_jump_time: float = 0.0
        self._last_duck_time: float = 0.0
        self._last_roll_time: float = 0.0

    @property
    def current_zone(self) -> HorizontalZone:
        return self._previous_zone

    def detect(
        self, landmarks: PoseLandmarks, frame_width: int, frame_height: int
    ) -> List[Gesture]:
        gestures: List[Gesture] = []

        horizontal = self._evaluate_horizontal(landmarks, frame_width)
        if horizontal is not None:
            gestures.append(horizontal)

        if self._evaluate_jump(landmarks):
            gestures.append(Gesture.JUMP)

        if self._evaluate_duck(landmarks):
            gestures.append(Gesture.DUCK)

        if self._evaluate_roll(landmarks):
            gestures.append(Gesture.ROLL)

        return gestures

    def _evaluate_horizontal(
        self, landmarks: PoseLandmarks, frame_width: int
    ) -> Optional[Gesture]:
        shoulder_mid_x = (
            landmarks.left_shoulder[0] + landmarks.right_shoulder[0]
        ) / 2.0

        left_threshold = frame_width * self._config.left_zone_ratio
        right_threshold = frame_width * self._config.right_zone_ratio

        if shoulder_mid_x <= left_threshold:
            zone = HorizontalZone.LEFT
        elif shoulder_mid_x >= right_threshold:
            zone = HorizontalZone.RIGHT
        else:
            zone = HorizontalZone.CENTER

        gesture: Optional[Gesture] = None
        prev = self._previous_zone

        if prev != zone:
            if zone in (HorizontalZone.RIGHT,) and prev in (
                HorizontalZone.CENTER,
                HorizontalZone.LEFT,
            ):
                gesture = Gesture.MOVE_RIGHT
            elif zone in (HorizontalZone.LEFT,) and prev in (
                HorizontalZone.CENTER,
                HorizontalZone.RIGHT,
            ):
                gesture = Gesture.MOVE_LEFT
            elif zone == HorizontalZone.CENTER:
                if prev == HorizontalZone.LEFT:
                    gesture = Gesture.MOVE_RIGHT
                elif prev == HorizontalZone.RIGHT:
                    gesture = Gesture.MOVE_LEFT

        self._previous_zone = zone
        return gesture

    def _evaluate_jump(self, landmarks: PoseLandmarks) -> bool:
        shoulder_mid_y = (
            landmarks.left_shoulder[1] + landmarks.right_shoulder[1]
        ) / 2.0
        now = time.time()

        triggered = False

        if self._previous_shoulder_y > 0:
            upward_movement = self._previous_shoulder_y - shoulder_mid_y
            cooldown_ok = (now - self._last_jump_time) > self._config.jump_cooldown

            if (
                upward_movement > self._config.jump_shoulder_threshold
                and cooldown_ok
            ):
                self._last_jump_time = now
                triggered = True

        if now - self._last_shoulder_sample_time > self._config.shoulder_sample_interval:
            self._previous_shoulder_y = shoulder_mid_y
            self._last_shoulder_sample_time = now

        return triggered

    def _evaluate_duck(self, landmarks: PoseLandmarks) -> bool:
        now = time.time()
        if (now - self._last_duck_time) <= self._config.duck_cooldown:
            return False

        shoulder_mid = (
            (landmarks.left_shoulder[0] + landmarks.right_shoulder[0]) / 2.0,
            (landmarks.left_shoulder[1] + landmarks.right_shoulder[1]) / 2.0,
        )
        hip_mid = (
            (landmarks.left_hip[0] + landmarks.right_hip[0]) / 2.0,
            (landmarks.left_hip[1] + landmarks.right_hip[1]) / 2.0,
        )

        torso_length = hypot(
            shoulder_mid[0] - hip_mid[0], shoulder_mid[1] - hip_mid[1]
        )

        if torso_length < 1.0:
            return False

        nose_to_hip = hypot(
            landmarks.nose[0] - hip_mid[0], landmarks.nose[1] - hip_mid[1]
        )

        ratio = nose_to_hip / torso_length

        if ratio < self._config.duck_body_ratio:
            self._last_duck_time = now
            return True

        return False

    def _evaluate_roll(self, landmarks: PoseLandmarks) -> bool:
        now = time.time()
        if (now - self._last_roll_time) <= self._config.roll_cooldown:
            return False

        wrist_distance = hypot(
            landmarks.left_wrist[0] - landmarks.right_wrist[0],
            landmarks.left_wrist[1] - landmarks.right_wrist[1],
        )

        if wrist_distance < self._config.hands_joined_threshold:
            self._last_roll_time = now
            return True

        return False
