import time
from typing import List, Tuple

import cv2
import numpy as np

from config import Config
from gestures import Gesture, HorizontalZone

_GESTURE_LABELS: dict = {
    Gesture.MOVE_LEFT: "LEFT",
    Gesture.MOVE_RIGHT: "RIGHT",
    Gesture.JUMP: "JUMP",
    Gesture.DUCK: "DUCK",
    Gesture.ROLL: "ROLL",
}

_ZONE_LINE_COLORS: dict = {
    HorizontalZone.LEFT: (80, 80, 255),
    HorizontalZone.CENTER: (80, 255, 80),
    HorizontalZone.RIGHT: (255, 80, 80),
}

_ZONE_TINT_COLORS: dict = {
    HorizontalZone.LEFT: (60, 60, 200),
    HorizontalZone.CENTER: None,
    HorizontalZone.RIGHT: (200, 60, 60),
}

_GESTURE_COLORS: dict = {
    Gesture.MOVE_LEFT: (0, 220, 255),
    Gesture.MOVE_RIGHT: (0, 220, 255),
    Gesture.JUMP: (0, 255, 120),
    Gesture.DUCK: (255, 120, 0),
    Gesture.ROLL: (220, 0, 255),
}


class Visualizer:
    def __init__(self, config: Config):
        self._config = config
        self._active_gestures: List[Tuple[Gesture, float]] = []
        self._fps_samples: List[float] = []
        self._last_frame_time: float = time.time()

    def tick_fps(self) -> float:
        now = time.time()
        elapsed = now - self._last_frame_time
        self._last_frame_time = now

        if elapsed > 0:
            self._fps_samples.append(1.0 / elapsed)
            if len(self._fps_samples) > self._config.fps_smoothing_window:
                self._fps_samples.pop(0)

        if not self._fps_samples:
            return 0.0
        return sum(self._fps_samples) / len(self._fps_samples)

    def register_gestures(self, gestures: List[Gesture]) -> None:
        now = time.time()
        cutoff = now - self._config.gesture_display_duration
        self._active_gestures = [
            (g, t) for g, t in self._active_gestures if t > cutoff
        ]
        for gesture in gestures:
            self._active_gestures.append((gesture, now))

    def render(
        self, frame: np.ndarray, zone: HorizontalZone, fps: float
    ) -> None:
        h, w = frame.shape[:2]
        left_x = int(w * self._config.left_zone_ratio)
        right_x = int(w * self._config.right_zone_ratio)

        self._render_zone_tint(frame, zone, w, h, left_x, right_x)
        self._render_zone_lines(frame, zone, h, left_x, right_x)
        self._render_zone_labels(frame, w, h, left_x, right_x)
        self._render_fps(frame, fps)
        self._render_active_gestures(frame)
        self._render_hints(frame, w, h)

    def _render_zone_tint(
        self,
        frame: np.ndarray,
        zone: HorizontalZone,
        w: int,
        h: int,
        left_x: int,
        right_x: int,
    ) -> None:
        tint = _ZONE_TINT_COLORS.get(zone)
        if tint is None:
            return
        overlay = frame.copy()
        if zone == HorizontalZone.LEFT:
            cv2.rectangle(overlay, (0, 0), (left_x, h), tint, -1)
        elif zone == HorizontalZone.RIGHT:
            cv2.rectangle(overlay, (right_x, 0), (w, h), tint, -1)
        cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)

    def _render_zone_lines(
        self,
        frame: np.ndarray,
        zone: HorizontalZone,
        h: int,
        left_x: int,
        right_x: int,
    ) -> None:
        color = _ZONE_LINE_COLORS.get(zone, (200, 200, 200))
        cv2.line(frame, (left_x, 0), (left_x, h), color, 2)
        cv2.line(frame, (right_x, 0), (right_x, h), color, 2)

    def _render_zone_labels(
        self, frame: np.ndarray, w: int, h: int, left_x: int, right_x: int
    ) -> None:
        labels = [
            ("LEFT", left_x // 2),
            ("CENTER", (left_x + right_x) // 2),
            ("RIGHT", (right_x + w) // 2),
        ]
        for label, x_pos in labels:
            text_x = x_pos - (len(label) * 7)
            cv2.putText(
                frame,
                label,
                (text_x, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 200, 200),
                1,
            )

    def _render_fps(self, frame: np.ndarray, fps: float) -> None:
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

    def _render_active_gestures(self, frame: np.ndarray) -> None:
        now = time.time()
        y = 75
        for gesture, triggered_at in self._active_gestures:
            age = now - triggered_at
            alpha = max(0.0, 1.0 - age / self._config.gesture_display_duration)
            base_color = _GESTURE_COLORS.get(gesture, (255, 255, 255))
            color = tuple(int(c * alpha) for c in base_color)
            label = _GESTURE_LABELS.get(gesture, "")
            cv2.putText(
                frame, label, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3
            )
            y += 50

    def _render_hints(self, frame: np.ndarray, w: int, h: int) -> None:
        hints = [
            "Lean left or right: Change lane",
            "Jump up: Jump",
            "Crouch forward: Slide",
            "Hands together: Roll",
            "Q: Quit",
        ]
        for i, hint in enumerate(hints):
            cv2.putText(
                frame,
                hint,
                (w - 330, h - 115 + i * 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (170, 170, 170),
                1,
            )
