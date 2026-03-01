from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


@dataclass
class PoseLandmarks:
    nose: Tuple[float, float]
    left_shoulder: Tuple[float, float]
    right_shoulder: Tuple[float, float]
    left_wrist: Tuple[float, float]
    right_wrist: Tuple[float, float]
    left_hip: Tuple[float, float]
    right_hip: Tuple[float, float]
    left_knee: Tuple[float, float]
    right_knee: Tuple[float, float]
    raw_results: object


class PoseDetector:
    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1,
    ):
        self._mp_pose = mp.solutions.pose
        self._mp_draw = mp.solutions.drawing_utils
        self._pose = self._mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=model_complexity,
        )

    def detect(self, frame: np.ndarray) -> Optional[PoseLandmarks]:
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._pose.process(rgb)

        if not results.pose_landmarks:
            return None

        lm = results.pose_landmarks.landmark
        pl = self._mp_pose.PoseLandmark

        def pt(idx: int) -> Tuple[float, float]:
            return (lm[idx].x * w, lm[idx].y * h)

        return PoseLandmarks(
            nose=pt(pl.NOSE),
            left_shoulder=pt(pl.LEFT_SHOULDER),
            right_shoulder=pt(pl.RIGHT_SHOULDER),
            left_wrist=pt(pl.LEFT_WRIST),
            right_wrist=pt(pl.RIGHT_WRIST),
            left_hip=pt(pl.LEFT_HIP),
            right_hip=pt(pl.RIGHT_HIP),
            left_knee=pt(pl.LEFT_KNEE),
            right_knee=pt(pl.RIGHT_KNEE),
            raw_results=results,
        )

    def draw_skeleton(self, frame: np.ndarray, landmarks: PoseLandmarks) -> None:
        self._mp_draw.draw_landmarks(
            frame,
            landmarks.raw_results.pose_landmarks,
            self._mp_pose.POSE_CONNECTIONS,
            self._mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
            self._mp_draw.DrawingSpec(color=(0, 180, 255), thickness=2),
        )

    def release(self) -> None:
        self._pose.close()
