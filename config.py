from dataclasses import dataclass


@dataclass
class Config:
    frame_width: int = 1280
    frame_height: int = 720
    left_zone_ratio: float = 10 / 27
    right_zone_ratio: float = 17 / 27
    jump_shoulder_threshold: float = 22.0
    duck_body_ratio: float = 1.3
    hands_joined_threshold: int = 100
    jump_cooldown: float = 0.4
    duck_cooldown: float = 0.5
    roll_cooldown: float = 0.6
    shoulder_sample_interval: float = 0.05
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    model_complexity: int = 1
    window_name: str = "Subway Surfers Motion Controller"
    gesture_display_duration: float = 0.5
    fps_smoothing_window: int = 30
