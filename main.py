import sys
import time

import cv2

from config import Config
from controller import InputController
from detector import PoseDetector
from gestures import GestureEngine
from visualizer import Visualizer

_COUNTDOWN_SECONDS = 3


class Application:
    def __init__(self, config: Config):
        self._config = config
        self._detector = PoseDetector(
            min_detection_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
            model_complexity=config.model_complexity,
        )
        self._gesture_engine = GestureEngine(config)
        self._input_controller = InputController()
        self._visualizer = Visualizer(config)

    def run(self) -> None:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.frame_height)

        if not cap.isOpened():
            print("Error: could not open webcam.")
            sys.exit(1)

        try:
            self._run_countdown(cap)
            self._run_main_loop(cap)
        finally:
            cap.release()
            self._detector.release()
            cv2.destroyAllWindows()

    def _run_countdown(self, cap: cv2.VideoCapture) -> None:
        start = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            elapsed = time.time() - start
            remaining = _COUNTDOWN_SECONDS - int(elapsed)

            if remaining <= 0:
                break

            h, w = frame.shape[:2]
            cv2.putText(
                frame,
                "Get ready...",
                (w // 2 - 130, h // 2 - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (255, 255, 255),
                3,
            )
            cv2.putText(
                frame,
                str(remaining),
                (w // 2 - 30, h // 2 + 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                3.0,
                (0, 200, 255),
                6,
            )
            cv2.imshow(self._config.window_name, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                cap.release()
                self._detector.release()
                cv2.destroyAllWindows()
                sys.exit(0)

    def _run_main_loop(self, cap: cv2.VideoCapture) -> None:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            fps = self._visualizer.tick_fps()
            landmarks = self._detector.detect(frame)

            gestures = []
            if landmarks is not None:
                self._detector.draw_skeleton(frame, landmarks)
                gestures = self._gesture_engine.detect(landmarks, w, h)
                self._input_controller.dispatch(gestures)

            self._visualizer.register_gestures(gestures)
            self._visualizer.render(
                frame, self._gesture_engine.current_zone, fps
            )

            cv2.imshow(self._config.window_name, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


def main() -> None:
    config = Config()
    app = Application(config)
    app.run()


if __name__ == "__main__":
    main()
