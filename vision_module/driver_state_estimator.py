from collections import deque
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np

from safety_module.intervention_policy import DriverState


LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
NOSE_TIP = 1


@dataclass
class DriverFrameSummary:
    state: DriverState
    fatigue_label: str
    ear: float
    face_detected: bool


class DriverStateEstimator:
    def __init__(self) -> None:
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

    @staticmethod
    def _eye_aspect_ratio(eye_points: list[int], landmarks, width: int, height: int) -> float:
        points = []
        for index in eye_points:
            lm = landmarks[index]
            points.append((int(lm.x * width), int(lm.y * height)))

        p1, p2, p3, p4, p5, p6 = points
        vertical_1 = np.linalg.norm(np.array(p2) - np.array(p6))
        vertical_2 = np.linalg.norm(np.array(p3) - np.array(p5))
        horizontal = np.linalg.norm(np.array(p1) - np.array(p4))
        if horizontal == 0:
            return 0.0
        return float((vertical_1 + vertical_2) / (2.0 * horizontal))

    def estimate_from_frames(self, frames: list[np.ndarray], fps: float = 15.0) -> DriverFrameSummary:
        closed_frames = 0
        blink_count = 0
        eye_closed = False
        ear_values: list[float] = []
        nose_positions: list[float] = []
        smoothed_ears: deque[float] = deque(maxlen=5)
        face_detected = False

        for frame in frames:
            height, width, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)
            if not results.multi_face_landmarks:
                continue

            face_detected = True
            landmarks = results.multi_face_landmarks[0].landmark
            left_ear = self._eye_aspect_ratio(LEFT_EYE, landmarks, width, height)
            right_ear = self._eye_aspect_ratio(RIGHT_EYE, landmarks, width, height)
            ear = (left_ear + right_ear) / 2.0
            smoothed_ears.append(ear)
            ear = float(sum(smoothed_ears) / len(smoothed_ears))
            ear_values.append(ear)

            nose_tip = landmarks[NOSE_TIP]
            nose_positions.append(float(nose_tip.y))

            if ear < 0.23:
                closed_frames += 1
                if not eye_closed:
                    blink_count += 1
                    eye_closed = True
            else:
                eye_closed = False

        frame_count = max(len(frames), 1)
        duration_s = max(frame_count / fps, 1.0)
        avg_ear = float(np.mean(ear_values)) if ear_values else 0.30
        eye_closure_ratio = min(closed_frames / frame_count, 1.0)
        blink_rate = (blink_count / duration_s) * 60.0

        head_motion = 0.0
        if len(nose_positions) > 2:
            head_motion = float(np.std(nose_positions))
        head_nod_rate = min(head_motion * 180.0, 15.0)

        fatigue_label = "ALERT"
        if eye_closure_ratio > 0.55 or avg_ear < 0.20:
            fatigue_label = "SLEEPY"
        elif eye_closure_ratio > 0.30 or avg_ear < 0.24:
            fatigue_label = "FATIGUE"

        state = DriverState(
            eye_closure_ratio=round(eye_closure_ratio, 2),
            blink_rate=round(blink_rate, 1),
            head_nod_rate=round(head_nod_rate, 1),
            voice_energy=0.5,
            reaction_delay_s=2.6 if fatigue_label == "SLEEPY" else 1.6 if fatigue_label == "FATIGUE" else 0.8,
            hands_on_wheel=True,
        )
        return DriverFrameSummary(
            state=state,
            fatigue_label=fatigue_label,
            ear=round(avg_ear, 3),
            face_detected=face_detected,
        )

