from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from safety_module.intervention_policy import DriverState


@dataclass
class DriverFrameSummary:
    state: DriverState
    fatigue_label: str
    ear: float
    face_detected: bool
    face_detect_ratio: float
    eyes_detect_ratio: float
    closed_duration_s: float


class DriverStateEstimator:
    def __init__(self) -> None:
        cascade_root = cv2.data.haarcascades
        self.face_cascade = cv2.CascadeClassifier(
            cascade_root + "haarcascade_frontalface_default.xml"
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cascade_root + "haarcascade_eye_tree_eyeglasses.xml"
        )

    def estimate_from_frames(self, frames: list[np.ndarray], fps: float = 15.0) -> DriverFrameSummary:
        closed_frames = 0
        blink_count = 0
        eye_closed = False
        ear_values: list[float] = []
        face_center_y: list[float] = []
        smoothed_ears: deque[float] = deque(maxlen=5)
        face_detected = False
        detected_face_frames = 0
        eyes_visible_frames = 0
        closed_streak = 0
        max_closed_streak = 0

        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(80, 80),
            )
            if len(faces) == 0:
                continue

            face_detected = True
            detected_face_frames += 1
            x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
            upper_face = gray[y : y + int(h * 0.55), x : x + w]
            eyes = self.eye_cascade.detectMultiScale(
                upper_face,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(18, 18),
            )

            eye_band = upper_face[int(upper_face.shape[0] * 0.18) : int(upper_face.shape[0] * 0.68), :]
            eye_edges = cv2.Canny(eye_band, 40, 120) if eye_band.size else None
            eye_texture_score = 0.0
            if eye_edges is not None and eye_edges.size:
                eye_texture_score = min(float(np.count_nonzero(eye_edges) / eye_edges.size) / 0.16, 1.0)

            # Webcam-friendly approximation:
            # combine eye detections with edge texture from the eye band to estimate eye openness.
            if len(eyes) >= 2:
                eyes_visible_frames += 1
                detection_score = 1.0
            elif len(eyes) == 1:
                eyes_visible_frames += 1
                detection_score = 0.6
            else:
                detection_score = 0.15

            openness_score = (0.7 * detection_score) + (0.3 * eye_texture_score)
            eye_open_score = 0.16 + (0.15 * openness_score)

            ear = eye_open_score
            smoothed_ears.append(ear)
            ear = float(sum(smoothed_ears) / len(smoothed_ears))
            ear_values.append(ear)
            face_center_y.append(float(y + (h / 2.0)))

            if ear < 0.23:
                closed_frames += 1
                closed_streak += 1
                max_closed_streak = max(max_closed_streak, closed_streak)
                if not eye_closed:
                    blink_count += 1
                    eye_closed = True
            else:
                eye_closed = False
                closed_streak = 0

        frame_count = max(len(frames), 1)
        observed_frame_count = max(detected_face_frames, 1)
        duration_s = max(observed_frame_count / fps, 1.0)
        avg_ear = float(np.mean(ear_values)) if ear_values else 0.30
        eye_closure_ratio = min(closed_frames / observed_frame_count, 1.0)
        blink_rate = (blink_count / duration_s) * 60.0

        head_motion = 0.0
        if len(face_center_y) > 2:
            head_motion = float(np.std(face_center_y))
        head_nod_rate = min(head_motion / 3.5, 15.0)
        closed_duration_s = max_closed_streak / max(fps, 1.0)
        face_detect_ratio = detected_face_frames / frame_count
        eyes_detect_ratio = eyes_visible_frames / observed_frame_count

        fatigue_label = "ALERT"
        if (
            eye_closure_ratio > 0.42
            or avg_ear < 0.205
            or closed_duration_s > 1.2
        ):
            fatigue_label = "SLEEPY"
        elif (
            eye_closure_ratio > 0.22
            or avg_ear < 0.235
            or closed_duration_s > 0.6
        ):
            fatigue_label = "FATIGUE"

        state = DriverState(
            eye_closure_ratio=round(eye_closure_ratio, 2),
            blink_rate=round(blink_rate, 1),
            head_nod_rate=round(head_nod_rate, 1),
            voice_energy=0.2 if fatigue_label == "SLEEPY" else 0.35 if fatigue_label == "FATIGUE" else 0.7,
            reaction_delay_s=3.6 if fatigue_label == "SLEEPY" else 2.1 if fatigue_label == "FATIGUE" else 0.8,
            hands_on_wheel=False if fatigue_label == "SLEEPY" else True,
        )
        return DriverFrameSummary(
            state=state,
            fatigue_label=fatigue_label,
            ear=round(avg_ear, 3),
            face_detected=face_detected,
            face_detect_ratio=round(face_detect_ratio, 2),
            eyes_detect_ratio=round(eyes_detect_ratio, 2),
            closed_duration_s=round(closed_duration_s, 2),
        )
