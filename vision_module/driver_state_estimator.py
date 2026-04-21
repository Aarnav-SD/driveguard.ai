from collections import deque
from dataclasses import dataclass
from typing import Optional

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
    face_box: Optional[tuple[int, int, int, int]]
    eyes_detected: int
    eye_openness_score: float
    baseline_eye_openness: float


class DriverStateEstimator:
    def __init__(self) -> None:
        cascade_root = cv2.data.haarcascades
        self.face_cascade = cv2.CascadeClassifier(
            cascade_root + "haarcascade_frontalface_default.xml"
        )
        self.face_cascade_alt = cv2.CascadeClassifier(
            cascade_root + "haarcascade_frontalface_alt2.xml"
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cascade_root + "haarcascade_eye_tree_eyeglasses.xml"
        )
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.baseline_eye_openness = 22.0

    def _normalize_gray(self, gray: np.ndarray) -> np.ndarray:
        mean_intensity = float(np.mean(gray))
        gamma = 1.0
        if mean_intensity < 85:
            gamma = 0.75
        elif mean_intensity > 165:
            gamma = 1.35

        if gamma != 1.0:
            normalized = np.power(gray / 255.0, gamma)
            gray = np.uint8(np.clip(normalized * 255.0, 0, 255))

        enhanced = self.clahe.apply(gray)
        return cv2.GaussianBlur(enhanced, (5, 5), 0)

    def _detect_face(self, gray: np.ndarray, normalized_gray: np.ndarray) -> list[tuple[int, int, int, int]]:
        candidates = [
            self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.08,
                minNeighbors=4,
                minSize=(70, 70),
            ),
            self.face_cascade.detectMultiScale(
                normalized_gray,
                scaleFactor=1.08,
                minNeighbors=4,
                minSize=(70, 70),
            ),
            self.face_cascade_alt.detectMultiScale(
                gray,
                scaleFactor=1.08,
                minNeighbors=4,
                minSize=(70, 70),
            ),
            self.face_cascade_alt.detectMultiScale(
                normalized_gray,
                scaleFactor=1.08,
                minNeighbors=4,
                minSize=(70, 70),
            ),
        ]
        for faces in candidates:
            if len(faces) > 0:
                return faces
        return []

    def _eye_region_score(
        self,
        normalized_gray: np.ndarray,
        face_box: tuple[int, int, int, int],
    ) -> float:
        x, y, w, h = face_box
        left_eye = normalized_gray[
            y + int(h * 0.24) : y + int(h * 0.44),
            x + int(w * 0.14) : x + int(w * 0.42),
        ]
        right_eye = normalized_gray[
            y + int(h * 0.24) : y + int(h * 0.44),
            x + int(w * 0.58) : x + int(w * 0.86),
        ]
        scores = []
        for eye_roi in (left_eye, right_eye):
            if eye_roi.size == 0:
                continue
            sobel_y = cv2.Sobel(eye_roi, cv2.CV_32F, 0, 1, ksize=3)
            vertical_energy = float(np.mean(np.abs(sobel_y)))
            local_contrast = float(np.std(eye_roi))
            dark_ratio = float(np.mean(eye_roi < 85))
            scores.append((vertical_energy * 0.7) + (local_contrast * 0.25) + (dark_ratio * 12.0))
        if not scores:
            return 0.0
        return float(np.mean(scores))

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
        last_face_box: Optional[tuple[int, int, int, int]] = None
        last_eyes_detected = 0
        zero_eye_frames = 0
        one_eye_frames = 0
        openness_scores: list[float] = []
        baseline_candidates: list[float] = []

        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            normalized_gray = self._normalize_gray(gray)
            faces = self._detect_face(gray, normalized_gray)
            if len(faces) == 0:
                continue

            face_detected = True
            detected_face_frames += 1
            x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
            last_face_box = (x, y, w, h)
            upper_face = normalized_gray[y : y + int(h * 0.58), x : x + w]
            eyes = self.eye_cascade.detectMultiScale(
                upper_face,
                scaleFactor=1.05,
                minNeighbors=2,
                minSize=(14, 14),
            )
            last_eyes_detected = len(eyes)
            openness_score = self._eye_region_score(normalized_gray, last_face_box)
            openness_scores.append(openness_score)

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
                one_eye_frames += 1
            else:
                detection_score = 0.0
                zero_eye_frames += 1

            relative_eye_score = 1.0
            if self.baseline_eye_openness > 0:
                relative_eye_score = min(openness_score / self.baseline_eye_openness, 1.25)

            openness_mix = (0.45 * detection_score) + (0.20 * eye_texture_score) + (0.35 * relative_eye_score)
            eye_open_score = 0.12 + (0.18 * openness_mix)
            frame_closed = (
                len(eyes) == 0
                or relative_eye_score < 0.72
                or (len(eyes) == 1 and relative_eye_score < 0.85)
                or (eye_texture_score < 0.18 and relative_eye_score < 0.88)
            )
            if not frame_closed and (len(eyes) >= 1 or relative_eye_score > 0.92):
                baseline_candidates.append(openness_score)

            ear = eye_open_score
            smoothed_ears.append(ear)
            ear = float(sum(smoothed_ears) / len(smoothed_ears))
            ear_values.append(ear)
            face_center_y.append(float(y + (h / 2.0)))

            if frame_closed:
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
        zero_eye_ratio = zero_eye_frames / observed_frame_count
        partial_eye_ratio = one_eye_frames / observed_frame_count
        current_eye_openness = float(np.mean(openness_scores)) if openness_scores else 0.0
        if baseline_candidates:
            candidate_baseline = float(np.percentile(baseline_candidates, 85))
            if candidate_baseline > self.baseline_eye_openness:
                self.baseline_eye_openness = candidate_baseline
            else:
                self.baseline_eye_openness = (self.baseline_eye_openness * 0.96) + (candidate_baseline * 0.04)
        baseline_ratio = 1.0
        if self.baseline_eye_openness > 0:
            baseline_ratio = current_eye_openness / self.baseline_eye_openness

        fatigue_label = "ALERT"
        if (
            eye_closure_ratio > 0.42
            or avg_ear < 0.18
            or closed_duration_s > 0.8
            or zero_eye_ratio > 0.5
            or baseline_ratio < 0.63
        ):
            fatigue_label = "SLEEPY"
        elif (
            eye_closure_ratio > 0.2
            or avg_ear < 0.215
            or closed_duration_s > 0.35
            or zero_eye_ratio > 0.2
            or partial_eye_ratio > 0.35
            or baseline_ratio < 0.8
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
            face_box=last_face_box,
            eyes_detected=last_eyes_detected,
            eye_openness_score=round(current_eye_openness, 2),
            baseline_eye_openness=round(self.baseline_eye_openness, 2),
        )
