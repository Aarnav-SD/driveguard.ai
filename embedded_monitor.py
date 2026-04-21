from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
import sys
import threading
import time
from typing import Optional

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from safety_module.intervention_policy import VehicleState, decide_intervention
from vision_module.driver_state_estimator import DriverStateEstimator
from vision_module.road_context_estimator import RoadContextEstimator

try:
    import winsound
except ImportError:  # pragma: no cover - Windows-only convenience
    winsound = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DriveGuard AI embedded-style live monitor."
    )
    parser.add_argument("--driver-camera", type=int, default=0)
    parser.add_argument("--road-camera", type=int, default=1)
    parser.add_argument("--speed-kmph", type=float, default=60.0)
    parser.add_argument("--steering-stability", type=float, default=0.78)
    parser.add_argument("--analysis-window", type=int, default=18)
    parser.add_argument("--fps-hint", type=float, default=15.0)
    parser.add_argument("--single-camera", action="store_true")
    return parser.parse_args()


def open_camera(index: int) -> Optional[cv2.VideoCapture]:
    camera = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if camera.isOpened():
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        camera.set(cv2.CAP_PROP_FPS, 15)
        return camera
    camera.release()
    return None


def color_for_band(risk_band: str) -> tuple[int, int, int]:
    if risk_band == "critical":
        return (0, 0, 255)
    if risk_band in {"warning", "assist"}:
        return (0, 255, 255)
    return (0, 200, 0)


class AlarmController:
    def __init__(self) -> None:
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _alarm_loop(self) -> None:
        pattern = [
            (2200, 350),
            (1600, 250),
            (2200, 350),
            (900, 250),
        ]
        while not self._stop_event.is_set():
            if winsound is None:
                time.sleep(0.25)
                continue
            for frequency, duration_ms in pattern:
                if self._stop_event.is_set():
                    break
                try:
                    winsound.Beep(frequency, duration_ms)
                except RuntimeError:
                    winsound.MessageBeep(winsound.MB_ICONHAND)
                    time.sleep(duration_ms / 1000.0)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._alarm_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.2)
        self._thread = None


def update_alert(
    risk_band: str,
    last_alert_time: float,
    current_alarm_mode: str,
    alarm_controller: AlarmController,
) -> tuple[float, str]:
    now = time.time()
    if winsound is None:
        return last_alert_time, current_alarm_mode
    try:
        if risk_band == "critical":
            if current_alarm_mode != "critical":
                alarm_controller.start()
            return now, "critical"

        if current_alarm_mode == "critical":
            alarm_controller.stop()
            current_alarm_mode = "idle"

        if risk_band == "assist" and (now - last_alert_time) > 1.8:
            winsound.Beep(1500, 450)
            return now, "assist"
        if risk_band == "warning" and (now - last_alert_time) > 2.5:
            winsound.Beep(1100, 250)
            return now, "warning"
    except RuntimeError:
        winsound.MessageBeep(winsound.MB_ICONHAND)
        return now, "fallback"
    return last_alert_time, current_alarm_mode


def alert_band_for_output(decision, driver_summary) -> str:
    if driver_summary.fatigue_label == "SLEEPY":
        return "critical"
    if driver_summary.fatigue_label == "FATIGUE":
        return "assist"
    return decision.risk_band


def annotate_driver_frame(frame, decision, driver_summary, road_summary) -> None:
    color = color_for_band(decision.risk_band)
    frame_height, frame_width = frame.shape[:2]

    if driver_summary.face_box:
        x, y, w, h = driver_summary.face_box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    status_lines = [
        f"Driver: {driver_summary.fatigue_label}",
        f"Fatigue Score: {decision.fatigue_score}",
        f"Action: {decision.risk_band.upper()}",
    ]
    for idx, line in enumerate(status_lines):
        cv2.putText(
            frame,
            line,
            (16, 28 + (idx * 22)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            2,
        )

    panel_height = 118
    panel_top = max(frame_height - panel_height, 0)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, panel_top), (frame_width, frame_height), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    left_lines = [
        f"Eyes visible: {driver_summary.eyes_detect_ratio:.2f}",
        f"Closed duration: {driver_summary.closed_duration_s:.2f}s",
        f"Eye score: {driver_summary.eye_openness_score:.2f}",
        f"Baseline: {driver_summary.baseline_eye_openness:.2f}",
        f"Blink rate: {driver_summary.state.blink_rate:.1f}/min",
        f"Head nod: {driver_summary.state.head_nod_rate:.1f}",
    ]
    right_lines = [
        f"Road: {road_summary.context.road_type}",
        f"Lane assist: {road_summary.context.lane_marking_quality} / {road_summary.visibility_score:.2f}",
        f"Target speed: {decision.target_speed_kmph} km/h",
        f"Stop: {decision.stop_strategy}",
    ]

    for idx, line in enumerate(left_lines):
        cv2.putText(
            frame,
            line,
            (18, panel_top + 24 + (idx * 22)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (245, 245, 245),
            1,
        )

    right_x = max((frame_width // 2) + 10, 250)
    for idx, line in enumerate(right_lines):
        cv2.putText(
            frame,
            line,
            (right_x, panel_top + 24 + (idx * 22)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (245, 245, 245),
            1,
        )


def annotate_road_frame(frame, road_summary, road_risk) -> None:
    frame_height, frame_width = frame.shape[:2]
    panel_height = 102
    panel_top = max(frame_height - panel_height, 0)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, panel_top), (frame_width, frame_height), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    lines = [
        f"Road Type: {road_summary.context.road_type}",
        f"Lane Count: {road_summary.context.lane_count}",
        f"Lane Markings: {road_summary.context.lane_marking_quality}",
        f"Traffic: {road_summary.context.traffic_density}",
        f"Weather: {road_summary.context.weather}",
        f"Lane Assist Conf: {road_risk.lane_assist_confidence:.2f}",
        f"Pull-over Safety: {road_risk.pull_over_safety:.2f}",
    ]
    for idx, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (18, panel_top + 22 + (idx * 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (255, 255, 255),
            1,
        )


def enhance_frame_for_display(frame: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y_channel, cr_channel, cb_channel = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    y_channel = clahe.apply(y_channel)
    merged = cv2.merge((y_channel, cr_channel, cb_channel))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)
    return cv2.convertScaleAbs(enhanced, alpha=1.03, beta=-8)


def main() -> int:
    args = parse_args()

    driver_camera = open_camera(args.driver_camera)
    if driver_camera is None:
        print(f"Could not open driver camera index {args.driver_camera}.")
        return 1

    road_camera = None if args.single_camera else open_camera(args.road_camera)
    if road_camera is None and not args.single_camera:
        print(
            f"Road camera index {args.road_camera} not available. "
            "Falling back to single-camera mode."
        )

    driver_estimator = DriverStateEstimator()
    road_estimator = RoadContextEstimator()
    frame_buffer: deque[np.ndarray] = deque(maxlen=max(args.analysis_window, 15))
    last_decision = None
    last_road_risk = None
    last_driver_summary = None
    last_road_summary = None
    last_alert_time = 0.0
    current_alarm_mode = "idle"
    frame_index = 0
    alarm_controller = AlarmController()

    print("DriveGuard AI embedded monitor started.")
    print("Press Q to quit.")

    try:
        while True:
            ok, driver_frame = driver_camera.read()
            if not ok:
                print("Driver camera feed lost.")
                break

            road_frame = None
            if road_camera is not None:
                road_ok, road_frame = road_camera.read()
                if not road_ok:
                    road_frame = None

            raw_driver_frame = driver_frame.copy()
            driver_frame = enhance_frame_for_display(driver_frame)
            if road_frame is not None:
                raw_road_frame = road_frame.copy()
                road_frame = enhance_frame_for_display(road_frame)
            else:
                raw_road_frame = None

            frame_buffer.append(raw_driver_frame)
            frame_index += 1

            if len(frame_buffer) >= frame_buffer.maxlen and frame_index % 2 == 0:
                sampled_frames = list(frame_buffer)[::2]
                last_driver_summary = driver_estimator.estimate_from_frames(
                    sampled_frames,
                    fps=max(args.fps_hint / 2.0, 1.0),
                )
                last_road_summary = road_estimator.estimate_from_frame(
                    raw_road_frame
                )
                vehicle_state = VehicleState(
                    speed_kmph=args.speed_kmph,
                    steering_stability=args.steering_stability,
                    brake_ready=True,
                )
                last_decision, last_road_risk = decide_intervention(
                    last_driver_summary.state,
                    vehicle_state,
                    last_road_summary.context,
                )
                last_alert_time, current_alarm_mode = update_alert(
                    alert_band_for_output(last_decision, last_driver_summary),
                    last_alert_time,
                    current_alarm_mode,
                    alarm_controller,
                )

            display_driver = driver_frame.copy()
            if last_decision and last_driver_summary and last_road_summary:
                annotate_driver_frame(
                    display_driver,
                    last_decision,
                    last_driver_summary,
                    last_road_summary,
                )
            else:
                cv2.putText(
                    display_driver,
                    "Collecting initial sensor window... Hold eyes closed for 1-2 seconds to test alert.",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )

            cv2.imshow("DriveGuard AI - Driver Monitor", display_driver)

            if road_frame is not None:
                display_road = road_frame.copy()
                if last_road_summary and last_road_risk:
                    annotate_road_frame(display_road, last_road_summary, last_road_risk)
                cv2.imshow("DriveGuard AI - Road Monitor", display_road)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("b"), ord("B")):
                last_alert_time, current_alarm_mode = update_alert(
                    "critical",
                    0.0,
                    current_alarm_mode,
                    alarm_controller,
                )
            if key in (ord("s"), ord("S")):
                alarm_controller.stop()
                current_alarm_mode = "idle"
            if key in (ord("q"), ord("Q")):
                break
    finally:
        alarm_controller.stop()
        driver_camera.release()
        if road_camera is not None:
            road_camera.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
