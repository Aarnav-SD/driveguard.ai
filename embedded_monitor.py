from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
import sys
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
    parser.add_argument("--analysis-window", type=int, default=20)
    parser.add_argument("--fps-hint", type=float, default=15.0)
    parser.add_argument("--single-camera", action="store_true")
    return parser.parse_args()


def open_camera(index: int) -> Optional[cv2.VideoCapture]:
    camera = cv2.VideoCapture(index)
    if camera.isOpened():
        return camera
    camera.release()
    return None


def color_for_band(risk_band: str) -> tuple[int, int, int]:
    if risk_band == "critical":
        return (0, 0, 255)
    if risk_band in {"warning", "assist"}:
        return (0, 255, 255)
    return (0, 200, 0)


def play_alert(risk_band: str, last_alert_time: float) -> float:
    now = time.time()
    if winsound is None:
        return last_alert_time
    try:
        if risk_band == "critical" and (now - last_alert_time) > 1.0:
            winsound.Beep(2000, 700)
            return now
        if risk_band == "assist" and (now - last_alert_time) > 1.8:
            winsound.Beep(1500, 450)
            return now
        if risk_band == "warning" and (now - last_alert_time) > 2.5:
            winsound.Beep(1100, 250)
            return now
    except RuntimeError:
        winsound.MessageBeep(winsound.MB_ICONHAND)
        return now
    return last_alert_time


def alert_band_for_output(decision, driver_summary) -> str:
    if driver_summary.fatigue_label == "SLEEPY":
        return "critical"
    if driver_summary.fatigue_label == "FATIGUE" and decision.risk_band == "monitor":
        return "assist"
    if driver_summary.fatigue_label == "FATIGUE" and decision.risk_band == "warning":
        return "assist"
    return decision.risk_band


def annotate_driver_frame(frame, decision, driver_summary, road_summary) -> None:
    color = color_for_band(decision.risk_band)
    lines = [
        f"Driver: {driver_summary.fatigue_label}",
        f"Fatigue Score: {decision.fatigue_score}",
        f"Action Band: {decision.risk_band}",
        f"EAR: {driver_summary.ear:.3f}",
        f"Eyes Visible Ratio: {driver_summary.eyes_detect_ratio:.2f}",
        f"Eyes Closed Duration: {driver_summary.closed_duration_s:.2f}s",
        f"Blink Rate: {driver_summary.state.blink_rate:.1f}/min",
        f"Head Nod Rate: {driver_summary.state.head_nod_rate:.1f}",
        f"Road: {road_summary.context.road_type}",
        f"Lane Assist Conf: {road_summary.context.lane_marking_quality} / {road_summary.visibility_score:.2f}",
        f"Target Speed: {decision.target_speed_kmph} km/h",
        f"Stop Strategy: {decision.stop_strategy}",
    ]

    for idx, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (20, 35 + (idx * 28)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color if idx < 3 else (255, 255, 255),
            2,
        )


def annotate_road_frame(frame, road_summary, road_risk) -> None:
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
            (20, 35 + (idx * 28)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )


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

            frame_buffer.append(driver_frame.copy())

            if len(frame_buffer) >= frame_buffer.maxlen:
                last_driver_summary = driver_estimator.estimate_from_frames(
                    list(frame_buffer),
                    fps=args.fps_hint,
                )
                last_road_summary = road_estimator.estimate_from_frame(
                    road_frame
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
                last_alert_time = play_alert(
                    alert_band_for_output(last_decision, last_driver_summary),
                    last_alert_time,
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
                play_alert("critical", 0.0)
            if key in (ord("q"), ord("Q")):
                break
    finally:
        driver_camera.release()
        if road_camera is not None:
            road_camera.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
