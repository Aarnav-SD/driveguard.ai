from pathlib import Path
import sys
import time

import cv2
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from safety_module.intervention_policy import VehicleState, decide_intervention
from vision_module.driver_state_estimator import DriverStateEstimator
from vision_module.road_context_estimator import RoadContextEstimator


st.set_page_config(page_title="DriveGuard AI", layout="wide")

st.title("DriveGuard AI")
st.caption("Automatic fatigue sensing with road-aware minimum-risk intervention planning.")

st.info(
    "This version removes manual condition entry. It captures the driver state from camera 0, "
    "tries to estimate road context from camera 1, and then generates a warning / safe-state plan."
)

st.subheader("Sensor Setup")
speed_kmph = st.slider("Estimated vehicle speed (km/h)", 0, 120, 60)
steering_stability = st.slider("Steering stability", 0.0, 1.0, 0.78, 0.01)
brake_ready = st.checkbox("Brake system ready", value=True)
driver_camera_index = st.number_input("Driver camera index", min_value=0, max_value=4, value=0, step=1)
road_camera_index = st.number_input("Road camera index", min_value=0, max_value=4, value=1, step=1)
sample_seconds = st.slider("Automatic capture window (seconds)", 2, 8, 4)


def capture_frames(camera_index: int, seconds: int, fps_hint: int = 15) -> list:
    capture = cv2.VideoCapture(int(camera_index))
    if not capture.isOpened():
        return []

    frames = []
    frame_target = max(seconds * fps_hint, 1)
    start_time = time.time()

    while len(frames) < frame_target and (time.time() - start_time) < (seconds + 1):
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)

    capture.release()
    return frames


decision = None
road_risk = None
driver_summary = None
road_summary = None

if st.button("Analyze Live Sensors", type="primary"):
    with st.spinner("Capturing sensor frames and running fatigue analysis..."):
        driver_frames = capture_frames(driver_camera_index, sample_seconds)
        road_frames = capture_frames(road_camera_index, sample_seconds)

        if not driver_frames:
            st.error("Driver camera could not be read. Automatic driver-state sensing needs a working camera feed.")
        else:
            driver_estimator = DriverStateEstimator()
            road_estimator = RoadContextEstimator()
            driver_summary = driver_estimator.estimate_from_frames(driver_frames)
            road_summary = road_estimator.estimate_from_frame(road_frames[-1] if road_frames else None)
            vehicle_state = VehicleState(
                speed_kmph=speed_kmph,
                steering_stability=steering_stability,
                brake_ready=brake_ready,
            )
            decision, road_risk = decide_intervention(driver_summary.state, vehicle_state, road_summary.context)

            st.session_state["decision"] = decision
            st.session_state["road_risk"] = road_risk
            st.session_state["driver_summary"] = driver_summary
            st.session_state["road_summary"] = road_summary

decision = st.session_state.get("decision")
road_risk = st.session_state.get("road_risk")
driver_summary = st.session_state.get("driver_summary")
road_summary = st.session_state.get("road_summary")

if decision is None or road_risk is None or driver_summary is None or road_summary is None:
    st.warning("Run live analysis to let the system sense driver and road conditions automatically.")
    st.stop()

metric_1, metric_2, metric_3 = st.columns(3)
metric_1.metric("Fatigue Score", decision.fatigue_score)
metric_2.metric("Lane Assist Confidence", road_risk.lane_assist_confidence)
metric_3.metric("Pull-Over Safety", road_risk.pull_over_safety)

if decision.risk_band == "monitor":
    st.success(decision.primary_action)
elif decision.risk_band in {"warning", "assist"}:
    st.warning(decision.primary_action)
else:
    st.error(decision.primary_action)

st.subheader("Recommended Safe-State Plan")
st.write(f"**Risk band:** {decision.risk_band}")
st.write(f"**Target speed:** {decision.target_speed_kmph} km/h")
st.write(f"**Throttle limit:** {decision.throttle_limit_pct}%")
st.write(f"**Hazard lights:** {'On' if decision.hazard_lights else 'Off'}")
st.write(f"**Lane assist:** {'Enabled' if decision.lane_assist_enabled else 'Disabled'}")
st.write(f"**Stop strategy:** {decision.stop_strategy}")
st.write(f"**Why:** {decision.rationale}")

st.subheader("Automatically Sensed Driver State")
st.write(f"**Face detected:** {'Yes' if driver_summary.face_detected else 'No'}")
st.write(f"**Fatigue label:** {driver_summary.fatigue_label}")
st.write(f"**Average EAR:** {driver_summary.ear}")
st.write(f"**Eye closure ratio:** {driver_summary.state.eye_closure_ratio}")
st.write(f"**Blink rate:** {driver_summary.state.blink_rate} blinks/min")
st.write(f"**Head nod rate:** {driver_summary.state.head_nod_rate}")

st.subheader("Road-Aware Interpretation")
st.write(road_risk.context_summary)
st.write(f"**Road type estimate:** {road_summary.context.road_type}")
st.write(f"**Lane count estimate:** {road_summary.context.lane_count}")
st.write(f"**Lane marking quality:** {road_summary.context.lane_marking_quality}")
st.write(f"**Traffic density estimate:** {road_summary.context.traffic_density}")
st.write(f"**Weather estimate:** {road_summary.context.weather}")
st.write(f"**Road estimator fallback used:** {'Yes' if road_summary.used_fallback else 'No'}")

st.subheader("Research Positioning")
st.markdown(
    """
    - Differentiates from alarm-only sleep detection by adapting intervention to road geometry and stopping safety.
    - Uses automatic sensing from camera feeds instead of asking the user to manually enter road or driver condition labels.
    - Handles narrow single-lane and ghat-like situations by preferring controlled in-lane slowdown over unsafe shoulder pull-over.
    - Keeps the project in a realistic Level 1/2 assistance scope rather than overclaiming full autonomous driving.
    """
)
