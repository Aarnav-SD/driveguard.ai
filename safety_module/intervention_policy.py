from dataclasses import dataclass

from .road_context import RoadContext, RoadRiskProfile, assess_road_risk


@dataclass
class DriverState:
    eye_closure_ratio: float
    blink_rate: float
    head_nod_rate: float
    voice_energy: float
    reaction_delay_s: float
    hands_on_wheel: bool


@dataclass
class VehicleState:
    speed_kmph: float
    steering_stability: float
    brake_ready: bool


@dataclass
class InterventionDecision:
    fatigue_score: int
    risk_band: str
    primary_action: str
    throttle_limit_pct: int
    target_speed_kmph: int
    hazard_lights: bool
    lane_assist_enabled: bool
    stop_strategy: str
    rationale: str


def compute_fatigue_score(driver_state: DriverState) -> int:
    score = 0.0
    score += min(driver_state.eye_closure_ratio, 1.0) * 35
    score += min(driver_state.blink_rate / 30.0, 1.0) * 15
    score += min(driver_state.head_nod_rate / 12.0, 1.0) * 20
    score += max(0.0, 1.0 - driver_state.voice_energy) * 10
    score += min(driver_state.reaction_delay_s / 3.0, 1.0) * 15
    score += 5 if not driver_state.hands_on_wheel else 0
    return max(min(int(round(score)), 100), 0)


def decide_intervention(
    driver_state: DriverState,
    vehicle_state: VehicleState,
    road_context: RoadContext,
) -> tuple[InterventionDecision, RoadRiskProfile]:
    fatigue_score = compute_fatigue_score(driver_state)
    road_risk = assess_road_risk(road_context)
    lane_assist_enabled = road_risk.lane_assist_confidence >= 0.55

    if fatigue_score < 35:
        return (
            InterventionDecision(
                fatigue_score=fatigue_score,
                risk_band="monitor",
                primary_action="Continue monitoring and log baseline behavior.",
                throttle_limit_pct=100,
                target_speed_kmph=int(vehicle_state.speed_kmph),
                hazard_lights=False,
                lane_assist_enabled=False,
                stop_strategy="none",
                rationale="Signals suggest the driver is alert enough for passive monitoring.",
            ),
            road_risk,
        )

    if fatigue_score < 60:
        return (
            InterventionDecision(
                fatigue_score=fatigue_score,
                risk_band="warning",
                primary_action="Issue escalating audio, seat, and dashboard alerts.",
                throttle_limit_pct=90,
                target_speed_kmph=max(int(vehicle_state.speed_kmph - 5), 20),
                hazard_lights=False,
                lane_assist_enabled=lane_assist_enabled,
                stop_strategy="driver_recovery_window",
                rationale="Early fatigue signs detected; the system should seek driver re-engagement first.",
            ),
            road_risk,
        )

    if fatigue_score < 80:
        return (
            InterventionDecision(
                fatigue_score=fatigue_score,
                risk_band="assist",
                primary_action="Limit speed, enable lane stabilization where confidence is high, and start response timeout.",
                throttle_limit_pct=70,
                target_speed_kmph=max(int(vehicle_state.speed_kmph * 0.7), 25),
                hazard_lights=True,
                lane_assist_enabled=lane_assist_enabled,
                stop_strategy="prepare_minimal_risk_manoeuvre",
                rationale="The driver is likely impaired; transition from warning to assisted stabilization.",
            ),
            road_risk,
        )

    if road_risk.recommended_stop_style == "controlled_shoulder_stop":
        action = "Run a minimum-risk maneuver: hazards on, reduce speed smoothly, maintain lane, then move to a safe shoulder."
        stop_strategy = "controlled_shoulder_stop"
    elif road_risk.recommended_stop_style == "delay_full_stop_until_safe_bay":
        action = "Maintain lane, use controlled deceleration and engine-brake friendly slowdown, and stop only at a verified bay or turnout."
        stop_strategy = "delay_full_stop_until_safe_bay"
    else:
        action = "Keep the truck centered, reduce speed aggressively but smoothly, and avoid abrupt pull-over until a safe stopping zone is identified."
        stop_strategy = road_risk.recommended_stop_style

    target_speed = max(int(vehicle_state.speed_kmph * 0.45), 15)
    if road_context.road_type.lower() == "ghat":
        target_speed = min(target_speed, 25)

    return (
        InterventionDecision(
            fatigue_score=fatigue_score,
            risk_band="critical",
            primary_action=action,
            throttle_limit_pct=45,
            target_speed_kmph=target_speed,
            hazard_lights=True,
            lane_assist_enabled=lane_assist_enabled,
            stop_strategy=stop_strategy,
            rationale=(
                "Critical fatigue with poor recovery likelihood. "
                + road_risk.context_summary
            ),
        ),
        road_risk,
    )

