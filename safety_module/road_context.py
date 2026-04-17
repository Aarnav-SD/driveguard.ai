from dataclasses import dataclass


@dataclass
class RoadContext:
    road_type: str
    lane_count: int
    lane_width_m: float
    shoulder_available: bool
    shoulder_width_m: float
    traffic_density: str
    curvature_level: str
    slope_level: str
    lane_marking_quality: str
    weather: str


@dataclass
class RoadRiskProfile:
    lane_assist_confidence: float
    pull_over_safety: float
    in_lane_stop_risk: float
    context_summary: str
    recommended_stop_style: str


def _scale_quality(value: str, mapping: dict[str, float], default: float) -> float:
    return mapping.get(value.lower(), default)


def assess_road_risk(context: RoadContext) -> RoadRiskProfile:
    lane_quality = _scale_quality(
        context.lane_marking_quality,
        {"poor": 0.35, "fair": 0.6, "good": 0.85},
        0.5,
    )
    weather_factor = _scale_quality(
        context.weather,
        {"clear": 1.0, "rain": 0.75, "fog": 0.55},
        0.8,
    )
    curvature_factor = _scale_quality(
        context.curvature_level,
        {"low": 1.0, "medium": 0.8, "high": 0.55},
        0.75,
    )
    slope_factor = _scale_quality(
        context.slope_level,
        {"flat": 1.0, "moderate": 0.8, "steep": 0.55},
        0.75,
    )
    traffic_factor = _scale_quality(
        context.traffic_density,
        {"low": 1.0, "medium": 0.8, "high": 0.55},
        0.75,
    )

    width_factor = min(max((context.lane_width_m - 2.7) / 1.0, 0.25), 1.0)
    lane_count_factor = 1.0 if context.lane_count >= 2 else 0.65

    lane_assist_confidence = (
        lane_quality
        * weather_factor
        * curvature_factor
        * slope_factor
        * width_factor
        * lane_count_factor
    )
    lane_assist_confidence = round(max(min(lane_assist_confidence, 1.0), 0.1), 2)

    shoulder_factor = 0.2
    if context.shoulder_available:
        shoulder_factor = min(max(context.shoulder_width_m / 3.0, 0.35), 1.0)

    road_type_factor = {
        "expressway": 0.95,
        "highway": 0.8,
        "urban": 0.45,
        "single_lane": 0.25,
        "ghat": 0.15,
    }.get(context.road_type.lower(), 0.5)

    pull_over_safety = round(
        max(min(shoulder_factor * road_type_factor * traffic_factor, 1.0), 0.05), 2
    )
    in_lane_stop_risk = round(
        max(min((1.05 - traffic_factor) + (1.05 - lane_count_factor), 1.0), 0.15), 2
    )

    if context.road_type.lower() == "ghat" or context.slope_level.lower() == "steep":
        summary = "Steep or hilly road: prioritize stable lane holding and controlled deceleration."
        stop_style = "delay_full_stop_until_safe_bay"
    elif context.road_type.lower() == "single_lane":
        summary = "Single-lane narrow road: avoid abrupt pull-over, keep vehicle centered and slow down."
        stop_style = "in_lane_slowdown_until_turnout"
    elif pull_over_safety >= 0.65 and lane_assist_confidence >= 0.55:
        summary = "Road supports assisted pull-over if driver remains unresponsive."
        stop_style = "controlled_shoulder_stop"
    else:
        summary = "Mixed conditions: use warnings, speed limiting, and lane stabilization first."
        stop_style = "stabilize_then_search_safe_stop"

    return RoadRiskProfile(
        lane_assist_confidence=lane_assist_confidence,
        pull_over_safety=pull_over_safety,
        in_lane_stop_risk=in_lane_stop_risk,
        context_summary=summary,
        recommended_stop_style=stop_style,
    )

