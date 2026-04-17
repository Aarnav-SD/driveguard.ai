from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from safety_module.intervention_policy import DriverState, VehicleState, decide_intervention
from safety_module.road_context import RoadContext


def run_scenario(name: str, driver_state: DriverState, vehicle_state: VehicleState, road_context: RoadContext) -> None:
    decision, road_risk = decide_intervention(driver_state, vehicle_state, road_context)
    print(f"Scenario: {name}")
    print(f"  Fatigue score: {decision.fatigue_score}")
    print(f"  Risk band: {decision.risk_band}")
    print(f"  Action: {decision.primary_action}")
    print(f"  Stop strategy: {decision.stop_strategy}")
    print(f"  Lane assist enabled: {decision.lane_assist_enabled}")
    print(f"  Target speed: {decision.target_speed_kmph} km/h")
    print(f"  Lane confidence: {road_risk.lane_assist_confidence}")
    print(f"  Pull-over safety: {road_risk.pull_over_safety}")
    print()


if __name__ == "__main__":
    scenarios = [
        (
            "Expressway shoulder stop",
            DriverState(0.82, 28, 9, 0.25, 2.7, False),
            VehicleState(82, 0.85, True),
            RoadContext("expressway", 3, 3.6, True, 3.2, "medium", "low", "flat", "good", "clear"),
        ),
        (
            "Single-lane village road",
            DriverState(0.78, 24, 8, 0.30, 2.5, False),
            VehicleState(46, 0.75, True),
            RoadContext("single_lane", 1, 2.9, False, 0.0, "medium", "medium", "moderate", "fair", "clear"),
        ),
        (
            "Ghat section",
            DriverState(0.84, 26, 10, 0.20, 2.9, False),
            VehicleState(38, 0.72, True),
            RoadContext("ghat", 1, 3.0, False, 0.0, "low", "high", "steep", "fair", "fog"),
        ),
    ]

    for name, driver_state, vehicle_state, road_context in scenarios:
        run_scenario(name, driver_state, vehicle_state, road_context)
