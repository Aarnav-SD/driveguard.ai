from dataclasses import dataclass

import cv2
import numpy as np

from safety_module.road_context import RoadContext


@dataclass
class RoadEstimateSummary:
    context: RoadContext
    lane_line_count: int
    edge_density: float
    visibility_score: float
    used_fallback: bool


def _default_context() -> RoadContext:
    return RoadContext(
        road_type="urban",
        lane_count=1,
        lane_width_m=3.0,
        shoulder_available=False,
        shoulder_width_m=0.0,
        traffic_density="medium",
        curvature_level="medium",
        slope_level="moderate",
        lane_marking_quality="poor",
        weather="clear",
    )


class RoadContextEstimator:
    def estimate_from_frame(self, frame: np.ndarray | None) -> RoadEstimateSummary:
        if frame is None:
            return RoadEstimateSummary(
                context=_default_context(),
                lane_line_count=0,
                edge_density=0.0,
                visibility_score=0.0,
                used_fallback=True,
            )

        height, width, _ = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi = gray[int(height * 0.45):, :]
        blurred = cv2.GaussianBlur(roi, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        edge_density = float(np.count_nonzero(edges) / edges.size)

        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=45,
            minLineLength=max(width // 10, 40),
            maxLineGap=35,
        )

        lane_candidates = []
        if lines is not None:
            for line in lines[:, 0]:
                x1, y1, x2, y2 = line
                if x2 == x1:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                if abs(slope) > 0.35:
                    lane_candidates.append((x1, y1, x2, y2, slope))

        left_lines = [line for line in lane_candidates if line[4] < 0]
        right_lines = [line for line in lane_candidates if line[4] > 0]
        lane_line_count = int(bool(left_lines)) + int(bool(right_lines))

        mean_brightness = float(np.mean(roi) / 255.0)
        contrast = float(np.std(roi) / 255.0)
        visibility_score = round(min(max((contrast * 1.8) + (mean_brightness * 0.4), 0.0), 1.0), 2)

        if visibility_score < 0.25:
            weather = "fog"
        elif mean_brightness < 0.25:
            weather = "rain"
        else:
            weather = "clear"

        if lane_line_count >= 2 and edge_density < 0.09:
            lane_marking_quality = "good"
        elif lane_line_count >= 1:
            lane_marking_quality = "fair"
        else:
            lane_marking_quality = "poor"

        if lane_line_count >= 2:
            lane_count = 2
            road_type = "highway"
            lane_width_m = 3.5
        else:
            lane_count = 1
            road_type = "single_lane"
            lane_width_m = 3.0

        slope_level = "steep" if mean_brightness < 0.18 and edge_density > 0.10 else "moderate"
        curvature_level = "high" if len(lane_candidates) >= 6 else "medium" if lane_candidates else "low"
        shoulder_available = lane_count >= 2 and edge_density < 0.08
        shoulder_width_m = 2.2 if shoulder_available else 0.0
        traffic_density = "high" if edge_density > 0.12 else "medium" if edge_density > 0.06 else "low"

        context = RoadContext(
            road_type=road_type,
            lane_count=lane_count,
            lane_width_m=lane_width_m,
            shoulder_available=shoulder_available,
            shoulder_width_m=shoulder_width_m,
            traffic_density=traffic_density,
            curvature_level=curvature_level,
            slope_level=slope_level,
            lane_marking_quality=lane_marking_quality,
            weather=weather,
        )
        return RoadEstimateSummary(
            context=context,
            lane_line_count=lane_line_count,
            edge_density=round(edge_density, 3),
            visibility_score=visibility_score,
            used_fallback=False,
        )

