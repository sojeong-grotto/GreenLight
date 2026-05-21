from dataclasses import dataclass
import math
import time

import cv2
import numpy as np

from config import (
    FORWARD_ZONE_X1,
    FORWARD_ZONE_X2,
    MOVING_APPROACH_SCALE,
    MOVING_APPROACH_WINDOW_SEC,
    MOVING_FAST_AREA_GROWTH_PER_SEC,
    MOVING_FRONTAL_MIN_BOTTOM_SHIFT,
    MOVING_LATERAL_MIN_AREA_RATIO,
    MOVING_LATERAL_MIN_BOTTOM_Y,
    MOVING_LATERAL_MIN_X_SHIFT,
    MOVING_MATCH_DISTANCE,
    MOVING_MIN_OBSERVATIONS,
    MOVING_NEAR_AREA_RATIO,
    MOVING_TRACK_MAX_GAP_SEC,
    PERSON_NEAR_AREA_RATIO,
    PERSON_NEAR_BOTTOM_Y,
    PERSON_NEAR_WIDTH_RATIO,
    PATH_ROI_POLYGON,
    ROI_BOTTOM_OVERLAP_RATIO,
    STATIC_CONFIRM_FRAMES,
    STATIC_NEAR_AREA_RATIO,
    STATIC_VEHICLE_SUPPRESS_OVERLAP_RATIO,
    SURFACE_BRAILLE_PRIORITY_RATIO,
    SURFACE_CONFIRM_FRAMES,
    SURFACE_MIN_DOMINANT_RATIO,
    VEHICLE_ROI_POLYGON,
)


STATIC_OBSTACLE_LABELS = {
    "vertical_obstacle",
    "temporary_obstacle",
    "bench",
    "traffic_sign",
    "bus_taxi_stop",
}
MOVING_RISK_LABELS = {"vehicle"}
PERSON_LABELS = {"person", "mobility_aid"}
ROADWAY_LABELS = {"roadway", "bike_lane", "caution_zone"}
ALLEY_LABELS = {"alley"}
BRAILLE_LABELS = {"braille_guide_blocks"}
CROSSWALK_LABELS = {"crosswalk"}


@dataclass
class AlertEvent:
    key: str
    message: str
    priority: int = 1
    object_id: str | None = None


class ObjectSafetyAnalyzer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.prev_objects = {}
        self.next_track_id = 1
        self.static_candidate_counts = {}

    def analyze(self, detections):
        events = []
        moving_current = []
        static_candidates = set()
        person_added = False
        vehicle_boxes = [det.box for det in detections if det.box is not None and det.label in MOVING_RISK_LABELS]

        for det in detections:
            if det.box is None:
                continue

            x1, y1, x2, y2 = det.box
            box_w = max(0.0, x2 - x1)
            box_h = max(0.0, y2 - y1)
            area_ratio = box_w * box_h
            bottom_y = y2
            cx = (x1 + x2) * 0.5
            cy = (y1 + y2) * 0.5

            if det.label in STATIC_OBSTACLE_LABELS:
                if (
                    not static_box_matches_vehicle(det.box, vehicle_boxes)
                    and box_bottom_overlaps_polygon(det.box, PATH_ROI_POLYGON)
                    and area_ratio >= STATIC_NEAR_AREA_RATIO
                ):
                    static_candidates.add(object_signature(det.label, cx, cy))
            elif det.label in MOVING_RISK_LABELS:
                in_path_roi = box_bottom_overlaps_polygon(det.box, PATH_ROI_POLYGON)
                in_vehicle_roi = vehicle_box_in_polygon(det.box, VEHICLE_ROI_POLYGON)
                moving_current.append((det.label, cx, cy, area_ratio, box_w, box_h, bottom_y, in_vehicle_roi, in_path_roi))
            elif det.label in PERSON_LABELS and not person_added:
                in_forward_zone = FORWARD_ZONE_X1 <= cx <= FORWARD_ZONE_X2
                in_path_roi = box_bottom_anchors_in_polygon(det.box, PATH_ROI_POLYGON)
                person_is_near = (
                    area_ratio >= PERSON_NEAR_AREA_RATIO
                    or bottom_y >= PERSON_NEAR_BOTTOM_Y
                    or box_w >= PERSON_NEAR_WIDTH_RATIO
                )
                if in_path_roi and in_forward_zone and person_is_near:
                    events.append(
                        AlertEvent(
                            "person_near",
                            "앞에 사람이 가깝습니다. 천천히 이동하세요.",
                            5,
                            object_signature(det.label, cx, cy),
                        )
                    )
                    person_added = True

        for object_id in self._confirmed_static_obstacles(static_candidates):
            events.append(AlertEvent("static_obstacle", "앞에 장애물이 있습니다. 주의하세요.", 7, object_id))

        moving_vehicle_ids = self._classify_vehicle_tracks(moving_current)
        for object_id in moving_vehicle_ids:
            events.append(AlertEvent("moving_obstacle", "차량이 가까워지고 있습니다. 주의하세요.", 9, object_id))

        events.sort(key=lambda event: event.priority, reverse=True)
        return events

    def _confirmed_static_obstacles(self, static_candidates):
        updated = {}
        confirmed = []
        for object_id in static_candidates:
            count = self.static_candidate_counts.get(object_id, 0) + 1
            updated[object_id] = count
            if count >= STATIC_CONFIRM_FRAMES:
                confirmed.append(object_id)
        self.static_candidate_counts = updated
        return confirmed

    def _classify_vehicle_tracks(self, current):
        now = time.monotonic()
        moving_vehicle_ids = []
        updated = {}

        for label, cx, cy, area, box_w, box_h, bottom_y, in_vehicle_roi, in_path_roi in current:
            best_id = None
            best_dist = MOVING_MATCH_DISTANCE
            for track_id, prev in self.prev_objects.items():
                if prev["label"] != label:
                    continue
                if now - prev["time"] > MOVING_TRACK_MAX_GAP_SEC:
                    continue
                dist = math.hypot(prev["cx"] - cx, prev["cy"] - cy)
                if dist < best_dist:
                    best_dist = dist
                    best_id = track_id

            if best_id is None:
                best_id = self.next_track_id
                self.next_track_id += 1
                history = []
            else:
                history = list(self.prev_objects[best_id]["history"])

            history.append(
                {
                    "time": now,
                    "cx": cx,
                    "cy": cy,
                    "area": area,
                    "width": box_w,
                    "height": box_h,
                    "bottom_y": bottom_y,
                    "in_vehicle_roi": in_vehicle_roi,
                    "in_path_roi": in_path_roi,
                }
            )
            history = [entry for entry in history if now - entry["time"] <= MOVING_APPROACH_WINDOW_SEC]

            is_moving_risk = self._is_moving_vehicle_track_risky(history)
            if is_moving_risk:
                moving_vehicle_ids.append(f"vehicle:{best_id}")

            updated[best_id] = {
                "label": label,
                "cx": cx,
                "cy": cy,
                "area": area,
                "width": box_w,
                "height": box_h,
                "bottom_y": bottom_y,
                "in_vehicle_roi": in_vehicle_roi,
                "in_path_roi": in_path_roi,
                "time": now,
                "history": history,
            }

        self.prev_objects = updated
        return moving_vehicle_ids

    def _is_moving_vehicle_track_risky(self, history):
        if len(history) < MOVING_MIN_OBSERVATIONS:
            return False

        current = history[-1]
        if not current["in_vehicle_roi"]:
            return False

        baseline = history[0]
        baseline_area = max(baseline["area"], 1e-6)
        elapsed = max(current["time"] - baseline["time"], 1e-3)
        area_growth = current["area"] / baseline_area
        area_growth_rate = (area_growth - 1.0) / elapsed
        bottom_shift = current["bottom_y"] - baseline["bottom_y"]
        x_shift = current["cx"] - baseline["cx"]
        moved_toward_center = abs(current["cx"] - 0.5) < abs(baseline["cx"] - 0.5)
        active_lateral_motion = moved_toward_center and abs(x_shift) >= MOVING_LATERAL_MIN_X_SHIFT

        frontal_approach = (
            current["area"] >= MOVING_NEAR_AREA_RATIO
            and area_growth >= MOVING_APPROACH_SCALE
            and area_growth_rate >= MOVING_FAST_AREA_GROWTH_PER_SEC
            and bottom_shift >= MOVING_FRONTAL_MIN_BOTTOM_SHIFT
        )
        lateral_intrusion = (
            active_lateral_motion
            and current["area"] >= MOVING_LATERAL_MIN_AREA_RATIO
            and current["bottom_y"] >= MOVING_LATERAL_MIN_BOTTOM_Y
        )

        return frontal_approach or lateral_intrusion


class SurfaceSafetyAnalyzer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.candidate_counts = {}
        self._roi_cache = {}
        self.last_dominant = None
        self.last_announced_key = None

    def analyze(self, detections):
        ratios = surface_ratios_in_path_roi(detections, self._roi_cache)
        dominant = prioritized_surface_label(ratios)
        self.last_dominant = dominant
        if dominant is None:
            self.candidate_counts = {}
            return []

        label, ratio = dominant
        if ratio < SURFACE_MIN_DOMINANT_RATIO:
            self.candidate_counts = {}
            return []

        event = surface_event_for_label(label)
        if event is None:
            self.candidate_counts = {}
            return []

        return self._confirmed_surface_events([event])

    def _confirmed_surface_events(self, candidates):
        updated = {}
        confirmed = []
        for event in candidates:
            count = self.candidate_counts.get(event.key, 0) + 1
            updated[event.key] = count
            if count >= SURFACE_CONFIRM_FRAMES and event.key != self.last_announced_key:
                confirmed.append(event)
                self.last_announced_key = event.key
        self.candidate_counts = updated
        return confirmed


def surface_ratios_in_path_roi(detections, roi_cache):
    masks = [det for det in detections if det.mask is not None]
    if not masks:
        return {}

    height, width = masks[0].mask.shape[:2]
    roi_mask = path_roi_mask(width, height, roi_cache)
    roi_pixels = int(np.count_nonzero(roi_mask))
    if roi_pixels <= 0:
        return {}

    ratios = {}
    for det in masks:
        mask = det.mask.astype(bool)
        if mask.shape[:2] != roi_mask.shape[:2]:
            mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
        overlap = int(np.count_nonzero(mask & roi_mask))
        ratios[det.label] = max(ratios.get(det.label, 0.0), overlap / roi_pixels)
    return ratios


def dominant_surface_label(ratios):
    if not ratios:
        return None
    return max(ratios.items(), key=lambda item: item[1])


def prioritized_surface_label(ratios):
    braille_ratio = sum(ratios.get(label, 0.0) for label in BRAILLE_LABELS)
    if braille_ratio >= SURFACE_BRAILLE_PRIORITY_RATIO:
        return "braille_guide_blocks", braille_ratio
    return dominant_surface_label(ratios)


def surface_event_for_label(label):
    if label in ROADWAY_LABELS:
        return AlertEvent("roadway_entry", "전방이 차도입니다. 주의하세요.", 8, "surface:roadway")
    if label in ALLEY_LABELS:
        return AlertEvent("alley_entry", "전방이 이면도로입니다. 주의하세요.", 8, "surface:alley")
    if label in CROSSWALK_LABELS:
        return AlertEvent("crosswalk_detected", "전방에 횡단보도가 있습니다.", 6, "surface:crosswalk")
    if label in BRAILLE_LABELS:
        return AlertEvent("braille_blocks", "전방에 점자블록이 있습니다.", 6, "surface:braille")
    return None


def path_roi_mask(width, height, roi_cache):
    key = (width, height)
    cached = roi_cache.get(key)
    if cached is not None:
        return cached
    points = np.array([[(int(x * width), int(y * height)) for x, y in PATH_ROI_POLYGON]], dtype=np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, points, 1)
    roi_cache[key] = mask.astype(bool)
    return roi_cache[key]


def alert_roi_polygon():
    return PATH_ROI_POLYGON


def vehicle_roi_polygon():
    return VEHICLE_ROI_POLYGON


def object_signature(label, cx, cy, grid=0.12):
    return f"{label}:{int(cx / grid)}:{int(cy / grid)}"


def static_box_matches_vehicle(static_box, vehicle_boxes):
    for vehicle_box in vehicle_boxes:
        if box_center_inside(static_box, vehicle_box):
            return True
        if box_intersection_ratio(static_box, vehicle_box) >= STATIC_VEHICLE_SUPPRESS_OVERLAP_RATIO:
            return True
    return False


def box_center_inside(inner_box, outer_box):
    x1, y1, x2, y2 = inner_box
    ox1, oy1, ox2, oy2 = outer_box
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    return ox1 <= cx <= ox2 and oy1 <= cy <= oy2


def box_intersection_ratio(box, other_box):
    x1, y1, x2, y2 = box
    ox1, oy1, ox2, oy2 = other_box
    inter_w = max(0.0, min(x2, ox2) - max(x1, ox1))
    inter_h = max(0.0, min(y2, oy2) - max(y1, oy1))
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if area <= 0.0:
        return 0.0
    return (inter_w * inter_h) / area


def vehicle_box_in_polygon(box, polygon):
    return box_bottom_overlaps_polygon(box, polygon)


def box_bottom_anchors_in_polygon(box, polygon):
    return bottom_anchor_count_in_polygon(box, polygon) >= 1


def box_bottom_overlaps_polygon(box, polygon, min_ratio=ROI_BOTTOM_OVERLAP_RATIO):
    x1, _y1, x2, y2 = box
    width = max(0.0, x2 - x1)
    if width <= 0:
        return False
    sample_points = bottom_overlap_sample_points(box)
    inside_count = sum(1 for point in sample_points if point_in_polygon(point, polygon))
    return inside_count / len(sample_points) >= min_ratio


def bottom_anchor_count_in_polygon(box, polygon):
    return sum(1 for point in bottom_anchor_points(box) if point_in_polygon(point, polygon))


def bottom_overlap_sample_points(box, samples=21):
    x1, _y1, x2, y2 = box
    if samples <= 1:
        return [((x1 + x2) * 0.5, y2)]
    step = (x2 - x1) / (samples - 1)
    return [(x1 + step * idx, y2) for idx in range(samples)]


def bottom_anchor_points(box):
    x1, _y1, x2, y2 = box
    width = max(0.0, x2 - x1)
    return [
        (x1 + width * 0.25, y2),
        ((x1 + x2) * 0.5, y2),
        (x1 + width * 0.75, y2),
    ]


def point_in_polygon(point, polygon):
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        if (yi > y) != (yj > y):
            x_intersect = (xj - xi) * (y - yi) / max(1e-9, yj - yi) + xi
            if x < x_intersect:
                inside = not inside
        j = i
    return inside
