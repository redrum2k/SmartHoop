"""Zone definitions and geometry helpers for hoop, backboard, attempt ROI."""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np


@dataclass
class ZonePolygon:
    """Polygon zone defined by list of (x, y) points."""
    points: List[Tuple[float, float]]

    def to_np(self) -> np.ndarray:
        return np.array(self.points, dtype=np.float32)


@dataclass
class ZoneEllipse:
    """Ellipse zone: center (cx, cy), axes (a, b), angle in degrees."""
    center: Tuple[float, float]
    axes: Tuple[float, float]
    angle: float = 0.0


Zone = Union[ZonePolygon, ZoneEllipse]


def point_in_polygon(p: Tuple[float, float], polygon: ZonePolygon) -> bool:
    """Check if point is inside polygon using OpenCV pointPolygonTest."""
    import cv2
    pts = polygon.to_np()
    result = cv2.pointPolygonTest(pts, p, False)
    return result >= 0


def point_in_ellipse(
    p: Tuple[float, float],
    center: Tuple[float, float],
    axes: Tuple[float, float],
    angle: float = 0.0,
) -> bool:
    """Check if point is inside ellipse."""
    import cv2
    cx, cy = center
    ax, ay = axes
    cos_a = np.cos(np.radians(-angle))
    sin_a = np.sin(np.radians(-angle))
    dx = p[0] - cx
    dy = p[1] - cy
    rx = dx * cos_a - dy * sin_a
    ry = dx * sin_a + dy * cos_a
    return (rx / ax) ** 2 + (ry / ay) ** 2 <= 1.0


def point_in_zone(p: Tuple[float, float], zone: Zone) -> bool:
    """Check if point is inside zone (polygon or ellipse)."""
    if isinstance(zone, ZonePolygon):
        return point_in_polygon(p, zone)
    if isinstance(zone, ZoneEllipse):
        return point_in_ellipse(p, zone.center, zone.axes, zone.angle)
    raise TypeError(f"Unknown zone type: {type(zone)}")


def bbox_overlaps_zone(bbox: Tuple[float, float, float, float], zone: ZonePolygon) -> bool:
    """Check if bounding box overlaps polygon zone (center or corners)."""
    x1, y1, x2, y2 = bbox
    corners = [
        ((x1 + x2) / 2, (y1 + y2) / 2),
        (x1, y1),
        (x2, y1),
        (x2, y2),
        (x1, y2),
    ]
    for c in corners:
        if point_in_polygon(c, zone):
            return True
    return False


def line_segment_crosses_hoop(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    hoop_zone: Zone,
    rim_y: Optional[float] = None,
) -> bool:
    """Check if line segment from p1 to p2 crosses rim (above->below) within hoop zone."""
    y1, y2 = p1[1], p2[1]
    if rim_y is None and isinstance(hoop_zone, ZonePolygon):
        pts = hoop_zone.points
        rim_y = sum(p[1] for p in pts) / len(pts) if pts else 0
    elif rim_y is None and isinstance(hoop_zone, ZoneEllipse):
        rim_y = hoop_zone.center[1]
    else:
        rim_y = rim_y or 0
    crosses = y1 < rim_y and y2 >= rim_y  # above (smaller y) -> below (larger y)
    if not crosses:
        return False
    mid = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
    return point_in_zone(mid, hoop_zone)


def ball_crosses_rim_downward(
    history: List[Tuple[float, float]],
    hoop_zone: Zone,
    rim_y: Optional[float] = None,
) -> bool:
    """Check if ball trajectory crosses rim from above to below within hoop zone."""
    if len(history) < 2:
        return False
    if rim_y is None and isinstance(hoop_zone, ZonePolygon):
        pts = hoop_zone.points
        rim_y = sum(p[1] for p in pts) / len(pts) if pts else 0
    elif rim_y is None and isinstance(hoop_zone, ZoneEllipse):
        rim_y = hoop_zone.center[1]
    else:
        rim_y = rim_y or 0
    for i in range(1, len(history)):
        if line_segment_crosses_hoop(history[i - 1], history[i], hoop_zone, rim_y):
            return True
    return False


def zones_from_dict(data: dict) -> dict:
    """Load zones from dict (e.g. from zones.json)."""
    result = {}
    if "hoop_zone" in data:
        hz = data["hoop_zone"]
        if "points" in hz:
            result["hoop_zone"] = ZonePolygon([(p[0], p[1]) for p in hz["points"]])
        elif "center" in hz:
            c = hz["center"]
            a = hz.get("axes", (50, 50))
            result["hoop_zone"] = ZoneEllipse(
                center=(c[0], c[1]),
                axes=(a[0], a[1]),
                angle=hz.get("angle", 0),
            )
    if "backboard_zone" in data:
        bz = data["backboard_zone"]
        result["backboard_zone"] = ZonePolygon([(p[0], p[1]) for p in bz["points"]])
    if "attempt_roi" in data:
        ar = data["attempt_roi"]
        result["attempt_roi"] = ZonePolygon([(p[0], p[1]) for p in ar["points"]])
    return result


def zones_to_dict(zones: dict) -> dict:
    """Serialize zones to dict for JSON export."""
    result = {}
    if "hoop_zone" in zones:
        z = zones["hoop_zone"]
        if isinstance(z, ZonePolygon):
            result["hoop_zone"] = {"points": [[float(p[0]), float(p[1])] for p in z.points]}
        elif isinstance(z, ZoneEllipse):
            result["hoop_zone"] = {
                "center": [float(z.center[0]), float(z.center[1])],
                "axes": [float(z.axes[0]), float(z.axes[1])],
                "angle": float(z.angle),
            }
    if "backboard_zone" in zones:
        z = zones["backboard_zone"]
        result["backboard_zone"] = {"points": [[float(p[0]), float(p[1])] for p in z.points]}
    if "attempt_roi" in zones:
        z = zones["attempt_roi"]
        result["attempt_roi"] = {"points": [[float(p[0]), float(p[1])] for p in z.points]}
    return result
