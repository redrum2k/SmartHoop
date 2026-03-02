"""Multi-frame ball tracking: nearest-neighbor association + optional Kalman smoothing."""

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

import numpy as np

from vision.ball_detector import BallDetection


@dataclass
class TrackedBall:
    """Tracked ball state."""
    cx: float
    cy: float
    velocity: Tuple[float, float]
    history: Deque[Tuple[float, float]]
    is_occluded: bool
    bbox: Optional[Tuple[float, float, float, float]] = None
    conf: float = 0.0


class BallTracker:
    """Track ball across frames with nearest-neighbor association and occlusion handling."""

    def __init__(
        self,
        max_occlusion_frames: int = 5,
        history_sec: float = 1.5,
        fps: float = 30.0,
        use_kalman: bool = False,
        association_max_distance: float = 80.0,
    ):
        self.max_occlusion_frames = max_occlusion_frames
        self.history_max_len = max(2, int(history_sec * fps))
        self.use_kalman = use_kalman
        self.association_max_distance = association_max_distance
        self._last_cx: Optional[float] = None
        self._last_cy: Optional[float] = None
        self._history: Deque[Tuple[float, float]] = deque(maxlen=self.history_max_len)
        self._occlusion_count = 0
        self._velocity = (0.0, 0.0)
        self._kf = None
        if use_kalman:
            self._init_kalman()

    def _init_kalman(self):
        """Initialize 2D constant-velocity Kalman filter."""
        try:
            from filterpy.kalman import KalmanFilter
            self._kf = KalmanFilter(dim_x=4, dim_z=2)
            self._kf.x = np.array([0, 0, 0, 0])
            self._kf.F = np.array([
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ])
            self._kf.H = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
            ])
            self._kf.R *= 10
            self._kf.P *= 100
        except ImportError:
            self.use_kalman = False

    def update(
        self,
        detections: List[BallDetection],
        dt: float = 1.0 / 30.0,
    ) -> Optional[TrackedBall]:
        """Update track with new detections. Returns TrackedBall or None."""
        if not detections and self._last_cx is None:
            return None

        if not detections:
            if self._occlusion_count >= self.max_occlusion_frames:
                self._reset()
                return None
            self._occlusion_count += 1
            if self.use_kalman and self._kf is not None:
                self._kf.predict()
                cx, cy = float(self._kf.x[0]), float(self._kf.x[1])
                vx, vy = float(self._kf.x[2]), float(self._kf.x[3])
            else:
                cx = self._last_cx + self._velocity[0] * dt
                cy = self._last_cy + self._velocity[1] * dt
                vx, vy = self._velocity
            self._last_cx, self._last_cy = cx, cy
            self._history.append((cx, cy))
            return TrackedBall(
                cx=cx,
                cy=cy,
                velocity=(vx, vy),
                history=deque(self._history, maxlen=self.history_max_len),
                is_occluded=True,
                bbox=None,
                conf=0.0,
            )

        best = self._nearest_detection(detections)
        if best is None:
            if self._occlusion_count >= self.max_occlusion_frames:
                self._reset()
                return None
            self._occlusion_count += 1
            if self.use_kalman and self._kf is not None:
                self._kf.predict()
                cx, cy = float(self._kf.x[0]), float(self._kf.x[1])
                vx, vy = float(self._kf.x[2]), float(self._kf.x[3])
            else:
                cx = self._last_cx + self._velocity[0] * dt
                cy = self._last_cy + self._velocity[1] * dt
                vx, vy = self._velocity
            self._last_cx, self._last_cy = cx, cy
            self._history.append((cx, cy))
            return TrackedBall(
                cx=cx,
                cy=cy,
                velocity=(vx, vy),
                history=deque(self._history, maxlen=self.history_max_len),
                is_occluded=True,
                bbox=None,
                conf=0.0,
            )

        self._occlusion_count = 0
        cx, cy = best.cx, best.cy
        if self._last_cx is not None:
            vx = (cx - self._last_cx) / dt if dt > 0 else 0
            vy = (cy - self._last_cy) / dt if dt > 0 else 0
            self._velocity = (vx, vy)
            if self.use_kalman and self._kf is not None:
                self._kf.update(np.array([cx, cy]))
                self._kf.predict()
                vx = float(self._kf.x[2])
                vy = float(self._kf.x[3])
                self._velocity = (vx, vy)
        self._last_cx, self._last_cy = cx, cy
        self._history.append((cx, cy))

        return TrackedBall(
            cx=cx,
            cy=cy,
            velocity=self._velocity,
            history=deque(self._history, maxlen=self.history_max_len),
            is_occluded=False,
            bbox=best.bbox,
            conf=best.conf,
        )

    def _nearest_detection(self, detections: List[BallDetection]) -> Optional[BallDetection]:
        """Return detection closest to last known position, within max distance."""
        if self._last_cx is None:
            return detections[0] if detections else None
        best = None
        best_dist = float("inf")
        for d in detections:
            dist = np.hypot(d.cx - self._last_cx, d.cy - self._last_cy)
            if dist < best_dist and dist <= self.association_max_distance:
                best_dist = dist
                best = d
        return best

    def _reset(self):
        """Reset track state."""
        self._last_cx = None
        self._last_cy = None
        self._history.clear()
        self._occlusion_count = 0
        self._velocity = (0.0, 0.0)
        if self._kf is not None:
            self._kf.x = np.array([0, 0, 0, 0])
