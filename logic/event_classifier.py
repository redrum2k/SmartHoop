"""Shot attempt state machine: classify hit, backboard, miss."""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, Optional, Tuple

from vision.zones import (
    Zone,
    ZonePolygon,
    ball_crosses_rim_downward,
    bbox_overlaps_zone,
    point_in_zone,
)


class State(Enum):
    IDLE = "IDLE"
    ATTEMPT = "ATTEMPT"
    CLASSIFIED = "CLASSIFIED"


class ShotOutcome(Enum):
    HIT = "hit"
    BACKBOARD = "backboard"
    MISS = "miss"


@dataclass
class ClassifierResult:
    """Result of classification step."""
    state: State
    outcome: Optional[ShotOutcome]
    hit: int
    backboard: int
    miss: int


class EventClassifier:
    """State machine for shot attempt classification."""

    def __init__(
        self,
        hoop_zone: Zone,
        backboard_zone: ZonePolygon,
        attempt_roi: Optional[ZonePolygon] = None,
        K_hit_frames: int = 2,
        K_backboard_frames: int = 2,
        T_end_sec: float = 3.0,
        cooldown_sec: float = 1.0,
        downward_velocity_threshold: float = 50.0,
        T_hit_max_sec: float = 2.0,
    ):
        self.hoop_zone = hoop_zone
        self.backboard_zone = backboard_zone
        self.attempt_roi = attempt_roi
        self.K_hit_frames = K_hit_frames
        self.K_backboard_frames = K_backboard_frames
        self.T_end_sec = T_end_sec
        self.cooldown_sec = cooldown_sec
        self.downward_velocity_threshold = downward_velocity_threshold
        self.T_hit_max_sec = T_hit_max_sec

        self._state = State.IDLE
        self._hit_count = 0
        self._backboard_count = 0
        self._miss_count = 0
        self._attempt_start_time: Optional[float] = None
        self._classified_at: Optional[float] = None
        self._hit_confirmation_count = 0
        self._backboard_confirmation_count = 0
        self._attempt_id = 0

    @property
    def hit(self) -> int:
        return self._hit_count

    @property
    def backboard(self) -> int:
        return self._backboard_count

    @property
    def miss(self) -> int:
        return self._miss_count

    @property
    def state(self) -> State:
        return self._state

    def update(
        self,
        tracked_ball: Optional[object],
        fps: float = 30.0,
    ) -> ClassifierResult:
        """
        Update classifier with current tracked ball.
        tracked_ball: TrackedBall or None
        Returns ClassifierResult with state, outcome (if just classified), counters.
        """
        now = time.time()
        outcome = None

        if self._state == State.CLASSIFIED:
            if self._classified_at and (now - self._classified_at) >= self.cooldown_sec:
                self._state = State.IDLE
                self._classified_at = None
            return ClassifierResult(
                state=self._state,
                outcome=None,
                hit=self._hit_count,
                backboard=self._backboard_count,
                miss=self._miss_count,
            )

        if tracked_ball is None:
            if self._state == State.ATTEMPT:
                if self._attempt_start_time and (now - self._attempt_start_time) >= self.T_end_sec:
                    self._miss_count += 1
                    outcome = ShotOutcome.MISS
                    self._state = State.CLASSIFIED
                    self._classified_at = now
                    self._attempt_id += 1
            return ClassifierResult(
                state=self._state,
                outcome=outcome,
                hit=self._hit_count,
                backboard=self._backboard_count,
                miss=self._miss_count,
            )

        cx = tracked_ball.cx
        cy = tracked_ball.cy
        velocity = tracked_ball.velocity
        history = list(tracked_ball.history) if hasattr(tracked_ball, "history") else []
        bbox = getattr(tracked_ball, "bbox", None)

        in_attempt_roi = (
            point_in_zone((cx, cy), self.attempt_roi)
            if self.attempt_roi
            else True
        )

        if self._state == State.IDLE:
            if in_attempt_roi:
                self._state = State.ATTEMPT
                self._attempt_start_time = now
                self._hit_confirmation_count = 0
                self._backboard_confirmation_count = 0
            return ClassifierResult(
                state=self._state,
                outcome=None,
                hit=self._hit_count,
                backboard=self._backboard_count,
                miss=self._miss_count,
            )

        if self._state == State.ATTEMPT:
            elapsed = now - self._attempt_start_time if self._attempt_start_time else 0

            # Hit: rim crossing + downward velocity + in hoop zone
            if elapsed <= self.T_hit_max_sec and len(history) >= 2:
                crosses = ball_crosses_rim_downward(history, self.hoop_zone)
                vy = velocity[1] if len(velocity) > 1 else 0
                if crosses and vy >= self.downward_velocity_threshold:
                    self._hit_confirmation_count += 1
                else:
                    self._hit_confirmation_count = 0
            else:
                self._hit_confirmation_count = 0

            if self._hit_confirmation_count >= self.K_hit_frames:
                self._hit_count += 1
                outcome = ShotOutcome.HIT
                self._state = State.CLASSIFIED
                self._classified_at = now
                self._attempt_id += 1
                return ClassifierResult(
                    state=self._state,
                    outcome=outcome,
                    hit=self._hit_count,
                    backboard=self._backboard_count,
                    miss=self._miss_count,
                )

            # Backboard: ball overlaps backboard zone (and not hit)
            if isinstance(self.backboard_zone, ZonePolygon) and bbox:
                if bbox_overlaps_zone(bbox, self.backboard_zone):
                    self._backboard_confirmation_count += 1
                else:
                    self._backboard_confirmation_count = 0
            elif point_in_zone((cx, cy), self.backboard_zone):
                self._backboard_confirmation_count += 1
            else:
                self._backboard_confirmation_count = 0

            if self._backboard_confirmation_count >= self.K_backboard_frames:
                self._backboard_count += 1
                outcome = ShotOutcome.BACKBOARD
                self._state = State.CLASSIFIED
                self._classified_at = now
                self._attempt_id += 1
                return ClassifierResult(
                    state=self._state,
                    outcome=outcome,
                    hit=self._hit_count,
                    backboard=self._backboard_count,
                    miss=self._miss_count,
                )

            # Miss: ball left ROI or timeout
            if not in_attempt_roi or elapsed >= self.T_end_sec:
                self._miss_count += 1
                outcome = ShotOutcome.MISS
                self._state = State.CLASSIFIED
                self._classified_at = now
                self._attempt_id += 1
                return ClassifierResult(
                    state=self._state,
                    outcome=outcome,
                    hit=self._hit_count,
                    backboard=self._backboard_count,
                    miss=self._miss_count,
                )

        return ClassifierResult(
            state=self._state,
            outcome=None,
            hit=self._hit_count,
            backboard=self._backboard_count,
            miss=self._miss_count,
        )

    def get_attempt_id(self) -> int:
        return self._attempt_id
