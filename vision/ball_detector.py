"""Ball detection via YOLOv8 with optional HSV fallback for debugging."""

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class BallDetection:
    """Single ball detection."""
    bbox: tuple  # (x1, y1, x2, y2)
    conf: float
    cx: float
    cy: float
    radius_est: float


class BallDetector:
    """YOLO-based ball detector with optional HSV fallback."""

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        custom_model_path: Optional[str] = None,
        ball_class_id: int = 32,
        confidence_threshold: float = 0.35,
        inference_size: int = 416,
        use_hsv_fallback: bool = False,
        hue_low: int = 0,
        hue_high: int = 10,
        hue_low2: int = 170,
        hue_high2: int = 180,
        sat_min: int = 100,
        val_min: int = 100,
    ):
        path = custom_model_path if custom_model_path else model_path
        self.model = YOLO(path)
        self.ball_class_id = ball_class_id
        self.confidence_threshold = confidence_threshold
        self.inference_size = inference_size
        self.use_hsv_fallback = use_hsv_fallback
        self._hsv_params = {
            "hue_low": hue_low,
            "hue_high": hue_high,
            "hue_low2": hue_low2,
            "hue_high2": hue_high2,
            "sat_min": sat_min,
            "val_min": val_min,
        }

    def detect(self, frame: np.ndarray) -> List[BallDetection]:
        """Detect balls in frame. Returns list of BallDetection."""
        if self.use_hsv_fallback:
            return self._detect_hsv(frame)
        return self._detect_yolo(frame)

    def _detect_yolo(self, frame: np.ndarray) -> List[BallDetection]:
        """Run YOLO inference and filter for sports ball class."""
        results = self.model.predict(
            frame,
            imgsz=self.inference_size,
            conf=self.confidence_threshold,
            verbose=False,
        )
        detections: List[BallDetection] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id != self.ball_class_id:
                    continue
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                w = x2 - x1
                h = y2 - y1
                radius_est = min(w, h) / 2
                detections.append(
                    BallDetection(
                        bbox=(x1, y1, x2, y2),
                        conf=conf,
                        cx=cx,
                        cy=cy,
                        radius_est=radius_est,
                    )
                )
        return detections

    def _detect_hsv(self, frame: np.ndarray) -> List[BallDetection]:
        """HSV red segmentation fallback for debugging."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lo = np.array([self._hsv_params["hue_low"], self._hsv_params["sat_min"], self._hsv_params["val_min"]])
        hi = np.array([self._hsv_params["hue_high"], 255, 255])
        lo2 = np.array([self._hsv_params["hue_low2"], self._hsv_params["sat_min"], self._hsv_params["val_min"]])
        hi2 = np.array([self._hsv_params["hue_high2"], 255, 255])
        mask1 = cv2.inRange(hsv, lo, hi)
        mask2 = cv2.inRange(hsv, lo2, hi2)
        mask = cv2.bitwise_or(mask1, mask2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: List[BallDetection] = []
        min_area = 50
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            (x, y), radius = cv2.minEnclosingCircle(c)
            radius = float(radius)
            if radius < 3:
                continue
            x1 = x - radius
            y1 = y - radius
            x2 = x + radius
            y2 = y + radius
            detections.append(
                BallDetection(
                    bbox=(x1, y1, x2, y2),
                    conf=0.8,
                    cx=float(x),
                    cy=float(y),
                    radius_est=radius,
                )
            )
        return detections
