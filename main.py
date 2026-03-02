"""SmartHoop: real-time mini basketball shot classification."""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from logic.event_classifier import EventClassifier, ShotOutcome
from vision.ball_detector import BallDetector
from vision.tracker import BallTracker
from vision.zones import ZoneEllipse, ZonePolygon, zones_from_dict, zones_to_dict

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"
DEFAULT_ZONES = PROJECT_ROOT / "zones.json"


def load_config(path: Path) -> dict:
    """Load config.yaml."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_zones(path: Path) -> dict:
    """Load zones from zones.json. Returns empty dict if file missing."""
    if not path.exists():
        return {}
    with open(path, "r") as f:
        data = json.load(f)
    return zones_from_dict(data)


def run_calibration(source, zones_path: Path, config: dict):
    """Run calibration: user clicks to define hoop, backboard, attempt_roi polygons."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open source {source}")
        sys.exit(1)
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        sys.exit(1)
    cap.release()

    zones = {}
    window_name = "Calibration - SmartHoop"
    cv2.namedWindow(window_name)

    def _collect_polygon(zone_name: str, color: tuple, min_points: int = 3) -> ZonePolygon:
        """Collect polygon points via mouse clicks. Enter to finish."""
        pts = []
        display = frame.copy()

        def on_mouse(event, x, y, flags, param):
            nonlocal pts, display
            if event == cv2.EVENT_LBUTTONDOWN:
                pts.append((float(x), float(y)))
                display = frame.copy()
                for i, p in enumerate(pts):
                    cv2.circle(display, (int(p[0]), int(p[1])), 4, color, -1)
                    if i > 0:
                        cv2.line(
                            display,
                            (int(pts[i - 1][0]), int(pts[i - 1][1])),
                            (int(p[0]), int(p[1])),
                            color,
                            2,
                        )
                if len(pts) >= min_points:
                    cv2.polylines(
                        display,
                        [np.array(pts, dtype=np.int32)],
                        True,
                        color,
                        2,
                    )
                cv2.imshow(window_name, display)

        cv2.setMouseCallback(window_name, on_mouse)
        cv2.putText(
            display,
            f"Click {zone_name} points (min {min_points}). Press ENTER when done.",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.imshow(window_name, display)
        while True:
            key = cv2.waitKey(100) & 0xFF
            if key == 13 and len(pts) >= min_points:  # Enter
                break
            if key == ord("q"):
                cv2.destroyAllWindows()
                sys.exit(0)
        cv2.setMouseCallback(window_name, lambda *a: None)
        return ZonePolygon(pts)

    # Hoop zone
    print("Calibrating HOOP ZONE: Click points around the rim opening, then press ENTER.")
    zones["hoop_zone"] = _collect_polygon("hoop", (0, 255, 0), min_points=3)

    # Backboard zone
    print("Calibrating BACKBOARD ZONE: Click points around the backboard, then press ENTER.")
    zones["backboard_zone"] = _collect_polygon("backboard", (0, 255, 255), min_points=3)

    # Attempt ROI (optional)
    print("Calibrating ATTEMPT ROI (optional): Click points. Press ENTER when done, or press 's' to skip.")
    pts = []
    display = frame.copy()

    def on_mouse_roi(event, x, y, flags, param):
        nonlocal pts, display
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((float(x), float(y)))
            display = frame.copy()
            for i, p in enumerate(pts):
                cv2.circle(display, (int(p[0]), int(p[1])), 4, (255, 255, 0), -1)
                if i > 0:
                    cv2.line(
                        display,
                        (int(pts[i - 1][0]), int(pts[i - 1][1])),
                        (int(p[0]), int(p[1])),
                        (255, 255, 0),
                        2,
                    )
            if len(pts) >= 3:
                cv2.polylines(
                    display,
                    [np.array(pts, dtype=np.int32)],
                    True,
                    (255, 255, 0),
                    2,
                )
            cv2.imshow(window_name, display)

    cv2.setMouseCallback(window_name, on_mouse_roi)
    cv2.putText(
        display,
        "Click attempt_roi points (min 3). ENTER=done, s=skip",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    cv2.imshow(window_name, display)
    while True:
        key = cv2.waitKey(100) & 0xFF
        if key == 13 and len(pts) >= 3:
            zones["attempt_roi"] = ZonePolygon(pts)
            break
        if key == ord("s"):
            break
        if key == ord("q"):
            cv2.destroyAllWindows()
            sys.exit(0)
    cv2.setMouseCallback(window_name, lambda *a: None)

    # Save
    with open(zones_path, "w") as f:
        json.dump(zones_to_dict(zones), f, indent=2)
    print(f"Zones saved to {zones_path}")
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="SmartHoop basketball shot classifier")
    parser.add_argument("--source", type=str, default=None, help="Webcam index (0) or video path")
    parser.add_argument("--calibrate", action="store_true", help="Run calibration mode")
    parser.add_argument("--debug", action="store_true", default=True, help="Show debug overlay")
    parser.add_argument("--no-debug", action="store_false", dest="debug", help="Hide debug overlay")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Config file path")
    parser.add_argument("--zones", type=Path, default=DEFAULT_ZONES, help="Zones file path")
    parser.add_argument("--hsv", action="store_true", help="Use HSV fallback instead of YOLO (for debugging)")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.hsv:
        config["use_hsv_fallback"] = True
    source = args.source if args.source is not None else config.get("source", 0)
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    if args.calibrate:
        run_calibration(source, args.zones, config)
        return

    zones = load_zones(args.zones)
    if not zones or "hoop_zone" not in zones or "backboard_zone" not in zones:
        print("Zones not found. Run calibration first: python main.py --calibrate --source 0")
        sys.exit(1)

    # Run mode
    run_detection_loop(source, config, zones, args.debug, args.config, args.zones)


def run_detection_loop(source, config, zones, debug, config_path, zones_path):
    """Main detection/tracking/classification loop."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open source {source}")
        sys.exit(1)

    # Get actual FPS for video files
    fps_source = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dt = 1.0 / fps_source

    det_cfg = config.get("detection", config)
    trk_cfg = config.get("tracking", config)
    cls_cfg = config.get("classification", config)

    detector = BallDetector(
        model_path=det_cfg.get("model", "yolov8n.pt"),
        custom_model_path=det_cfg.get("custom_model_path"),
        ball_class_id=det_cfg.get("ball_class_id", 32),
        confidence_threshold=det_cfg.get("confidence_threshold", 0.35),
        inference_size=det_cfg.get("inference_size", 416),
        use_hsv_fallback=det_cfg.get("use_hsv_fallback", False),
        hue_low=det_cfg.get("hue_low", 0),
        hue_high=det_cfg.get("hue_high", 10),
        hue_low2=det_cfg.get("hue_low2", 170),
        hue_high2=det_cfg.get("hue_high2", 180),
        sat_min=det_cfg.get("sat_min", 100),
        val_min=det_cfg.get("val_min", 100),
    )

    tracker = BallTracker(
        max_occlusion_frames=trk_cfg.get("max_occlusion_frames", 5),
        history_sec=trk_cfg.get("history_sec", 1.5),
        fps=fps_source,
        use_kalman=trk_cfg.get("use_kalman", False),
        association_max_distance=trk_cfg.get("association_max_distance", 80),
    )

    classifier = EventClassifier(
        hoop_zone=zones["hoop_zone"],
        backboard_zone=zones["backboard_zone"],
        attempt_roi=zones.get("attempt_roi"),
        K_hit_frames=cls_cfg.get("K_hit_frames", 2),
        K_backboard_frames=cls_cfg.get("K_backboard_frames", 2),
        T_end_sec=cls_cfg.get("T_end_sec", 3.0),
        cooldown_sec=cls_cfg.get("cooldown_sec", 1.0),
        downward_velocity_threshold=cls_cfg.get("downward_velocity_threshold", 50),
        T_hit_max_sec=cls_cfg.get("T_hit_max_sec", 2.0),
    )

    infer_every = det_cfg.get("infer_every_n_frames", 1)
    roi_crop = config.get("attempt_roi_crop")

    window_name = "SmartHoop"
    cv2.namedWindow(window_name)

    frame_count = 0
    last_detections = []
    fps_smooth = 0.0
    t_prev = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if roi_crop:
            x, y, w, h = roi_crop
            frame = frame[int(y) : int(y + h), int(x) : int(x + w)]

        frame_count += 1
        run_inference = (frame_count % infer_every) == 1
        if run_inference:
            last_detections = detector.detect(frame)
        detections = last_detections

        tracked = tracker.update(detections, dt=dt)
        result = classifier.update(tracked, fps=fps_source)

        if result.outcome is not None:
            log_event(result.outcome.value, classifier.get_attempt_id(), tracked, debug)

        if debug:
            frame = draw_overlay(
                frame, zones, tracked, result, fps_smooth
            )

        t_now = time.perf_counter()
        elapsed = t_now - t_prev
        fps_smooth = 0.9 * fps_smooth + 0.1 * (1.0 / elapsed) if elapsed > 0 else 0
        t_prev = t_now

        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def log_event(event: str, attempt_id: int, tracked, debug: bool):
    """Print structured JSON log for classification event."""
    import datetime
    ts = datetime.datetime.now().isoformat()
    conf = getattr(tracked, "conf", 0) if tracked else 0
    notes = "rim-crossing" if event == "hit" else ""
    log = {
        "event": event,
        "timestamp": ts,
        "attempt_id": attempt_id,
        "confidence": conf,
        "notes": notes,
    }
    print(json.dumps(log))


def draw_overlay(frame, zones, tracked, result, fps: float):
    """Draw debug overlay: ball, trajectory, zones, state, counters, FPS."""
    # Zones
    if "hoop_zone" in zones:
        z = zones["hoop_zone"]
        if isinstance(z, ZonePolygon):
            pts = np.array(z.points, dtype=np.int32)
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
        elif isinstance(z, ZoneEllipse):
            cx, cy = int(z.center[0]), int(z.center[1])
            ax, ay = int(z.axes[0]), int(z.axes[1])
            cv2.ellipse(frame, (cx, cy), (ax, ay), z.angle, 0, 360, (0, 255, 0), 2)
    if "backboard_zone" in zones:
        z = zones["backboard_zone"]
        pts = np.array(z.points, dtype=np.int32)
        cv2.polylines(frame, [pts], True, (0, 255, 255), 2)
    if "attempt_roi" in zones:
        z = zones["attempt_roi"]
        pts = np.array(z.points, dtype=np.int32)
        cv2.polylines(frame, [pts], True, (255, 255, 0), 2)

    # Ball and trajectory
    if tracked:
        cx, cy = int(tracked.cx), int(tracked.cy)
        if tracked.bbox:
            x1, y1, x2, y2 = [int(v) for v in tracked.bbox]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
        hist = list(tracked.history)
        if len(hist) >= 2:
            pts = np.array([[int(p[0]), int(p[1])] for p in hist], dtype=np.int32)
            cv2.polylines(frame, [pts], False, (0, 200, 0), 2)

    # State and counters
    state_str = result.state.value
    cv2.putText(
        frame, f"State: {state_str}", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
    )
    cv2.putText(
        frame, f"hit: {result.hit} | backboard: {result.backboard} | miss: {result.miss}",
        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
    )
    cv2.putText(
        frame, f"FPS: {fps:.1f}", (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
    )
    return frame


if __name__ == "__main__":
    main()
