"""Shared device configuration for the SmartHoop shortdemo dispatch layer.

Reads from config.yaml in the project root so there is a single source of truth.
Import this on the laptop side; do NOT import on MicroPython devices.
"""

from pathlib import Path
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_raw = yaml.safe_load(open(_PROJECT_ROOT / "config.yaml"))
_dev = _raw.get("devices", {})
_pi5 = _dev.get("pi5", {})

DISPATCH_ENABLED: bool = _raw.get("dispatch_enabled", False)

PI5_IP: str   = _pi5.get("ip", "")
PI5_PORT: int = int(_pi5.get("port", 5000))
PI5_SHOT_URL: str = f"http://{PI5_IP}:{PI5_PORT}/shot"
