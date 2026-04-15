"""Shared device configuration for the SmartHoop hardware dispatch layer.

Reads from config.yaml in the project root so there is a single source of truth.
Import this on the laptop side; do NOT import on MicroPython devices.
"""

from pathlib import Path
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_raw = yaml.safe_load(open(_PROJECT_ROOT / "config.yaml"))
_dev = _raw.get("devices", {})
_pico_hoop = _dev.get("pico_hoop", {})
_pi5 = _dev.get("pi5", {})

DISPATCH_ENABLED: bool = _raw.get("dispatch_enabled", False)

PICO_HOOP_IP: str = _pico_hoop.get("ip", "")
PICO_HOOP_PORT: int = int(_pico_hoop.get("port", 80))
PICO_HOOP_URL: str = f"http://{PICO_HOOP_IP}:{PICO_HOOP_PORT}/shot"

PI5_IP: str = _pi5.get("ip", "")
PI5_PORT: int = int(_pi5.get("port", 5000))
PI5_SHOT_URL: str = f"http://{PI5_IP}:{PI5_PORT}/shot"
PI5_SITTING_URL: str = f"http://{PI5_IP}:{PI5_PORT}/sitting_alert"
