"""
pi5_server.py — SmartHoop shortdemo Raspberry Pi 5 server

Receives shot events from the laptop CV pipeline (POST /shot).
On 'hit': plays the air-horn MP3 through the audio output and increments
the I2C LCD scoreboard (HD44780 via PCF8574 backpack, same wiring as pico_hoop).

Pico W devices are NOT used in this branch.

Dependencies (Pi 5 only):
    pip install flask>=3.0 smbus2>=1.1.0
    sudo apt install mpg123

Run:
    python devices/pi5_server.py
"""

import logging
import subprocess
import threading
import time
from pathlib import Path

from flask import Flask, jsonify, request

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
MP3_PATH = _PROJECT_ROOT / "inputs" / "dragon-studio-air-horn-sound-effect-372453.mp3"

# ---------------------------------------------------------------------------
# I2C LCD driver (HD44780 via PCF8574 backpack, 4-bit mode)
#
# PCF8574 pin mapping (standard wiring):
#   P0 = RS   P1 = RW   P2 = EN   P3 = Backlight
#   P4 = D4   P5 = D5   P6 = D6   P7 = D7
#
# Pi 5 wiring: PCF8574 SDA → GPIO2 (pin 3), SCL → GPIO3 (pin 5)
# ---------------------------------------------------------------------------

_LCD_BL = 0x08  # P3 — backlight always on
_LCD_EN = 0x04  # P2
_LCD_RS = 0x01  # P0

_LCD_AVAILABLE = False
_bus = None
_lcd_addr: int = 0x27


def lcd_send_nibble(bus, addr: int, nibble: int, rs: int):
    base = ((nibble & 0x0F) << 4) | _LCD_BL | (rs & _LCD_RS)
    bus.write_byte(addr, base | _LCD_EN)
    time.sleep(0.001)
    bus.write_byte(addr, base & ~_LCD_EN)
    time.sleep(0.001)


def lcd_send_byte(bus, addr: int, byte: int, rs: int):
    lcd_send_nibble(bus, addr, (byte >> 4) & 0x0F, rs)
    lcd_send_nibble(bus, addr, byte & 0x0F, rs)


def lcd_init(bus, addr: int):
    time.sleep(0.05)
    for _ in range(3):
        lcd_send_nibble(bus, addr, 0x03, 0)
        time.sleep(0.005)
    lcd_send_nibble(bus, addr, 0x02, 0)
    time.sleep(0.001)
    lcd_send_byte(bus, addr, 0x28, 0)  # Function set: 2 lines, 5x8
    time.sleep(0.001)
    lcd_send_byte(bus, addr, 0x08, 0)  # Display off
    time.sleep(0.001)
    lcd_send_byte(bus, addr, 0x01, 0)  # Clear display
    time.sleep(0.002)
    lcd_send_byte(bus, addr, 0x06, 0)  # Entry mode: increment, no shift
    time.sleep(0.001)
    lcd_send_byte(bus, addr, 0x0C, 0)  # Display on, cursor off
    time.sleep(0.001)


def lcd_write_line(bus, addr: int, row: int, text: str):
    cmd = (0x80, 0xC0)[row & 0x01]
    lcd_send_byte(bus, addr, cmd, 0)
    line = text[:16].ljust(16)
    for ch in line:
        lcd_send_byte(bus, addr, ord(ch), 1)


def lcd_scan_addr(bus) -> int:
    for addr in (0x27, 0x3F):
        try:
            bus.read_byte(addr)
            return addr
        except OSError:
            pass
    raise RuntimeError("LCD not found at 0x27 or 0x3F")


def _init_lcd():
    global _LCD_AVAILABLE, _bus, _lcd_addr
    try:
        import smbus2
        _bus = smbus2.SMBus(1)
        _lcd_addr = lcd_scan_addr(_bus)
        lcd_init(_bus, _lcd_addr)
        lcd_write_line(_bus, _lcd_addr, 0, "SmartHoop Demo")
        lcd_write_line(_bus, _lcd_addr, 1, "Ready")
        _LCD_AVAILABLE = True
        logger.info("LCD initialised at 0x%02X", _lcd_addr)
    except ImportError:
        logger.warning("smbus2 not installed — LCD disabled (pip install smbus2)")
    except Exception as e:
        logger.warning("LCD init failed: %s — display disabled", e)


_init_lcd()

# ---------------------------------------------------------------------------
# LCD lock + event labels
# ---------------------------------------------------------------------------

_lcd_lock = threading.Lock()

_EVENT_LABELS = {
    "hit":       "  ** HIT! **    ",
    "backboard": "  BACKBOARD!    ",
    "miss":      "    MISS...     ",
}


def _update_lcd(hits: int, attempts: int, event: str):
    if not _LCD_AVAILABLE:
        return
    label = _EVENT_LABELS.get(event, event.center(16))
    try:
        with _lcd_lock:
            lcd_write_line(_bus, _lcd_addr, 0, f"SCORE: {hits}/{attempts}")
            lcd_write_line(_bus, _lcd_addr, 1, label)
    except Exception as e:
        logger.error("LCD write error: %s", e)


# ---------------------------------------------------------------------------
# MP3 playback
# ---------------------------------------------------------------------------

def _play_horn():
    if not MP3_PATH.exists():
        logger.error("MP3 not found: %s", MP3_PATH)
        return
    try:
        subprocess.run(["mpg123", "-q", str(MP3_PATH)], timeout=15, check=False)
    except FileNotFoundError:
        logger.error("mpg123 not found — install with: sudo apt install mpg123")
    except subprocess.TimeoutExpired:
        logger.error("mpg123 timed out after 15 s")
    except Exception as e:
        logger.error("audio playback error: %s", e)


# ---------------------------------------------------------------------------
# Flask application
# ---------------------------------------------------------------------------

app = Flask(__name__)

_hit_count = 0
_attempt_count = 0

VALID_SHOT_EVENTS = {"hit", "backboard", "miss"}


@app.route("/shot", methods=["POST"])
def shot_endpoint():
    """Receive a shot event from the laptop CV pipeline.

    Expected JSON body::

        {"event": "hit"|"backboard"|"miss", "hits": int, "attempts": int}

    On "hit": plays the air-horn MP3 and updates the LCD scoreboard.
    On all events: updates the LCD with current score and event label.
    Returns 200 immediately; audio and LCD updates run in background threads.
    """
    global _hit_count, _attempt_count

    body = request.get_json(silent=True) or {}
    event = body.get("event")
    if event not in VALID_SHOT_EVENTS:
        logger.warning("POST /shot — invalid event: %r", event)
        return jsonify({"error": "invalid event"}), 400

    _hit_count = body.get("hits", _hit_count)
    _attempt_count = body.get("attempts", _attempt_count)
    logger.info("POST /shot  event=%s  hits=%s  attempts=%s", event, _hit_count, _attempt_count)

    threading.Thread(
        target=_update_lcd, args=(_hit_count, _attempt_count, event), daemon=True
    ).start()

    if event == "hit":
        threading.Thread(target=_play_horn, daemon=True).start()

    return jsonify({"status": "ok", "event": event}), 200


@app.route("/healthz", methods=["GET"])
def healthz():
    return jsonify({"status": "ok", "hits": _hit_count, "attempts": _attempt_count}), 200


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("SmartHoop shortdemo Pi 5 server starting on port 5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
