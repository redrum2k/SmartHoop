"""
pi5_server.py — SmartHoop Raspberry Pi 5 Flask server

Receives shot events from the laptop (POST /shot) and sitting-timer alerts
from the Pico W chair sensor (POST /sitting_alert), synthesises a randomised
TTS phrase via pyttsx3, and plays it through the Pi's audio output using aplay.

Dependencies: flask>=3.0, pyttsx3>=2.90, espeak-ng (system package)
"""

import logging
import os
import random
import subprocess
import tempfile
import threading

import pyttsx3
from flask import Flask, jsonify, request

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Phrase bank
# ---------------------------------------------------------------------------

PHRASES: dict[str, list[str]] = {
    "hit": ["Nice shot!", "Swish!", "That's money!", "Nothing but net!", "Bucket!"],
    "backboard": [
        "Bank shot!",
        "Off the glass!",
        "Nice bank!",
        "Use the backboard!",
        "Glass works!",
    ],
    "miss": [
        "Brick!",
        "Try again!",
        "Keep shooting!",
        "Oh so close!",
        "You'll get it!",
    ],
    "sitting_alert": [
        "Time to stand up!",
        "Movement break!",
        "You've been sitting for 20 minutes!",
        "Stretch it out!",
        "Stand and shoot a few!",
    ],
}

VALID_SHOT_EVENTS = {"hit", "backboard", "miss"}

# ---------------------------------------------------------------------------
# TTS helpers
# ---------------------------------------------------------------------------

_tts_lock = threading.Lock()
_engine: "pyttsx3.Engine | None" = None


def _get_engine() -> "pyttsx3.Engine":
    global _engine
    if _engine is None:
        _engine = pyttsx3.init()
    return _engine


def speak(text: str) -> None:
    """Synthesize *text* to a temp WAV file and play it via aplay.

    Blocks until playback is complete (or fails).  Acquires _tts_lock so that
    concurrent calls are serialised — pyttsx3 is not thread-safe.
    """
    with _tts_lock:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp.name
        tmp.close()
        try:
            engine = _get_engine()
            engine.save_to_file(text, tmp_path)
            engine.runAndWait()
            try:
                result = subprocess.run(
                    ["aplay", tmp_path],
                    check=False,
                    timeout=10,
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    logger.error("aplay exited %d: %s", result.returncode, result.stderr.strip())
            except FileNotFoundError:
                logger.error("aplay not found — install alsa-utils: sudo apt install alsa-utils")
            except subprocess.TimeoutExpired:
                logger.error("aplay timed out after 10 s")
        except Exception as e:
            logger.error("speak() failed: %s", e)
            _engine = None
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _speak_async(text: str) -> None:
    """Fire-and-forget TTS: speak *text* in a background daemon thread."""
    t = threading.Thread(target=speak, args=(text,), daemon=True)
    t.start()


# ---------------------------------------------------------------------------
# Flask application
# ---------------------------------------------------------------------------

app = Flask(__name__)


@app.route("/shot", methods=["POST"])
def shot_endpoint():
    """Receive a shot event from the laptop and respond with a spoken phrase.

    Expected JSON body::

        {"event": "hit"|"backboard"|"miss", "hits": int, "attempts": int}

    Returns 400 if ``event`` is missing or not one of the valid values.
    Returns 200 with ``{"status": "ok", "event": event, "phrase": phrase}``.
    """
    body = request.get_json(silent=True) or {}

    event = body.get("event")
    if event not in VALID_SHOT_EVENTS:
        logger.warning("POST /shot — invalid event: %r", event)
        return jsonify({"error": "invalid event"}), 400

    hits = body.get("hits")
    attempts = body.get("attempts")
    logger.info("POST /shot  event=%s  hits=%s  attempts=%s", event, hits, attempts)

    phrase = random.choice(PHRASES[event])
    _speak_async(phrase)

    return jsonify({"status": "ok", "event": event, "phrase": phrase}), 200


@app.route("/sitting_alert", methods=["POST"])
def sitting_alert_endpoint():
    """Receive a sitting-timer alert from the Pico W chair sensor.

    The request body is optional; the endpoint fires regardless of its content.
    Returns 200 with ``{"status": "ok", "phrase": phrase}``.
    """
    logger.info("POST /sitting_alert — motivational prompt triggered")

    phrase = random.choice(PHRASES["sitting_alert"])
    _speak_async(phrase)

    return jsonify({"status": "ok", "phrase": phrase}), 200


@app.route("/healthz", methods=["GET"])
def healthz():
    return jsonify({"status": "ok"}), 200


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("SmartHoop Pi5 server starting on port 5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
