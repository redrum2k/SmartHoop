# pico_chair.py — SmartHoop Chair Pico W firmware
# Flash as main.py on the Pico W built into the chair.
# MicroPython only — no external libraries required.

import network
import utime
import machine
import uasyncio as asyncio

# ---------------------------------------------------------------------------
# Config constants — fill in / tune before flashing
# ---------------------------------------------------------------------------
WIFI_SSID = "BU Guest (unencrypted)"
WIFI_PASSWORD = ""
PI5_IP            = "192.168.1.XXX"   # Raspberry Pi 5 IP
PI5_PORT          = 5000
SITTING_THRESHOLD = 40000   # ADC raw value (0-65535); below this = person sitting
SITTING_MINUTES   = 20      # minutes of continuous sitting before alert fires
POLL_INTERVAL_MS  = 500     # ADC sample interval
SITTING_THRESHOLD_MS = SITTING_MINUTES * 60 * 1000

# ---------------------------------------------------------------------------
# Pre-allocated HTTP response bytes
# ---------------------------------------------------------------------------
_RESP_200 = b"HTTP/1.0 200 OK\r\nContent-Type: application/json\r\n\r\n{\"status\":\"ok\"}"
_RESP_404 = b"HTTP/1.0 404 Not Found\r\nContent-Type: text/plain\r\n\r\nNot Found"

# ---------------------------------------------------------------------------
# WiFi connection
# ---------------------------------------------------------------------------

async def connect_wifi():
    """Connect to WiFi. Retry up to 10 times, 1 s between checks."""
    wlan = network.WLAN(network.STA_IF)
    if not wlan.active():
        wlan.active(True)
    if wlan.isconnected():
        return wlan.ifconfig()[0]

    print("Connecting to WiFi SSID:", WIFI_SSID)
    wlan.connect(WIFI_SSID, WIFI_PASSWORD)

    for attempt in range(10):
        if wlan.isconnected():
            ip = wlan.ifconfig()[0]
            print("WiFi connected. IP:", ip)
            return ip
        print("  Waiting for WiFi... attempt {}/10".format(attempt + 1))
        if attempt < 9:
            await asyncio.sleep_ms(1000)

    raise RuntimeError("WiFi connection failed after 10 retries")


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

class State:
    def __init__(self):
        self.sit_start   = None
        self.alert_fired = False


# ---------------------------------------------------------------------------
# PWM buzzer helpers
# ---------------------------------------------------------------------------

async def play_tone(pwm_pin, freq: int, duration_ms: int):
    """Play a single tone at freq Hz for duration_ms milliseconds."""
    pwm_pin.freq(freq)
    pwm_pin.duty_u16(32768)  # 50% duty cycle
    await asyncio.sleep_ms(duration_ms)
    pwm_pin.duty_u16(0)


async def play_alert_pattern(pwm_pin):
    """3 quick beeps at 880 Hz with 100 ms silences between them."""
    await play_tone(pwm_pin, 880, 100)
    await asyncio.sleep_ms(100)
    await play_tone(pwm_pin, 880, 100)
    await asyncio.sleep_ms(100)
    await play_tone(pwm_pin, 880, 100)


# ---------------------------------------------------------------------------
# HTTP POST to Pi 5 — fire sitting alert
# ---------------------------------------------------------------------------

async def fire_sitting_alert():
    reader = None
    writer = None
    try:
        body = b'{"trigger": "sitting_alert"}'
        reader, writer = await asyncio.open_connection(PI5_IP, PI5_PORT)
        request = (
            "POST /sitting_alert HTTP/1.0\r\n"
            "Host: {}:{}\r\n"
            "Content-Type: application/json\r\n"
            "Content-Length: {}\r\n"
            "\r\n"
        ).format(PI5_IP, PI5_PORT, len(body))
        writer.write(request.encode())
        writer.write(body)
        await writer.drain()
        await reader.read(256)
        print("Sitting alert sent")
    except Exception as e:
        print("Sitting alert failed:", e)
    finally:
        if writer is not None:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Coroutine 1: ADC polling
# ---------------------------------------------------------------------------

async def poll_adc(adc, state):
    """Sample the pressure sensor and fire alert after SITTING_MINUTES."""
    while True:
        try:
            reading = adc.read_u16()
            if reading < SITTING_THRESHOLD:
                if state.sit_start is None:
                    state.sit_start = utime.ticks_ms()
                elif not state.alert_fired:
                    elapsed = utime.ticks_diff(utime.ticks_ms(), state.sit_start)
                    if elapsed >= SITTING_THRESHOLD_MS:
                        state.alert_fired = True          # before the await
                        await fire_sitting_alert()
            else:
                state.sit_start = None
                state.alert_fired = False
        except Exception as e:
            print("poll_adc error:", e)
        await asyncio.sleep_ms(POLL_INTERVAL_MS)          # outside try


# ---------------------------------------------------------------------------
# HTTP server client handler
# ---------------------------------------------------------------------------

async def serve_client(reader, writer, state, speaker_pin):
    """Handle a single inbound HTTP client connection."""
    try:
        # Read request line
        request_line = await reader.readline()
        request_line = request_line.decode("utf-8").strip()
        print("HTTP:", request_line)

        # Read headers, track Content-Length
        content_length = 0
        while True:
            header_line = await reader.readline()
            if header_line in (b"\r\n", b"\n", b""):
                break
            header_str = header_line.decode("utf-8").strip()
            if header_str.lower().startswith("content-length:"):
                try:
                    content_length = int(header_str.split(":", 1)[1].strip())
                except ValueError:
                    pass

        # Read body
        body = b""
        if content_length > 0:
            body = await reader.read(content_length)

        # Route request
        parts  = request_line.split(" ")
        method = parts[0] if len(parts) > 0 else ""
        path   = parts[1] if len(parts) > 1 else ""

        if method == "POST" and path == "/play":
            await play_alert_pattern(speaker_pin)
            writer.write(_RESP_200)
            await writer.drain()
        else:
            writer.write(_RESP_404)
            await writer.drain()

    except Exception as e:
        print("serve_client error:", e)
    finally:
        writer.close()
        await writer.wait_closed()


# ---------------------------------------------------------------------------
# Coroutine 2: HTTP server
# ---------------------------------------------------------------------------

async def http_server(state, speaker_pin):
    async def client_handler(reader, writer):
        await serve_client(reader, writer, state, speaker_pin)
    try:
        server = await asyncio.start_server(client_handler, "0.0.0.0", 80)
        print("HTTP server listening on port 80")
        await server.serve_forever()
    except Exception as e:
        print("Failed to start HTTP server:", e)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def main():
    # 1. Connect WiFi
    try:
        ip = await connect_wifi()
        print("IP address:", ip)
    except Exception as e:
        print("WiFi error:", e)

    # 2. Init ADC on GP26
    adc = machine.ADC(machine.Pin(26))

    # 3. Init PWM buzzer on GP0
    speaker_pin = machine.PWM(machine.Pin(0))
    speaker_pin.duty_u16(0)  # silent until needed

    # 4. Shared state
    state = State()

    # 5. Run HTTP server and ADC polling concurrently
    await asyncio.gather(
        http_server(state, speaker_pin),
        poll_adc(adc, state),
    )


asyncio.run(main())
