# pico_hoop.py — SmartHoop Raspberry Pi Pico W firmware
# Flash as main.py on the Pico W.
# MicroPython only — no external libraries required.

import network
import utime
import ujson
import machine
import uasyncio as asyncio

# ---------------------------------------------------------------------------
# WiFi credentials — fill in before flashing
# ---------------------------------------------------------------------------
WIFI_SSID = "BU Guest (unencrypted)"
WIFI_PASSWORD = ""

# ---------------------------------------------------------------------------
# WiFi connection
# ---------------------------------------------------------------------------

def connect_wifi(ssid: str, password: str, retries: int = 10) -> str:
    """Connect to WiFi. Retry up to `retries` times. Return IP on success."""
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    if wlan.isconnected():
        return wlan.ifconfig()[0]

    print(f"Connecting to WiFi SSID: {ssid}")
    wlan.connect(ssid, password)

    for attempt in range(retries):
        if wlan.isconnected():
            ip = wlan.ifconfig()[0]
            print(f"WiFi connected. IP: {ip}")
            return ip
        print(f"  Waiting for WiFi... attempt {attempt + 1}/{retries}")
        if attempt < retries - 1:  # only sleep if there's another attempt coming
            utime.sleep_ms(1000)
    else:
        raise RuntimeError("WiFi connection failed after {} retries".format(retries))


# ---------------------------------------------------------------------------
# I2C LCD driver (HD44780 via PCF8574 backpack, 4-bit mode)
#
# PCF8574 pin mapping (standard wiring):
#   P0 = RS   P1 = RW   P2 = EN   P3 = Backlight
#   P4 = D4   P5 = D5   P6 = D6   P7 = D7
# ---------------------------------------------------------------------------

_LCD_BL  = 0x08  # P3 — backlight always on
_LCD_EN  = 0x04  # P2
_LCD_RW  = 0x02  # P1 — always 0 (write)
_LCD_RS  = 0x01  # P0

_LCD_BUF = bytearray(1)  # pre-allocated buffer for LCD I2C writes

_RESP_200 = (
    b"HTTP/1.0 200 OK\r\n"
    b"Content-Type: application/json\r\n"
    b"\r\n"
    b'{"status":"ok"}'
)
_RESP_404 = (
    b"HTTP/1.0 404 Not Found\r\n"
    b"Content-Type: text/plain\r\n"
    b"\r\n"
    b"Not Found"
)


def lcd_scan_addr(i2c) -> int:
    """Return I2C address of the PCF8574 backpack (0x27 or 0x3F)."""
    found = i2c.scan()
    for addr in (0x27, 0x3F):
        if addr in found:
            return addr
    raise RuntimeError("LCD not found on I2C bus (scanned: {})".format(found))


def lcd_send_nibble(i2c, addr: int, nibble: int, rs: int):
    """Send a 4-bit nibble to the LCD via PCF8574.

    nibble — upper 4 bits that will be placed on P4–P7
    rs     — 0 = command, 1 = data
    """
    # Build the base byte: data nibble in upper 4 bits + backlight + RS
    base = ((nibble & 0x0F) << 4) | _LCD_BL | (rs & _LCD_RS)

    # Pulse EN high then low
    _LCD_BUF[0] = base | _LCD_EN
    i2c.writeto(addr, _LCD_BUF)
    utime.sleep_ms(1)
    _LCD_BUF[0] = base & ~_LCD_EN
    i2c.writeto(addr, _LCD_BUF)
    utime.sleep_ms(1)


def lcd_send_byte(i2c, addr: int, byte: int, rs: int):
    """Send a full byte as two nibbles, high nibble first."""
    lcd_send_nibble(i2c, addr, (byte >> 4) & 0x0F, rs)
    lcd_send_nibble(i2c, addr, byte & 0x0F, rs)


def lcd_init(i2c, addr: int):
    """Full HD44780 4-bit initialisation sequence."""
    utime.sleep_ms(50)            # Power-on delay

    # Step 1-3: Send 0x03 nibble three times (8-bit mode compatibility)
    for _ in range(3):
        lcd_send_nibble(i2c, addr, 0x03, 0)
        utime.sleep_ms(5)

    # Step 4: Switch to 4-bit mode
    lcd_send_nibble(i2c, addr, 0x02, 0)
    utime.sleep_ms(1)

    # Step 5: Function set — 2 lines, 5x8 font (0x28)
    lcd_send_byte(i2c, addr, 0x28, 0)
    utime.sleep_ms(1)

    # Step 6: Display off (0x08)
    lcd_send_byte(i2c, addr, 0x08, 0)
    utime.sleep_ms(1)

    # Step 7: Clear display (0x01)
    lcd_send_byte(i2c, addr, 0x01, 0)
    utime.sleep_ms(2)

    # Step 8: Entry mode — increment cursor, no display shift (0x06)
    lcd_send_byte(i2c, addr, 0x06, 0)
    utime.sleep_ms(1)

    # Step 9: Display on, cursor off, blink off (0x0C)
    lcd_send_byte(i2c, addr, 0x0C, 0)
    utime.sleep_ms(1)


def lcd_clear(i2c, addr: int):
    """Clear display and return cursor to home."""
    lcd_send_byte(i2c, addr, 0x01, 0)
    utime.sleep_ms(2)


def lcd_set_cursor(i2c, addr: int, row: int, col: int):
    """Move cursor to (row, col). row 0 → 0x80 base, row 1 → 0xC0 base."""
    row_offsets = (0x80, 0xC0)
    cmd = row_offsets[row & 0x01] | (col & 0x0F)
    lcd_send_byte(i2c, addr, cmd, 0)


def lcd_write_string(i2c, addr: int, text: str):
    """Write a string at the current cursor position."""
    for ch in text:
        lcd_send_byte(i2c, addr, ord(ch), 1)


def lcd_write_line(i2c, addr: int, row: int, text: str):
    """Write text to a row, padded or truncated to exactly 16 characters."""
    line = text[:16].ljust(16)
    lcd_set_cursor(i2c, addr, row, 0)
    lcd_write_string(i2c, addr, line)


# ---------------------------------------------------------------------------
# PWM tone helpers
# ---------------------------------------------------------------------------

async def play_tone(pwm_pin, freq: int, duration_ms: int):
    """Play a single tone at freq Hz for duration_ms milliseconds."""
    pwm_pin.freq(freq)
    # 50% duty cycle — PWM duty is 0–65535 on MicroPython
    pwm_pin.duty_u16(32768)
    await asyncio.sleep_ms(duration_ms)
    pwm_pin.duty_u16(0)


async def play_pattern(pwm_pin, event: str):
    """Play a tone pattern corresponding to the shot event."""
    if event == "hit":
        # Three ascending tones: C5, E5, G5
        await play_tone(pwm_pin, 523, 80)
        await play_tone(pwm_pin, 659, 80)
        await play_tone(pwm_pin, 784, 80)
    elif event == "backboard":
        # Two medium beeps separated by silence
        await play_tone(pwm_pin, 440, 60)
        await asyncio.sleep_ms(60)
        await play_tone(pwm_pin, 440, 60)
    elif event == "miss":
        # Two descending tones
        await play_tone(pwm_pin, 330, 150)
        await play_tone(pwm_pin, 262, 150)
    else:
        # Unknown event — single short beep
        await play_tone(pwm_pin, 440, 50)


# ---------------------------------------------------------------------------
# Shot handler — update LCD + play tone
# ---------------------------------------------------------------------------

_EVENT_LABELS = {
    "hit":       "  ** HIT! **    ",
    "backboard": "  BACKBOARD!    ",
    "miss":      "    MISS...     ",
}


async def handle_shot(body: bytes, i2c, lcd_addr: int, speaker_pin, writer):
    """Parse JSON body, update LCD rows, send HTTP response, play tone."""
    try:
        data = ujson.loads(body)
    except Exception as exc:
        print("handle_shot: JSON parse error:", exc)
        writer.write(_RESP_200)
        await writer.drain()
        return

    event    = data.get("event", "")
    hits     = data.get("hits", 0)
    attempts = data.get("attempts", 0)

    # Row 0: score
    score_text = "SCORE: {}/{}".format(hits, attempts)
    lcd_write_line(i2c, lcd_addr, 0, score_text)

    # Row 1: event label (pre-formatted, 16 chars)
    label = _EVENT_LABELS.get(event, event.center(16))
    lcd_write_line(i2c, lcd_addr, 1, label)

    # Respond immediately before playing tones
    writer.write(_RESP_200)
    await writer.drain()

    # Audio feedback
    await play_pattern(speaker_pin, event)


# ---------------------------------------------------------------------------
# Async HTTP server
# ---------------------------------------------------------------------------

async def serve_client(reader, writer, i2c, lcd_addr, speaker_pin):
    """Handle a single HTTP client connection."""
    try:
        # --- Read request line ---
        request_line = await reader.readline()
        request_line = request_line.decode("utf-8").strip()
        print("HTTP:", request_line)

        # --- Skip headers, track Content-Length ---
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

        # --- Read body ---
        body = b""
        if content_length > 0:
            body = await reader.read(content_length)

        # --- Route request ---
        parts = request_line.split(" ")
        method = parts[0] if len(parts) > 0 else ""
        path   = parts[1] if len(parts) > 1 else ""

        if method == "POST" and path == "/shot":
            await handle_shot(body, i2c, lcd_addr, speaker_pin, writer)
        else:
            writer.write(_RESP_404)
            await writer.drain()

    except Exception as exc:
        print("serve_client error:", exc)
    finally:
        writer.close()
        await writer.wait_closed()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def main():
    # 1. Connect WiFi
    try:
        ip = connect_wifi(WIFI_SSID, WIFI_PASSWORD)
    except Exception as exc:
        print("WiFi error:", exc)
        ip = "0.0.0.0"

    # 2. Init I2C (GP4=SDA, GP5=SCL, bus 0)
    i2c = machine.I2C(0,
                      sda=machine.Pin(4),
                      scl=machine.Pin(5),
                      freq=400000)

    # 3. Scan for LCD address
    try:
        lcd_addr = lcd_scan_addr(i2c)
        print("LCD found at 0x{:02X}".format(lcd_addr))
    except RuntimeError as exc:
        print("LCD scan error:", exc)
        lcd_addr = 0x27  # fallback — display ops will silently fail

    # 4. Init LCD and show startup message
    try:
        lcd_init(i2c, lcd_addr)
        lcd_write_line(i2c, lcd_addr, 0, "SmartHoop Ready")
        lcd_write_line(i2c, lcd_addr, 1, ip)
    except Exception as exc:
        print("LCD init error:", exc)

    # 5. Init PWM speaker on GP0
    speaker_pin = machine.PWM(machine.Pin(0))
    speaker_pin.duty_u16(0)  # silent until needed

    # 6. Start HTTP server on port 80
    print("Starting HTTP server on port 80...")

    async def client_handler(reader, writer):
        await serve_client(reader, writer, i2c, lcd_addr, speaker_pin)

    try:
        server = await asyncio.start_server(client_handler, "0.0.0.0", 80)
        print("HTTP server listening on port 80")
    except Exception as e:
        print("Failed to start server:", e)
        return

    # 7. Loop forever
    while True:
        try:
            await asyncio.sleep(1)
        except Exception as e:
            print("Loop error:", e)


asyncio.run(main())
