#!/usr/bin/env python3
"""
neopixel_status.py — PiSlider status LED controller
7 NeoPixels on GP24 (pin 18), powered from 5V (pins 2/4), GND common.

Install:  sudo pip install rpi-ws281x --break-system-packages
          Must run as root (DMA access) or via sudo.

LED layout (0-indexed):
  0  Power / app alive
  1  Sequence loop phase  (shutter / motors / waiting)  |  Cinematic: record
  2  File save status                                   |  Cinematic: slider speed
  3  HG phase colour                                    |  Cinematic: pan speed
  4  Progress                                           |  Cinematic: tilt speed
  5  Relay 1
  6  Relay 2

ERROR state: all LEDs solid red.
"""

from __future__ import annotations

import math
import threading
import time
import logging
from typing import Optional

logger = logging.getLogger("PiSlider.LEDs")

# ── Hardware constants ────────────────────────────────────────────────────────
LED_COUNT   = 7
LED_PIN     = 18      # GP24 = BCM 24 … wait — rpi_ws281x uses BCM numbering.
                      # GP24 on the Pi header = BCM 24. We use pin 18 on the
                      # 40-pin header which is BCM 24. Double-check: header pin
                      # 18 = GPIO24 = BCM 24.  ✓
LED_FREQ    = 800_000  # WS2812B data frequency (Hz)
LED_DMA     = 10       # DMA channel — 10 is safe on Pi 4/5
LED_INVERT  = False    # True if using NPN transistor inversion
LED_CHANNEL = 0        # PWM channel (0 for BCM 18/12, 1 for BCM 13/19)
                       # BCM 24 uses channel 0 when driven via PWM alt mode.
                       # Actually BCM 24 doesn't support hardware PWM —
                       # use PCM (LED_CHANNEL=0 with DMA still works via
                       # the ws281x library's SPI/PCM fallback).
LED_BRIGHTNESS = 80    # 0-255 global brightness cap (lower = less blinding at night)

# ── Colour palette ─────────────────────────────────────────────────────────────
# All colours are (R, G, B) tuples at full brightness;
# the library handles the packing.

OFF          = (0,   0,   0)
WHITE        = (255, 255, 255)
GREEN        = (0,   200, 0)
RED          = (200, 0,   0)
BLUE         = (0,   80,  200)
DEEP_BLUE    = (0,   20,  180)
PURPLE       = (120, 0,   180)
AMBER        = (200, 80,  0)
TEAL         = (0,   180, 140)
MAGENTA      = (200, 0,   150)
DIM_GREEN    = (0,   60,  0)
DIM_RED      = (60,  0,   0)

# HG phase colours
HG_PHASE_COLOURS = {
    "day":      TEAL,         # sky blue / teal
    "golden":   AMBER,        # warm amber
    "twilight": BLUE,         # blue
    "night":    DEEP_BLUE,    # deep blue
    "prior":    (40, 40, 40), # grey — tracker not warmed up yet
    "unknown":  (40, 40, 40),
}

# Progress colour: hue rotates from 300° (magenta) → 120° (green) as % increases
def _progress_colour(pct: float) -> tuple:
    """Return RGB for progress %. 0%=magenta(300°), 100%=green(120°)."""
    hue = 300.0 - pct * 180.0   # 300→120 degrees
    hue = hue % 360.0
    # HSV→RGB (S=1, V=1)
    h = hue / 60.0
    i = int(h) % 6
    f = h - int(h)
    q = 1.0 - f
    t = f
    rgb = [
        (1, t, 0), (q, 1, 0), (0, 1, t),
        (0, q, 1), (t, 0, 1), (1, 0, q),
    ][i]
    r, g, b = [int(v * 180) for v in rgb]  # cap at 180 to avoid blinding
    return (r, g, b)


def _dim(colour: tuple, factor: float) -> tuple:
    """Scale a colour by factor 0..1."""
    return tuple(int(c * max(0.0, min(1.0, factor))) for c in colour)


def _motor_speed_colour(velocity_abs: int, max_velocity: int = 28000) -> tuple:
    """Purple, brighter = faster. Off when stopped."""
    if velocity_abs == 0:
        return OFF
    pct = min(1.0, velocity_abs / max_velocity)
    brightness = 0.1 + pct * 0.9
    return _dim(PURPLE, brightness)


# ── NeoPixel driver wrapper ───────────────────────────────────────────────────

try:
    from rpi_ws281x import PixelStrip, Color
    _HAS_WS281X = True
except ImportError:
    _HAS_WS281X = False
    logger.warning("rpi-ws281x not installed — LED status disabled. "
                   "Install with: sudo pip install rpi-ws281x --break-system-packages")


class _Strip:
    """Thin wrapper so the rest of the module doesn't care if hardware is absent."""

    def __init__(self):
        self._strip = None
        if _HAS_WS281X:
            try:
                self._strip = PixelStrip(
                    LED_COUNT, LED_PIN, LED_FREQ, LED_DMA,
                    LED_INVERT, LED_BRIGHTNESS, LED_CHANNEL
                )
                self._strip.begin()
                logger.info(f"NeoPixel strip ready: {LED_COUNT} LEDs on BCM {LED_PIN}")
            except Exception as e:
                logger.warning(f"NeoPixel init failed: {e} — LED status disabled.")
                self._strip = None

    def set(self, index: int, rgb: tuple):
        if self._strip and 0 <= index < LED_COUNT:
            r, g, b = rgb
            self._strip.setPixelColor(index, Color(r, g, b))

    def show(self):
        if self._strip:
            try:
                self._strip.show()
            except Exception as e:
                logger.debug(f"LED show: {e}")

    def all_off(self):
        for i in range(LED_COUNT):
            self.set(i, OFF)
        self.show()

    def all_colour(self, rgb: tuple):
        for i in range(LED_COUNT):
            self.set(i, rgb)
        self.show()

    @property
    def available(self) -> bool:
        return self._strip is not None


# ── Main status controller ────────────────────────────────────────────────────

class StatusLEDs:
    """
    Thread-safe LED status controller.

    The app calls simple state-setter methods; an internal
    background thread handles all animation timing.

    Usage:
        leds = StatusLEDs()
        leds.start()

        leds.set_mode("timelapse_idle")
        leds.set_sequence_phase("shutter")
        leds.set_save_ok()
        leds.set_hg_phase("golden")
        leds.set_progress(0.42)
        leds.set_relay(1, True)
        leds.set_motor_speeds(slider=12000, pan=0, tilt=5000)
        leds.set_error()     # all red
        leds.stop()
    """

    # Modes
    MODES = {
        "startup",
        "timelapse_idle",   # setup, not yet running
        "timelapse_run",    # sequence active
        "macro_idle",
        "macro_run",
        "cinematic",
        "error",
    }

    def __init__(self):
        self._strip       = _Strip()
        self._lock        = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running     = False

        # ── Mutable state (all protected by _lock) ────────────────────────
        self._mode        = "startup"
        self._seq_phase   = "waiting"   # 'shutter' | 'motors' | 'waiting'
        self._save_ok       = None  # True=ok blink, False=error solid, None=idle
        self._save_timer    = 0.0
        self._save_had_drop = False  # latches True permanently once any save fails
        self._hg_phase    = "unknown"
        self._progress    = 0.0         # 0..1
        self._relay1      = False
        self._relay2      = False
        self._shutter_s   = 0.0         # actual shutter duration for LED timing
        self._slider_vel  = 0
        self._pan_vel     = 0
        self._tilt_vel    = 0
        self._recording   = False
        self._error       = False

        # ── Animation clocks ─────────────────────────────────────────────
        self._t0          = time.monotonic()

    # ── Public state setters ──────────────────────────────────────────────────

    def set_mode(self, mode: str):
        with self._lock:
            self._mode  = mode
            self._error = (mode == "error")
            # Reset drop history when starting a fresh sequence
            if mode in ("timelapse_run", "macro_run"):
                self._save_ok       = None
                self._save_had_drop = False

    def set_sequence_phase(self, phase: str, shutter_s: float = 0.0):
        """phase: 'shutter' | 'motors' | 'waiting'"""
        with self._lock:
            self._seq_phase = phase
            self._shutter_s = shutter_s

    def set_save_ok(self):
        """Call after a successful file write."""
        with self._lock:
            self._save_ok    = True
            self._save_timer = time.monotonic()
            # NOTE: _save_had_drop is NOT cleared — once a frame is lost it
            # stays orange for the rest of the sequence so the user always
            # knows the sequence has gaps, even if later saves succeed.

    def set_save_error(self):
        """Call when a file save fails — holds red, latches drop flag permanently."""
        with self._lock:
            self._save_ok       = False
            self._save_had_drop = True

    def set_hg_phase(self, phase: str):
        with self._lock:
            self._hg_phase = phase

    def set_progress(self, fraction: float):
        """fraction 0..1"""
        with self._lock:
            self._progress = max(0.0, min(1.0, fraction))

    def set_relay(self, relay: int, on: bool):
        with self._lock:
            if relay == 1:
                self._relay1 = on
            elif relay == 2:
                self._relay2 = on

    def set_motor_speeds(self, slider: int = 0, pan: int = 0, tilt: int = 0):
        """Absolute velocity values from hw.set_tmc_velocity calls."""
        with self._lock:
            self._slider_vel = abs(slider)
            self._pan_vel    = abs(pan)
            self._tilt_vel   = abs(tilt)

    def set_recording(self, recording: bool):
        with self._lock:
            self._recording = recording

    def set_error(self):
        """All LEDs solid red. Call on crash."""
        with self._lock:
            self._error = True
            self._mode  = "error"

    def clear_error(self):
        with self._lock:
            self._error = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self):
        if not self._strip.available:
            return
        self._running = True
        self._thread  = threading.Thread(
            target=self._loop, daemon=True, name="LED-status"
        )
        self._thread.start()
        logger.info("LED status thread started.")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        self._strip.all_off()

    # ── Animation loop ────────────────────────────────────────────────────────

    def _loop(self):
        TICK = 0.033   # ~30fps refresh
        while self._running:
            try:
                self._render()
            except Exception as e:
                logger.debug(f"LED render: {e}")
            time.sleep(TICK)

    def _render(self):
        with self._lock:
            mode       = self._mode
            error      = self._error
            seq_phase  = self._seq_phase
            shutter_s  = self._shutter_s
            save_ok       = self._save_ok
            save_timer    = self._save_timer
            save_had_drop = self._save_had_drop
            hg_phase   = self._hg_phase
            progress   = self._progress
            relay1     = self._relay1
            relay2     = self._relay2
            slider_vel = self._slider_vel
            pan_vel    = self._pan_vel
            tilt_vel   = self._tilt_vel
            recording  = self._recording

        t = time.monotonic() - self._t0

        # ── Global error override ─────────────────────────────────────────────
        if error:
            self._strip.all_colour(RED)
            return

        # ── Build all 7 LED values ────────────────────────────────────────────
        leds = [OFF] * LED_COUNT

        # ── LED 0: Power / alive — breathing green ────────────────────────────
        breath = (math.sin(t * math.pi * 0.5) ** 2)   # 0..1, period ~4s
        leds[0] = _dim(GREEN, 0.08 + breath * 0.92)

        # ── LED 5 & 6: Relays (same in all modes) ────────────────────────────
        leds[5] = GREEN if relay1 else OFF
        leds[6] = GREEN if relay2 else OFF

        # ══════════════════════════════════════════════════════════════════════
        # CINEMATIC MODE
        # ══════════════════════════════════════════════════════════════════════
        if mode == "cinematic":
            # LED 1: Record — red when recording, green when not
            leds[1] = RED if recording else DIM_GREEN

            # LED 2: Slider speed (purple, brightness = speed)
            leds[2] = _motor_speed_colour(slider_vel, max_velocity=28000)

            # LED 3: Pan speed
            leds[3] = _motor_speed_colour(pan_vel, max_velocity=18000)

            # LED 4: Tilt speed
            leds[4] = _motor_speed_colour(tilt_vel, max_velocity=12000)

        # ══════════════════════════════════════════════════════════════════════
        # IDLE (setup before sequence start) — timelapse or macro
        # ══════════════════════════════════════════════════════════════════════
        elif mode in ("timelapse_idle", "macro_idle", "startup"):
            # LEDs 1-4 breathe green in a slow ripple cascade
            for i in range(1, 5):
                phase_offset = i * 0.4   # stagger each LED by 0.4s
                b = (math.sin((t - phase_offset) * math.pi * 0.5) ** 2)
                leds[i] = _dim(GREEN, 0.08 + b * 0.70)

        # ══════════════════════════════════════════════════════════════════════
        # TIMELAPSE / MACRO SEQUENCE RUNNING
        # ══════════════════════════════════════════════════════════════════════
        elif mode in ("timelapse_run", "macro_run"):

            # ── LED 1: Sequence loop phase ────────────────────────────────────
            if seq_phase == "shutter":
                # Solid green for the shutter duration, then off
                leds[1] = GREEN
            elif seq_phase == "motors":
                # Solid purple while motors are moving
                leds[1] = PURPLE
            else:
                # Waiting: slow dim blue pulse
                pulse = (math.sin(t * math.pi * 0.8) ** 2)
                leds[1] = _dim(BLUE, 0.1 + pulse * 0.5)

            # ── LED 2: File save status ───────────────────────────────────────
            # None  → OFF (no save attempted yet)
            # False → solid RED (current frame failed)
            # True, no prior drops → GREEN flash fading 1.5s, then dim green
            # True, had prior drop → ORANGE flash fading 1.5s, then dim orange
            #   Orange latches for the rest of the sequence — tells the user
            #   there are gaps even when current saves are succeeding.
            if save_ok is False:
                leds[2] = RED
            elif save_ok is True:
                elapsed    = time.monotonic() - save_timer
                ok_colour  = (200, 80, 0) if save_had_drop else GREEN
                dim_colour = (40,  16, 0) if save_had_drop else DIM_GREEN
                if elapsed < 1.5:
                    fade = max(0.0, 1.0 - elapsed / 1.5)
                    leds[2] = _dim(ok_colour, fade)
                else:
                    leds[2] = dim_colour
            else:
                leds[2] = OFF

            # ── LED 3: HG phase colour ────────────────────────────────────────
            leds[3] = HG_PHASE_COLOURS.get(hg_phase, HG_PHASE_COLOURS["unknown"])

            # ── LED 4: Progress ───────────────────────────────────────────────
            # 5-second cycle; on-fraction = progress %; hue shifts magenta→green
            cycle_period = 5.0
            cycle_pos    = (t % cycle_period) / cycle_period  # 0..1
            on_fraction  = progress   # e.g. 0.42 = on for 42% of cycle
            colour       = _progress_colour(progress)
            if progress >= 1.0:
                # Complete — solid green
                leds[4] = GREEN
            elif cycle_pos < on_fraction:
                leds[4] = colour
            else:
                leds[4] = OFF

        # Render
        for i, col in enumerate(leds):
            self._strip.set(i, col)
        self._strip.show()


# ── Module-level singleton ────────────────────────────────────────────────────
leds = StatusLEDs()


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test (run without Pi hardware — prints colour values)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    print("NeoPixel Status LED — self test (no hardware)")
    print(f"WS281X available: {_HAS_WS281X}")
    print()

    # Test colour helpers
    for pct in [0, 25, 50, 75, 100]:
        col = _progress_colour(pct / 100)
        print(f"  Progress {pct:3d}%: RGB{col}")

    print()
    for speed in [0, 5000, 14000, 28000]:
        col = _motor_speed_colour(speed)
        print(f"  Slider vel {speed:5d}: RGB{col}")

    print()
    for phase, col in HG_PHASE_COLOURS.items():
        print(f"  HG phase '{phase}': RGB{col}")

    print()
    print("Breath animation sample (t=0..4s):")
    for t_tenth in range(0, 40, 4):
        t = t_tenth / 10
        b = (math.sin(t * math.pi * 0.5) ** 2)
        brightness = 0.08 + b * 0.92
        col = _dim(GREEN, brightness)
        bar = "█" * int(brightness * 20)
        print(f"  t={t:.1f}s  brightness={brightness:.2f}  {bar}")
