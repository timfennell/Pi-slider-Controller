#!/usr/bin/env python3
"""
gamepad.py — 8BitDo Pro 2 async gamepad reader for PiSlider.

Reads the controller via Linux evdev and publishes named events
to an asyncio queue consumed by the cinematic engine and joystick handler.

8BitDo Pro 2 axis/button map (Linux HID mode, XInput-style):
  ABS_X        Left stick X    → Pan
  ABS_Y        Left stick Y    → Tilt  (inverted: up = positive tilt)
  ABS_RX       Right stick X   → Slider
  ABS_RY       Right stick Y   → (future: focus / zoom)
  ABS_Z        L2 analog       → slow modifier (analog)
  ABS_RZ       R2 analog       → fast modifier (analog)
  ABS_HAT0X    D-pad X         → nudge pan
  ABS_HAT0Y    D-pad Y         → nudge tilt / slider
  BTN_SOUTH    A               → record start/stop
  BTN_EAST     B               → return to start
  BTN_NORTH    Y               → arctan lock toggle
  BTN_WEST     X               → add keyframe
  BTN_TL       L1              → slow speed modifier (held)
  BTN_TR       R1              → fast speed modifier (held)
  BTN_START    Start/Menu      → play programmed move
  BTN_SELECT   Select/View     → stop / e-stop
  BTN_THUMBL   Left stick click  → set origin
  BTN_THUMBR   Right stick click → (reserved)
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional, Dict, Any

logger = logging.getLogger("PiSlider.Gamepad")

# ─── AXIS / BUTTON CONSTANTS ─────────────────────────────────────────────────
# (evdev event codes — same regardless of evdev import availability)
EV_ABS    = 3
EV_KEY    = 1
EV_SYN    = 0

ABS_X     = 0
ABS_Y     = 1
ABS_RX    = 3
ABS_RY    = 4
ABS_Z     = 2    # L2
ABS_RZ    = 5    # R2
ABS_HAT0X = 16
ABS_HAT0Y = 17

BTN_SOUTH  = 304   # A
BTN_EAST   = 305   # B
BTN_NORTH  = 308   # Y
BTN_WEST   = 307   # X
BTN_TL     = 310   # L1
BTN_TR     = 311   # R1
BTN_SELECT = 314
BTN_START  = 315
BTN_THUMBL = 317
BTN_THUMBR = 318

# Axis deadzone (raw value out of 32767)
DEADZONE = 2000
AXIS_MAX = 32767.0

# Normalize raw axis value to [-1.0, 1.0] with deadzone
def _norm(raw: int) -> float:
    if abs(raw) < DEADZONE:
        return 0.0
    sign = 1 if raw > 0 else -1
    return sign * (abs(raw) - DEADZONE) / (AXIS_MAX - DEADZONE)


class GamepadEvent:
    """A named event from the gamepad."""
    __slots__ = ("name", "value", "raw")

    def __init__(self, name: str, value: Any, raw: int = 0):
        self.name  = name    # e.g. "axis_pan", "btn_record"
        self.value = value   # float for axes, True/False for buttons
        self.raw   = raw


class GamepadReader:
    """
    Async evdev gamepad reader.

    Usage:
        reader = GamepadReader(event_queue)
        asyncio.create_task(reader.run())

    Events published to event_queue (asyncio.Queue):
        axis_slider  float [-1, 1]   right stick X
        axis_pan     float [-1, 1]   left stick X
        axis_tilt    float [-1, 1]   left stick Y (inverted)
        axis_l2      float [0, 1]    L2 analog
        axis_r2      float [0, 1]    R2 analog
        dpad_x       int  -1/0/1     D-pad horizontal
        dpad_y       int  -1/0/1     D-pad vertical
        btn_record   bool            A button
        btn_return   bool            B button
        btn_arctan   bool            Y button (toggle)
        btn_keyframe bool            X button (add keyframe)
        btn_l1       bool            L1 (slow modifier, held)
        btn_r1       bool            R1 (fast modifier, held)
        btn_play     bool            Start
        btn_stop     bool            Select (e-stop)
        btn_origin   bool            Left stick click (set origin)
    """

    def __init__(self, queue: asyncio.Queue, device_path: Optional[str] = None):
        self._queue  = queue
        self._path   = device_path
        self._stop   = asyncio.Event()
        self.connected = False

        # Current axis state (for publishing deltas only when changed)
        self._axes: Dict[str, float] = {
            "axis_slider": 0.0,
            "axis_pan":    0.0,
            "axis_tilt":   0.0,
            "axis_l2":     0.0,
            "axis_r2":     0.0,
        }
        self._buttons: Dict[str, bool] = {}

    def stop(self):
        self._stop.set()

    async def run(self):
        """Main loop — auto-detects device, reads events, publishes to queue."""
        while not self._stop.is_set():
            path = self._path or await self._find_device()
            if not path:
                logger.warning("Gamepad: no device found, retrying in 5s…")
                await asyncio.sleep(5)
                continue

            logger.info(f"Gamepad: opening {path}")
            try:
                await self._read_loop(path)
            except Exception as e:
                logger.warning(f"Gamepad: disconnected ({e}), retrying in 3s…")
                self.connected = False
                await self._queue.put(GamepadEvent("gamepad_disconnected", False))
                await asyncio.sleep(3)

    async def _find_device(self) -> Optional[str]:
        """Scan /dev/input/js* and /dev/input/event* for a gamepad."""
        # Prefer /dev/input/js0 (joystick interface — simpler)
        for candidate in ["/dev/input/js0", "/dev/input/js1"]:
            if os.path.exists(candidate):
                return candidate
        # Fallback: scan event devices
        try:
            import evdev
            for path in evdev.list_devices():
                try:
                    dev = evdev.InputDevice(path)
                    caps = dev.capabilities()
                    if EV_ABS in caps and EV_KEY in caps:
                        return path
                except Exception:
                    pass
        except ImportError:
            pass
        return None

    async def _read_loop(self, path: str):
        """
        Read events from device. Supports both:
        - evdev InputDevice (preferred, event interface)
        - /dev/input/js* via raw struct read (fallback)
        """
        try:
            import evdev
            dev = evdev.InputDevice(path)
            self.connected = True
            await self._queue.put(GamepadEvent("gamepad_connected", True))
            logger.info(f"Gamepad connected: {dev.name}")

            async for event in dev.async_read_loop():
                if self._stop.is_set():
                    break
                self._handle_evdev_event(event.type, event.code, event.value)

        except ImportError:
            # Fallback: raw js0 protocol (16-byte struct: time, value, type, number)
            import struct
            self.connected = True
            await self._queue.put(GamepadEvent("gamepad_connected", True))
            logger.info(f"Gamepad (js0 fallback): {path}")

            with open(path, "rb") as f:
                while not self._stop.is_set():
                    data = await asyncio.to_thread(f.read, 8)
                    if len(data) < 8:
                        break
                    _, value, typ, number = struct.unpack("IhBB", data)
                    self._handle_js0_event(typ, number, value)
                    await asyncio.sleep(0)

    def _handle_evdev_event(self, typ: int, code: int, value: int):
        """Map evdev events to named gamepad events."""
        if typ == EV_ABS:
            self._handle_abs(code, value)
        elif typ == EV_KEY:
            self._handle_key(code, bool(value))

    def _handle_js0_event(self, typ: int, number: int, value: int):
        """Map js0 raw events (type 1=button, 2=axis) to named events."""
        if typ == 2:   # axis
            # js0 axis numbers for 8BitDo Pro 2 (XInput mode):
            js0_axis_map = {
                0: ("axis_pan",    False),   # Left X
                1: ("axis_tilt",   True),    # Left Y (invert)
                2: ("axis_slider", False),   # Right X
                3: (None,          False),   # Right Y (future)
                4: ("axis_l2",     False),   # L2
                5: ("axis_r2",     False),   # R2
                6: ("dpad_x",      False),
                7: ("dpad_y",      True),
            }
            if number in js0_axis_map:
                name, invert = js0_axis_map[number]
                if name is None:
                    return
                if name.startswith("dpad"):
                    v = -1 if value < -DEADZONE else (1 if value > DEADZONE else 0)
                    if invert:
                        v = -v
                    self._put(GamepadEvent(name, v, value))
                else:
                    norm = _norm(value)
                    if invert:
                        norm = -norm
                    if abs(norm - self._axes.get(name, 0)) > 0.01:
                        self._axes[name] = norm
                        self._put(GamepadEvent(name, norm, value))

        elif typ == 1:  # button
            js0_btn_map = {
                0: "btn_record",    # A
                1: "btn_return",    # B
                2: "btn_keyframe",  # X
                3: "btn_arctan",    # Y
                4: "btn_l1",
                5: "btn_r1",
                6: "btn_select",    # Select
                7: "btn_play",      # Start
                10: "btn_origin",   # L3
                11: "btn_stop",     # R3
            }
            if number in js0_btn_map:
                name = js0_btn_map[number]
                pressed = bool(value)
                if self._buttons.get(name) != pressed:
                    self._buttons[name] = pressed
                    self._put(GamepadEvent(name, pressed, value))

    def _handle_abs(self, code: int, value: int):
        mapping = {
            ABS_X:     ("axis_pan",    False),
            ABS_Y:     ("axis_tilt",   True),    # invert Y
            ABS_RX:    ("axis_slider", False),
            ABS_RY:    (None,          False),
            ABS_Z:     ("axis_l2",     False),
            ABS_RZ:    ("axis_r2",     False),
        }
        if code in mapping:
            name, invert = mapping[code]
            if name is None:
                return
            norm = _norm(value)
            if invert:
                norm = -norm
            if abs(norm - self._axes.get(name, 0)) > 0.01:
                self._axes[name] = norm
                self._put(GamepadEvent(name, norm, value))

        elif code == ABS_HAT0X:
            v = -1 if value < 0 else (1 if value > 0 else 0)
            self._put(GamepadEvent("dpad_x", v, value))
        elif code == ABS_HAT0Y:
            v = -1 if value < 0 else (1 if value > 0 else 0)
            self._put(GamepadEvent("dpad_y", v, value))

    def _handle_key(self, code: int, pressed: bool):
        btn_map = {
            BTN_SOUTH:  "btn_record",
            BTN_EAST:   "btn_return",
            BTN_NORTH:  "btn_arctan",
            BTN_WEST:   "btn_keyframe",
            BTN_TL:     "btn_l1",
            BTN_TR:     "btn_r1",
            BTN_START:  "btn_play",
            BTN_SELECT: "btn_stop",
            BTN_THUMBL: "btn_origin",
            BTN_THUMBR: "btn_stop_r",
        }
        if code in btn_map:
            name = btn_map[code]
            if self._buttons.get(name) != pressed:
                self._buttons[name] = pressed
                self._put(GamepadEvent(name, pressed, int(pressed)))

    def _put(self, event: GamepadEvent):
        """Non-blocking put — drop if queue is full (stale input)."""
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            pass
