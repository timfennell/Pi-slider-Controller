#!/usr/bin/env python3
"""
app.py — PiSlider Master Orchestrator v2.4

Changes in v2.4:
  - STATE PERSISTENCE: session.json saved on every meaningful change.
    New clients reconnecting get full state via 'init' packet.
    Browser can close/reopen freely; timelapse continues on Pi.
  - HG CALIBRATION SHOT: sequence now starts with a single AE/AWB-ON
    exposure to measure ambient EV, which seeds _smooth_ev so the HG
    engine starts in the right ballpark instead of ramping from cold.
  - MOTION DETECTION TRIGGER: 'picam_motion' trigger mode uses
    background subtraction on the preview stream to fire the shutter
    when pixel change in a user-defined ROI exceeds a threshold.
    Two variants: picam_motion_only (like aux_only) and
    picam_motion_hybrid (like aux_hybrid — fires at deadline if quiet).
  - REMOVED generate_hg_plan: plan runs automatically; no explicit user
    action needed. simulate_plan() still called internally at start.
  - REMOVED build_sky_map: placeholder removed from command table;
    kept as stub returning a clear "not wired" message.
  - SEQUENCE PROGRESS: broadcasts estimated_end_time, estimated_frames,
    current_interval so UI can show live HG interval and end-time.
  - Stop button becomes Reset when idle (handled client-side).
  - HG mode locks exposure engine controls (UI-side only; backend already
    overrides them each frame).
"""

import asyncio
import json
import os
import time
import math
import datetime
import logging
import signal
import atexit
import subprocess
import shutil
import requests
import cv2
import numpy as np
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from hardware import HardwareController
from holygrail import HolyGrailController, HGSettings
from motion_engine import MotionEngine
from slider import TrajectoryPlayer, LinearAxis, RotationAxis
from distributions import CURVE_FUNCTIONS, normalize
from macro_engine import (MacroEngine, MacroSession, ExposureSlot, LensProfile,
                          rail_frame_count, total_image_count, estimated_storage_gb)
from cinematic_engine import (SoftLimitGuard, InertiaEngine, ArcTanTracker,
                               ProgrammedMove, MoveLibrary, Keyframe, RIG_PRESETS)
from gamepad import GamepadReader, GamepadEvent
from neopixel_status import leds as status_leds

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PiSlider")

app    = FastAPI()
hw     = HardwareController()
hg     = HolyGrailController()
engine = MotionEngine()
player = TrajectoryPlayer(hw)

slider_axis = LinearAxis(hw)
pan_axis    = RotationAxis(hw, addr=1)
tilt_axis   = RotationAxis(hw, addr=2)

# Start LED status thread
status_leds.start()
status_leds.set_mode("startup")

# ─── CLEAN SHUTDOWN ───────────────────────────────────────────────────────────
# Ensure GPIO handles are always released — prevents "GPIO busy" on next start.
def _shutdown_cleanup():
    """Called on any exit: normal, Ctrl+C, kill, or os.execv restart."""
    try:
        status_leds.stop()
    except Exception:
        pass
    try:
        hw.cleanup()
    except Exception:
        pass

atexit.register(_shutdown_cleanup)

def _signal_handler(sig, frame):
    """Handle SIGTERM/SIGINT so atexit fires cleanly."""
    _shutdown_cleanup()
    raise SystemExit(0)

signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT,  _signal_handler)

# ─── MACRO ENGINE INSTANCE ────────────────────────────────────────────────────
_macro_task: Optional[asyncio.Task] = None

# ─── CINEMATIC ENGINE INSTANCES ───────────────────────────────────────────────
_soft_guard    = SoftLimitGuard()
_arctan        = ArcTanTracker()
_move_library  = MoveLibrary()
_inertia: Optional[InertiaEngine] = None
_prog_move: Optional[ProgrammedMove] = None
_prog_task: Optional[asyncio.Task] = None
_gamepad_queue: asyncio.Queue = asyncio.Queue(maxsize=64)
_gamepad_reader: Optional[GamepadReader] = None
_gamepad_task: Optional[asyncio.Task] = None
_cinematic_mode: str = "live"   # "live" | "programmed"

# Video recording state
_recording: bool = False
_record_start_time: Optional[float] = None
_video_output_path: Optional[str] = None

PREVIEW_CONFIG_43  = {
    "size":   (1280, 960),   # full IMX477 4:3 sensor area, downscaled from 4056×3040
    "format": "RGB888"
}
PREVIEW_CONFIG_169 = {
    "size":   (1280, 720),   # 16:9 cinematic — slight sensor crop at top/bottom
    "format": "RGB888"
}
PREVIEW_CONFIG     = PREVIEW_CONFIG_43   # default
STREAM_SIZE_43     = (640, 480)
STREAM_SIZE_169    = (640, 360)
STREAM_SIZE        = STREAM_SIZE_43      # default

SESSION_FILE = Path("/home/tim/.pislider_session.json")

_last_frame:  Optional[np.ndarray] = None
_last_capture_time: float = 0.0   # timestamp of last still capture (for ISP settle guard)
_latest_shot: Optional[bytes]      = None

# Motion detection state — optical flow based
_motion_triggered: bool             = False
_motion_prev_gray: Optional[np.ndarray] = None   # previous frame for LK flow
_motion_consec_hits: int            = 0           # consecutive trigger frames

# ─── CAMERA INIT ──────────────────────────────────────────────────────────────
try:
    from picamera2 import Picamera2
    picam = Picamera2()
    # Use full IMX477 sensor area (4056×3040) downscaled to 1280×960 preview.
    # transform= default keeps full sensor FOV — no ISP centre crop.
    cfg = picam.create_video_configuration(
        main=PREVIEW_CONFIG_43,
        raw={"size": picam.camera_properties["PixelArraySize"]},
    )
    picam.configure(cfg)
    picam.start()
    _HAS_PICAM = True
    logger.info("PiCamera2 (IMX477) started — full sensor FOV at 1280×960.")
except Exception as e:
    logger.warning(f"PiCamera2 not available: {e}")
    # Fallback: try simple preview config (no raw stream)
    try:
        from picamera2 import Picamera2
        picam = Picamera2()
        picam.configure(picam.create_preview_configuration(main=PREVIEW_CONFIG_43))
        picam.start()
        _HAS_PICAM = True
        logger.info("PiCamera2 (IMX477) started — standard preview (full sensor FOV unavailable).")
    except Exception as e2:
        logger.warning(f"PiCamera2 fallback also failed: {e2}")
        picam = None
        _HAS_PICAM = False

# ─── CAMERA RECOVERY ──────────────────────────────────────────────────────────
def _restart_picam() -> bool:
    """
    Attempt to recover from a camera frontend timeout or ISP error.
    Stops the camera, waits briefly, restarts, and immediately re-locks
    HG exposure controls so the preview doesn't revert to auto.
    Returns True if recovery succeeded.
    """
    global picam, _HAS_PICAM
    logger.warning("Camera recovery: attempting stop → restart…")
    try:
        if picam:
            try:
                picam.stop()
            except Exception:
                pass
            time.sleep(1.5)
            active_mode = state.get("active_mode", "timelapse")
            cfg_main = PREVIEW_CONFIG_169 if active_mode == "cinematic" else PREVIEW_CONFIG_43
            try:
                picam.configure(picam.create_video_configuration(
                    main=cfg_main,
                    raw={"size": picam.camera_properties["PixelArraySize"]},
                ))
            except Exception:
                picam.configure(picam.create_preview_configuration(main=cfg_main))
            picam.start()
            time.sleep(0.5)

            # Re-lock HG controls immediately after restart
            if hg.settings.enabled:
                _reapply_hg_after_capture()

            logger.info("Camera recovery: restart successful.")
            return True
    except Exception as e:
        logger.error(f"Camera recovery failed: {e}")
    return False


_DEFAULTS = {
    "active_camera":   "picam",
    "active_mode":     "timelapse",  # timelapse | cinematic | macro
    "camera_orientation": "landscape",  # landscape | portrait_cw | portrait_ccw | inverted
    "cine_fps":           24,           # cinematic recording frame rate: 24|25|30|60
    "save_path":       "/home/tim/Pictures/PiSlider",
    "sony_ssid":       "Sony_A7III_WiFi",
    "sony_ip":         "192.168.122.1",
    "is_running":      False,
    "current_frame":   0,
    "total_frames":    300,
    "manual_interval": 5.0,  # interval used when HG is disabled
    "vibe_delay":      1.0,
    "exp_margin":      0.2,
    "picam_ae":        True,
    "picam_awb":       True,
    "picam_shutter_s": 1/125,
    "picam_iso":       400,
    "picam_kelvin":    5500,
    "pan_min":  -90.0,
    "pan_max":   90.0,
    "tilt_min": -30.0,
    "tilt_max":  30.0,
    "trigger_mode":  "normal",
    "aux_triggered": False,
    "origin_az":   0.0,
    "origin_tilt": 0.0,
    # Motion detection ROI: fraction of frame [x1,y1,x2,y2] 0.0–1.0
    "motion_roi":        [0.25, 0.25, 0.75, 0.75],
    "motion_threshold":  2000,   # frame-diff: total blob area in px² to trigger (tune 500–20000)
    "motion_warmup_frames": 10,  # frames before triggering begins
}

# ─── SESSION HISTORY (for graph tab) ─────────────────────────────────────────
from collections import deque
_session_history: deque = deque(maxlen=2000)   # last 2000 frames in memory
_SESSION_HISTORY_FILE = Path("/home/tim/.pislider_graph_history.json")

def _load_session_history():
    """Load persisted graph history from disk on startup."""
    try:
        if _SESSION_HISTORY_FILE.exists():
            data = json.loads(_SESSION_HISTORY_FILE.read_text())
            for frame in data.get("frames", []):
                _session_history.append(frame)
            logger.info(f"Graph history loaded: {len(_session_history)} frames")
    except Exception as e:
        logger.warning(f"Graph history load failed: {e}")

def _save_session_history():
    """Persist graph history to disk (called every 10 frames)."""
    try:
        _SESSION_HISTORY_FILE.write_text(
            json.dumps({"frames": list(_session_history)}, separators=(',', ':'))
        )
    except Exception as e:
        logger.warning(f"Graph history save failed: {e}")

# ─── STATE PERSISTENCE ────────────────────────────────────────────────────────
def _load_session() -> dict:
    s = dict(_DEFAULTS)
    s["stop_event"]    = asyncio.Event()
    s["aux_triggered"] = False
    if SESSION_FILE.exists():
        try:
            saved = json.loads(SESSION_FILE.read_text())
            safe_keys = [k for k in _DEFAULTS if k not in ("stop_event", "aux_triggered")]
            for k in safe_keys:
                if k in saved:
                    s[k] = saved[k]
            # ALWAYS reset run state on startup — asyncio tasks don't survive restarts.
            # If the server was killed mid-sequence, is_running would be stuck True.
            if s.get("is_running"):
                s["_was_interrupted"] = True   # tells the client to show a warning
            s["is_running"] = False
            # Restore HG settings — strip any unknown fields from old sessions
            if "hg_settings" in saved:
                try:
                    import dataclasses as _dc
                    valid_keys = {f.name for f in _dc.fields(HGSettings)}
                    clean = {k: v for k, v in saved["hg_settings"].items() if k in valid_keys}
                    hg.set_settings(HGSettings(**clean))
                except Exception as e:
                    logger.warning(f"Session HG restore failed: {e}")
            # Restore axis positions
            if "pan_deg"    in saved: pan_axis.current_deg    = saved["pan_deg"]
            if "tilt_deg"   in saved: tilt_axis.current_deg   = saved["tilt_deg"]
            if "slider_mm"  in saved: slider_axis.current_mm  = saved["slider_mm"]
            logger.info(f"Session restored from {SESSION_FILE}")
        except Exception as e:
            logger.warning(f"Session load failed: {e}")
    _load_session_history()
    return s


def save_session():
    """Persist serialisable state to disk so browser reloads and reconnects work."""
    try:
        import dataclasses
        saveable = {k: v for k, v in state.items()
                    if k not in ("stop_event", "aux_triggered") and isinstance(v, (str, int, float, bool, list, dict))}
        saveable["hg_settings"] = {
            k: (v if not isinstance(v, datetime.datetime) else v.isoformat())
            for k, v in dataclasses.asdict(hg.settings).items()
            if k != "start_dt"
        }
        saveable["pan_deg"]    = pan_axis.current_deg
        saveable["tilt_deg"]   = tilt_axis.current_deg
        saveable["slider_mm"]  = slider_axis.current_mm
        SESSION_FILE.write_text(json.dumps(saveable, indent=2))
    except Exception as e:
        logger.error(f"save_session: {e}")
        if "No space left" in str(e):
            # Non-blocking broadcast — don't await inside sync function
            import threading
            def _alert():
                import asyncio as _asyncio
                loop = _asyncio.get_event_loop()
                if loop.is_running():
                    loop.call_soon_threadsafe(
                        loop.create_task,
                        broadcast({"type": "disk_full",
                                   "msg": "⛔ DISK FULL — sequence halted. Free space on destination drive."})
                    )
            threading.Thread(target=_alert, daemon=True).start()


def reset_session():
    """Wipe persisted session and reload defaults."""
    if SESSION_FILE.exists():
        SESSION_FILE.unlink()
    for k, v in _DEFAULTS.items():
        if k not in ("stop_event", "aux_triggered"):
            state[k] = v
    hg.set_settings(HGSettings())
    pan_axis.current_deg   = 0.0
    tilt_axis.current_deg  = 0.0
    slider_axis.current_mm = 0.0
    engine.clear_keyframes()
    logger.info("Session reset to defaults.")


def _keyframes_to_list() -> list:
    """Serialise ProgrammedMove keyframes for the WebSocket client."""
    if not _prog_move:
        return []
    return [
        {
            "index":      i,
            "slider_mm":  round(kf.slider_mm, 3),
            "pan_deg":    round(kf.pan_deg,   3),
            "tilt_deg":   round(kf.tilt_deg,  3),
            "duration_s": kf.duration_s,
            "easing":     kf.easing,
        }
        for i, kf in enumerate(_prog_move.keyframes)
    ]


def _build_macro_session(msg: dict) -> "MacroSession":
    """
    Construct a MacroSession from a WebSocket macro_start message.
    Falls back to persisted state values where fields are absent.
    """
    # Slots
    raw_slots = msg.get("slots", [{}])
    slots = []
    for i, rs in enumerate(raw_slots):
        slots.append(ExposureSlot(
            id               = rs.get("id",    f"slot_{chr(65+i)}"),
            label            = rs.get("label", f"slot {i+1}"),
            enabled          = bool(rs.get("enabled", True)),
            relay1           = bool(rs.get("relay1",  False)),
            relay2           = bool(rs.get("relay2",  False)),
            relay_settle_ms  = int(rs.get("relay_settle_ms",  0)),
            relay_release_ms = int(rs.get("relay_release_ms", 0)),
            iso              = int(rs.get("iso",     400)),
            shutter_s        = float(rs.get("shutter_s", 1/125)),
            kelvin           = int(rs.get("kelvin",  5500)),
            ae               = bool(rs.get("ae",  False)),
            awb              = bool(rs.get("awb", False)),
        ))

    # Lens profile
    lp = msg.get("lens_profile", state.get("macro_lens_profile", {}))
    lens = LensProfile(
        name                = lp.get("name",                "unknown"),
        lens_type           = lp.get("lens_type",           "macro"),
        magnification       = float(lp.get("magnification", 1.0)),
        working_distance_mm = float(lp.get("working_distance_mm", 0.0)),
        notes               = lp.get("notes",               ""),
    )

    return MacroSession(
        project_name         = msg.get("project_name",    "macro_project"),
        orbit_label          = msg.get("orbit_label",     "orbit_001"),
        session_mode         = msg.get("session_mode",    "scan"),
        rail_start_mm        = float(msg.get("rail_start_mm",
                                state.get("macro_rail_start_mm", 0.0))),
        rail_end_mm          = float(msg.get("rail_end_mm",
                                state.get("macro_rail_end_mm", 5.0))),
        rail_step_um         = float(msg.get("rail_step_um",      100.0)),
        rail_soft_min        = float(msg.get("rail_soft_min",    -999.0)),
        rail_soft_max        = float(msg.get("rail_soft_max",     999.0)),
        rotation_mode        = msg.get("rotation_mode",           "full"),
        rotation_start_deg   = float(msg.get("rotation_start_deg",
                                state.get("macro_rotation_start_deg", 0.0))),
        rotation_end_deg     = float(msg.get("rotation_end_deg",
                                state.get("macro_rotation_end_deg", 360.0))),
        num_stacks           = int(msg.get("num_stacks",           36)),
        rotation_easing      = msg.get("rotation_easing",         "even"),
        rotation_axis_angle_deg = float(msg.get("rotation_axis_angle_deg", 90.0)),
        rotation_axis_description = msg.get("rotation_axis_description",   "vertical"),
        aux_enabled          = bool(msg.get("aux_enabled",         False)),
        aux_label            = msg.get("aux_label",                "aux"),
        aux_start_deg        = float(msg.get("aux_start_deg",
                                state.get("macro_aux_start_deg", 0.0))),
        aux_end_deg          = float(msg.get("aux_end_deg",
                                state.get("macro_aux_end_deg", 0.0))),
        aux_easing           = msg.get("aux_easing",               "even"),
        vibe_delay_s         = float(msg.get("vibe_delay_s",       0.5)),
        exp_margin_s         = float(msg.get("exp_margin_s",       0.2)),
        active_camera        = state.get("active_camera",          "picam"),
        lens                 = lens,
        save_path            = msg.get("save_path", state.get("save_path",
                                       "/home/tim/Pictures/PiSlider")),
        slots                = slots,
    )


state = _load_session()


# ─── AUX GPIO INTERRUPT ───────────────────────────────────────────────────────
def _setup_aux_trigger():
    try:
        import lgpio
        def _cb(chip, gpio, level, tick):
            if level == 0:
                state["aux_triggered"] = True
                logger.info("AUX: GPIO trigger fired.")
        lgpio.gpio_claim_alert(hw.gpio_chip, 13, lgpio.FALLING_EDGE)
        lgpio.callback(hw.gpio_chip, 13, lgpio.FALLING_EDGE, _cb)
    except Exception as e:
        logger.warning(f"AUX GPIO setup skipped: {e}")

_setup_aux_trigger()


# ─── PICAM SETTINGS ───────────────────────────────────────────────────────────
def apply_picam_settings():
    if not _HAS_PICAM or not picam:
        return
    try:
        controls = {"AeEnable": state["picam_ae"], "AwbEnable": state["picam_awb"]}
        if not state["picam_ae"]:
            controls["ExposureTime"] = int(state["picam_shutter_s"] * 1_000_000)
            controls["AnalogueGain"] = state["picam_iso"] / 100.0
        if not state["picam_awb"]:
            controls["ColourTemperature"] = int(state["picam_kelvin"])
        picam.set_controls(controls)
    except Exception as e:
        logger.error(f"apply_picam_settings: {e}")


_last_hg_params: Optional[dict] = None   # cache for post-capture re-application

def apply_picam_from_hg(params: dict):
    """
    Push HG-computed exposure to picamera2.
    Caches params so _reapply_hg_after_capture() can immediately restore
    manual control after switch_mode_and_capture_file resets AE/AWB to auto.
    """
    global _last_hg_params
    if not _HAS_PICAM or not picam:
        return
    try:
        # Apply preview-adjusted controls (gain-boosted for long-shutter nights)
        # The actual DNG capture uses the full shutter/ISO via switch_mode_and_capture_file
        picam.set_controls(_preview_controls_from_hg(params))
        _last_hg_params = params
    except Exception as e:
        logger.error(f"apply_picam_from_hg: {e}")


# Maximum preview exposure — picamera2 preview stream can't go slower than
# this without dropping to <1fps and becoming useless as a live view.
# When HG requests longer shutters, we boost AnalogueGain instead so the
# preview approximately matches the brightness of the actual captured frames.
_PREVIEW_MAX_SHUTTER_S = 0.25   # 4fps minimum preview rate


def _preview_controls_from_hg(params: dict) -> dict:
    """
    Compute picam controls for the PREVIEW stream when HG is active.

    The preview stream cannot run at shutter speeds > ~0.25s (it would drop
    below 4fps and feel broken as a live view). When HG requests a longer
    exposure (e.g. 1s at night), we cap the preview shutter and boost gain
    proportionally so the live view brightness approximately matches the
    actual DNG captures. This is display-only — the stills always use the
    full HG shutter/ISO.
    """
    shutter_s = params["shutter_s"]
    iso       = params["iso"]

    if shutter_s <= _PREVIEW_MAX_SHUTTER_S:
        # Short enough — preview matches stills exactly
        preview_shutter = shutter_s
        preview_gain    = iso / 100.0
    else:
        # Boost gain to compensate for capped preview shutter
        ratio           = shutter_s / _PREVIEW_MAX_SHUTTER_S
        preview_shutter = _PREVIEW_MAX_SHUTTER_S
        preview_gain    = min((iso / 100.0) * ratio, 64.0)   # cap at ~ISO 6400 equiv

    return {
        "AeEnable":          False,
        "AwbEnable":         False,
        "ExposureTime":      int(preview_shutter * 1_000_000),
        "AnalogueGain":      preview_gain,
        "ColourTemperature": int(params["kelvin"]),
    }


def _reapply_hg_after_capture():
    """
    Re-apply the last HG controls immediately after switch_mode_and_capture_file
    returns.  picamera2 resets AeEnable/AwbEnable to True on every reconfigure,
    so without this the preview runs in full auto for up to one full interval
    before the next apply_picam_from_hg call.  Called inside capture_picam()
    on the thread-pool thread, right after the DNG is saved.
    """
    if not _HAS_PICAM or not picam or not _last_hg_params:
        return
    try:
        picam.set_controls(_preview_controls_from_hg(_last_hg_params))
    except Exception as e:
        logger.warning(f"_reapply_hg_after_capture: {e}")




# ─── HG CALIBRATION SHOT ──────────────────────────────────────────────────────
def _capture_cal_frame(dest_path: str) -> Optional[str]:
    """
    Capture a single calibration DNG to dest_path.
    Uses the current picam HG settings (already applied before this call).
    Returns saved path or None on failure. Does NOT update _latest_shot or
    push to HG tracker — the caller handles that.
    """
    if not _HAS_PICAM or not picam:
        return None
    try:
        still_cfg = picam.create_still_configuration(
            main={"size": (4056, 3040), "format": "RGB888"},
            raw={}
        )
        picam.switch_mode_and_capture_file(still_cfg, dest_path, name="raw")
        if hg.settings.enabled:
            _reapply_hg_after_capture()
        return dest_path if os.path.exists(dest_path) else None
    except Exception as e:
        logger.error(f"Cal frame capture failed: {e}")
        return None


def _load_cal_frame_rgb(path: str) -> Optional[object]:
    """
    Load a DNG cal frame as an RGB numpy array for the HG sky analyser.
    Uses PIL — works for DNG (TIFF) and JPEG. Returns None on failure.
    """
    try:
        from PIL import Image
        import numpy as np
        with Image.open(path) as img:
            rgb = img.convert("RGB")
            # Downsample for speed — sky analyser doesn't need full 12MP
            rgb = rgb.resize((1280, 960), Image.LANCZOS)
            return np.array(rgb)
    except Exception as e:
        logger.debug(f"Cal frame load failed: {e}")
        return None



def hg_calibration_shot() -> Optional[float]:
    """
    Take an AE/AWB-ON exposure to measure ambient EV, then immediately
    lock the camera back to manual control using the computed values.

    KEY FIX: waits for picam2 AE to genuinely converge before reading
    metadata.  3 frames at 30fps = 100ms — not enough for a bright snow
    scene where AE starts slow and needs 1-2 seconds to settle.
    We poll until ExposureTime stabilises (< 5% change over 3 consecutive
    frames) or we time out at 3 seconds.

    Also sanity-checks the result against the astronomical EV prior:
    if the measured value is more than 3 stops away from what the sun
    position predicts, we blend toward the prior to avoid seeding the
    tracker with a bad cal caused by convergence failure.
    """
    global _last_hg_params
    if not _HAS_PICAM or not picam:
        return None
    try:
        picam.set_controls({"AeEnable": True, "AwbEnable": True})

        # ── Wait for AE to converge ───────────────────────────────────────────
        # Poll metadata until ExposureTime is stable for 3 consecutive frames
        # or we've waited 3 seconds.  This handles bright snow / dark scenes
        # where the sensor starts far from the correct exposure.
        prev_exp_us  = None
        stable_count = 0
        deadline     = time.time() + 3.0
        frame        = None
        metadata     = None

        while time.time() < deadline:
            frame    = picam.capture_array()
            metadata = picam.capture_metadata()
            exp_us   = metadata.get("ExposureTime", 0)

            if prev_exp_us is not None and prev_exp_us > 0:
                ratio = abs(exp_us - prev_exp_us) / prev_exp_us
                if ratio < 0.05:   # < 5% change = stable
                    stable_count += 1
                    if stable_count >= 3:
                        break
                else:
                    stable_count = 0
            prev_exp_us = exp_us

        exp_us    = metadata.get("ExposureTime", 1_000_000 // 125)
        gain      = metadata.get("AnalogueGain", 1.0)
        shutter_s = exp_us / 1_000_000
        iso_equiv = int(gain * 100)

        # ── EV from pixels, NOT from aperture math ────────────────────────────
        # anchor_ev MUST be on the same scale as the tracker's pixel-based EV.
        # The tracker computes: ev = log2(lum_linear / 0.18) + 12
        # where lum_linear = (lum_8bit/255)^2.2
        #
        # Using aperture in the EV formula (log2(N²/t) - log2(ISO/100)) gives
        # a "camera EV" that differs from pixel EV by 2*log2(aperture) stops.
        # For unknown apertures (PiCam fixed lens, manual glass) this creates
        # a systematic offset that makes the delta system over- or under-expose
        # by a fixed amount on every single frame.
        #
        # Solution: compute anchor_ev the same way _meter() does — from the
        # actual pixel luminance of the converged AE frame.
        import numpy as _np
        frame_arr = _np.array(frame, dtype=float)
        if frame_arr.ndim == 3:
            lum_arr = (0.2126 * frame_arr[:,:,0]
                     + 0.7152 * frame_arr[:,:,1]
                     + 0.0722 * frame_arr[:,:,2])
        else:
            lum_arr = frame_arr
        # Exclude blown highlights (top 10%) — snow scenes clip easily
        hi = float(_np.percentile(lum_arr, 90))
        lo = float(_np.percentile(lum_arr,  5))
        valid = (lum_arr >= lo) & (lum_arr <= hi) & (lum_arr > 0)
        lum_mean_cal = float(_np.mean(lum_arr[valid])) if _np.any(valid) else 128.0
        lum_linear   = (max(lum_mean_cal, 1.0) / 255.0) ** 2.2
        ev_measured  = math.log2(max(lum_linear, 1e-6) / 0.18) + 12.0

        logger.info(
            f"HG Cal: AE settled SS={shutter_s:.4f}s ISO={iso_equiv} "
            f"lum={lum_mean_cal:.1f} → pixel-EV={ev_measured:.2f} "
            f"(aperture NOT used — pixel scale matches tracker)"
        )

        # ── Kelvin seed for calibration ───────────────────────────────────────
        # At night, pixel-derived Kelvin from the preview frame is polluted by
        # artificial light (streetlamps, sodium, LED) and can't be trusted.
        # Seed directly from the user's configured kelvin_night target instead,
        # so WB starts at the right value from frame 1 rather than slowly
        # crawling down from a wrong 5500K+ reading over many frames.
        try:
            from astral.sun import elevation as _sun_el_cal
            _sun_alt_cal = _sun_el_cal(hg._location.observer,
                                       datetime.datetime.now(hg._tzinfo))
        except Exception:
            _sun_alt_cal = 0.0

        if _sun_alt_cal < 0:
            # Night/twilight — use configured target directly
            kelvin_measured = float(hg._kelvin_for_phase(_sun_alt_cal))
        else:
            # Day/golden — pixel ratios are reliable
            from holygrail import SkyAnalyser, _rg_bg_to_kelvin
            analyser = SkyAnalyser()
            m = analyser.analyse(frame, cam_alt=hg.settings.cam_alt)
            kelvin_measured = (_rg_bg_to_kelvin(m.rg_ratio, m.bg_ratio, m.lum_mean)
                               if m else float(hg.settings.kelvin_day))

        logger.info(
            f"HG Cal shot: SS={shutter_s:.4f}s ISO={iso_equiv} "
            f"pixel-EV={ev_measured:.2f} K={kelvin_measured}"
        )

        hg.seed_from_calibration(ev_measured, kelvin_measured)
        hg.push_capture_frame(frame)

        # ── Set anchor exposure — this is the source of truth ─────────────────
        # From this point _ev_to_exposure works in delta space relative to
        # this frame. Aperture is never used again — unknown lenses work correctly.
        hg.settings.anchor_shutter_s = shutter_s
        hg.settings.anchor_iso       = iso_equiv
        hg.settings.anchor_ev        = ev_measured
        logger.info(
            f"HG anchor set: {shutter_s:.4f}s ISO{iso_equiv} EV{ev_measured:.2f} — "
            f"aperture-free delta mode active."
        )

        # ── Immediately lock camera to manual with calibrated values ──────────
        lock_params = {
            "shutter_s": shutter_s,
            "iso":       iso_equiv,
            "kelvin":    kelvin_measured,
        }
        try:
            picam.set_controls({
                "AeEnable":          False,
                "AwbEnable":         False,
                "ExposureTime":      int(shutter_s * 1_000_000),
                "AnalogueGain":      iso_equiv / 100.0,
                "ColourTemperature": kelvin_measured,
            })
            _last_hg_params = lock_params
            logger.info("HG Cal: camera locked to manual control.")
        except Exception as e:
            logger.warning(f"HG Cal: could not lock controls: {e}")

        return ev_measured

    except Exception as e:
        logger.error(f"hg_calibration_shot: {e}")
        return None


# ─── XMP SIDECAR ──────────────────────────────────────────────────────────────
def _read_aperture_from_exif(path: str) -> Optional[float]:
    """
    Read FNumber from EXIF of a DNG, ARW, or JPEG file.
    DNG and ARW are both TIFF-based — PIL handles them directly.

    Returns f-stop as float (e.g. 2.8, 5.6) or None if not present.
    None means: manual/unknown lens, stay in anchor-delta mode.
    """
    try:
        from PIL import Image
        with Image.open(path) as img:
            exif = img.getexif()
            if not exif:
                return None
            # Tag 33437 = FNumber (preferred — direct f-stop rational)
            fnumber = exif.get(33437)
            if fnumber is not None:
                v = float(fnumber)
                if v > 0:
                    logger.info(f"EXIF FNumber from {os.path.basename(path)}: f/{v}")
                    return v
            # Fallback: tag 37378 = ApertureValue (APEX stops)
            # f-stop = 2^(ApertureValue/2)
            apex = exif.get(37378)
            if apex is not None:
                v = float(apex)
                fstop = round(2 ** (v / 2), 1)
                if fstop > 0:
                    logger.info(
                        f"EXIF ApertureValue {v:.2f} APEX → f/{fstop} "
                        f"from {os.path.basename(path)}"
                    )
                    return fstop
    except Exception as e:
        logger.debug(f"EXIF aperture read failed for {path}: {e}")
    return None



def write_sidecar(dng_path: str, params: dict):
    xmp_path = dng_path.replace(".dng", ".xmp").replace(".ARW", ".xmp")
    world_az = (state.get("origin_az", 0.0) + pan_axis.current_deg) % 360
    xmp = f"""<?xpacket begin='' id='W5M0MpCehiHzreSzNTczkc9d'?>
<x:xmpmeta xmlns:x='adobe:ns:meta/'>
  <rdf:RDF xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'>
    <rdf:Description rdf:about=''
      xmlns:ps='http://ns.pislider.io/1.0/'
      xmlns:crs='http://ns.adobe.com/camera-raw-settings/1.0/'>
      <ps:HG_Mode>{params.get('mode','manual')}</ps:HG_Mode>
      <ps:HG_Phase>{params.get('phase','')}</ps:HG_Phase>
      <ps:HG_EV_Target>{params.get('ev_target',0):.3f}</ps:HG_EV_Target>
      <ps:HG_EV_Final>{params.get('ev_final',0):.3f}</ps:HG_EV_Final>
      <ps:HG_ISO>{params.get('iso',0)}</ps:HG_ISO>
      <ps:HG_Shutter>{params.get('shutter','')}</ps:HG_Shutter>
      <ps:HG_Kelvin>{params.get('kelvin',0)}</ps:HG_Kelvin>
      <ps:Sun_Alt>{params.get('sun_alt',0):.4f}</ps:Sun_Alt>
      <ps:Sun_Az>{params.get('sun_az',0):.4f}</ps:Sun_Az>
      <ps:Moon_Alt>{params.get('moon_alt',0):.4f}</ps:Moon_Alt>
      <ps:Moon_Az>{params.get('moon_az',0):.4f}</ps:Moon_Az>
      <ps:Moon_Phase>{params.get('moon_phase',0):.4f}</ps:Moon_Phase>
      <ps:Rig_Pan_Deg>{pan_axis.current_deg:.3f}</ps:Rig_Pan_Deg>
      <ps:Rig_Tilt_Deg>{tilt_axis.current_deg:.3f}</ps:Rig_Tilt_Deg>
      <ps:Rig_Slider_MM>{slider_axis.current_mm:.3f}</ps:Rig_Slider_MM>
      <ps:Rig_World_Az>{world_az:.3f}</ps:Rig_World_Az>
      <crs:WhiteBalance>Custom</crs:WhiteBalance>
      <crs:Temperature>{params.get('kelvin',5500)}</crs:Temperature>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
<?xpacket end='w'?>"""
    try:
        with open(xmp_path, "w") as f:
            f.write(xmp)
    except Exception as e:
        logger.error(f"XMP: {e}")


# ─── STREAMS ──────────────────────────────────────────────────────────────────
def get_sony_liveview():
    url = f"http://{state['sony_ip']}:8080/liveview/liveviewstream"
    try:
        r = requests.get(url, stream=True, timeout=3)
        for chunk in r.iter_content(chunk_size=1024):
            yield chunk
    except Exception:
        err = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(err, "SONY OFFLINE", (200, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        _, buf = cv2.imencode('.jpg', err)
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n'


def meter_sony_preview_frame() -> None:
    """
    Grab one JPEG from the Sony liveview stream, decode it, and push
    into the HG adaptive tracker.  Called from the sequence loop
    between shots when active_camera == 'sony'.
    """
    if not hg.settings.enabled:
        return
    try:
        url = f"http://{state['sony_ip']}:8080/liveview/liveviewstream"
        r = requests.get(url, stream=True, timeout=3)
        # Read just enough to get one MJPEG frame
        buf = b""
        for chunk in r.iter_content(chunk_size=4096):
            buf += chunk
            # JPEG starts with FF D8, ends with FF D9
            start = buf.find(b'\xff\xd8')
            end   = buf.find(b'\xff\xd9', start + 2) if start >= 0 else -1
            if start >= 0 and end >= 0:
                jpeg_bytes = buf[start:end + 2]
                arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is not None:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    cam_ev = None
                    if _last_hg_params:
                        try:
                            p = _last_hg_params
                            cam_ev = (math.log2((hg.settings.aperture_day ** 2)
                                      / p["shutter_s"])
                                      - math.log2(p["iso"] / 100.0))
                        except Exception:
                            cam_ev = None
                    m = hg.push_preview_frame(frame_rgb, camera_ev=cam_ev)
                    if m:
                        logger.debug(
                            f"Sony preview meter: EV={m.ev:.2f} "
                            f"cond={m.condition}"
                        )
                break
        r.close()
    except Exception as e:
        logger.debug(f"Sony preview meter: {e}")


async def get_picam_liveview():
    global _last_frame
    enc = [cv2.IMWRITE_JPEG_QUALITY, 60]
    _hg_meter_interval = 2.0   # seconds between sky measurements
    _hg_last_meter     = 0.0

    while True:
        if _HAS_PICAM and picam:
            try:
                frame = await asyncio.to_thread(picam.capture_array)
                _last_frame = frame
                active_mode = state.get("active_mode", "timelapse")
                sz = STREAM_SIZE_169 if active_mode == "cinematic" else STREAM_SIZE_43
                small = cv2.resize(frame, sz, interpolation=cv2.INTER_LINEAR)

                # Apply camera orientation transform
                orient = state.get("camera_orientation", "landscape")
                if orient == "portrait_cw":
                    small = cv2.rotate(small, cv2.ROTATE_90_CLOCKWISE)
                elif orient == "portrait_ccw":
                    small = cv2.rotate(small, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif orient == "inverted":
                    small = cv2.rotate(small, cv2.ROTATE_180)

                _, buf = cv2.imencode('.jpg', small, enc)
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n'

                # ── HG sky metering (timelapse mode only, every 2s) ────────────
                now_t = time.time()
                # Skip metering until ISP settles after a still capture.
                # The settle time scales with shutter speed — at night a 4s
                # exposure means the mode-switch takes longer to recover from.
                # Minimum 3s, or 2× the last shutter time, whichever is longer.
                last_shutter = _last_hg_params.get("shutter_s", 0.0) if _last_hg_params else 0.0
                settle_needed = max(3.0, last_shutter * 2.0)
                isp_settled = (now_t - _last_capture_time) > settle_needed
                if (active_mode == "timelapse"
                        and hg.settings.enabled
                        and state.get("is_running", False)
                        and isp_settled
                        and now_t - _hg_last_meter >= _hg_meter_interval):
                    _hg_last_meter = now_t
                    # Fire-and-forget on thread pool — but only if previous
                    # meter task has completed (backpressure guard).
                    # We keep a reference and check done() before submitting.
                    loop = asyncio.get_running_loop()
                    if not hasattr(get_picam_liveview, '_meter_future') or \
                            get_picam_liveview._meter_future.done():
                        # Compute current camera EV so the analyser can
                        # down-weight measurements from blown-out frames.
                        cam_ev = None
                        if _last_hg_params:
                            try:
                                p = _last_hg_params
                                cam_ev = (math.log2((hg.settings.aperture_day ** 2)
                                          / p["shutter_s"])
                                          - math.log2(p["iso"] / 100.0))
                            except Exception:
                                cam_ev = None
                        get_picam_liveview._meter_future = loop.run_in_executor(
                            None, hg.push_preview_frame, frame.copy(), cam_ev
                        )

            except Exception as e:
                logger.error(f"Liveview: {e}")
                await asyncio.sleep(0.5)
        else:
            err = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(err, "PI CAMERA OFFLINE", (140, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
            _, buf = cv2.imencode('.jpg', err, enc)
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n'
        await asyncio.sleep(0.05)   # ~20fps preview


@app.get("/video_feed")
async def video_feed():
    # preview_camera can be overridden independently of capture camera
    preview_cam = state.get("preview_camera", state["active_camera"])
    if preview_cam == "sony":
        return StreamingResponse(get_sony_liveview(),
            media_type="multipart/x-mixed-replace; boundary=frame")
    return StreamingResponse(get_picam_liveview(),
        media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/latest_frame")
async def latest_frame():
    global _latest_shot
    if _latest_shot:
        return StreamingResponse(iter([_latest_shot]), media_type="image/jpeg",
            headers={"Cache-Control": "no-cache, no-store"})
    blank = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.putText(blank, "WAITING FOR FIRST FRAME", (80, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60,60,60), 1)
    _, buf = cv2.imencode('.jpg', blank)
    return StreamingResponse(iter([buf.tobytes()]), media_type="image/jpeg")

@app.get("/browse")
async def browse_dir(path: str = Query(default="/home/tim/Pictures")):
    try:
        p = Path(path).resolve()
        if not p.is_dir():
            return JSONResponse({"error": "Not a directory"}, status_code=400)
        entries = []
        if p.parent != p:
            entries.append({"name": "..", "path": str(p.parent), "type": "dir"})
        for child in sorted(p.iterdir()):
            entries.append({"name": child.name, "path": str(child),
                            "type": "dir" if child.is_dir() else "file"})
        # Include disk usage for this path's filesystem
        usage = shutil.disk_usage(str(p))
        return JSONResponse({"path": str(p), "entries": entries,
                             "disk_free": usage.free, "disk_total": usage.total})
    except PermissionError:
        return JSONResponse({"error": "Permission denied"}, status_code=403)


@app.get("/disk_info")
async def disk_info():
    """Return free/total bytes for the current save path's filesystem."""
    try:
        save = state.get("save_path", "/home/tim/Pictures")
        # Fall back to / if save path doesn't exist yet
        check = save if os.path.exists(save) else "/"
        usage = shutil.disk_usage(check)
        return JSONResponse({"path": save, "free": usage.free, "total": usage.total,
                             "used": usage.used})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/loupe_crop")
async def loupe_crop(cx: float = 0.5, cy: float = 0.5, r: float = 0.15):
    """
    Return a high-quality JPEG crop from the current live preview frame.
    cx, cy are fractions in the *displayed* (possibly rotated) frame.
    We transform them back to raw-frame coordinates before cropping.
    """
    global _last_frame
    frame = _last_frame
    if frame is None or not _HAS_PICAM:
        blank = np.zeros((320, 320, 3), dtype=np.uint8)
        cv2.putText(blank, "NO FEED", (80, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60,60,60), 1)
        _, buf = cv2.imencode('.jpg', blank, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return StreamingResponse(iter([buf.tobytes()]), media_type="image/jpeg",
                                 headers={"Cache-Control": "no-cache, no-store"})

    # Transform (cx, cy) from displayed-frame coords back to raw-frame coords
    orient = state.get("camera_orientation", "landscape")
    if orient == "portrait_cw":
        # display: rotated 90° CW → raw: cx_raw = cy_disp, cy_raw = 1 - cx_disp
        cx, cy = cy, 1.0 - cx
    elif orient == "portrait_ccw":
        # display: rotated 90° CCW → raw: cx_raw = 1 - cy_disp, cy_raw = cx_disp
        cx, cy = 1.0 - cy, cx
    elif orient == "inverted":
        cx, cy = 1.0 - cx, 1.0 - cy

    h, w = frame.shape[:2]
    rp = max(20, int(r * w))
    px = int(cx * w)
    py = int(cy * h)
    x1 = max(0, px - rp);  x2 = min(w, px + rp)
    y1 = max(0, py - rp);  y2 = min(h, py + rp)
    crop = frame[y1:y2, x1:x2]

    # Re-apply the same rotation so the loupe image appears upright
    if orient == "portrait_cw":
        crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
    elif orient == "portrait_ccw":
        crop = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif orient == "inverted":
        crop = cv2.rotate(crop, cv2.ROTATE_180)

    out = cv2.resize(crop, (320, 320), interpolation=cv2.INTER_LANCZOS4)
    _, buf = cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return StreamingResponse(iter([buf.tobytes()]), media_type="image/jpeg",
                             headers={"Cache-Control": "no-cache, no-store"})


# ─── CAPTURE ──────────────────────────────────────────────────────────────────
def capture_sony(frame_id: str) -> Optional[str]:
    dest = os.path.join(state["save_path"], f"FRAME_{frame_id}.ARW")
    try:
        subprocess.run(
            ["gphoto2", "--port", f"ptpip:{state['sony_ip']}",
             "--capture-image-and-download", "--filename", dest],
            check=True, timeout=30)
        return dest
    except Exception as e:
        logger.error(f"Sony capture: {e}")
        return None



def _save_thumb(save_path: str, frame_id: str, frame_rgb: "np.ndarray") -> None:
    """Save a 640×480 JPEG thumbnail for the graph timelapse player."""
    try:
        thumb_dir = os.path.join(save_path, "thumbs")
        os.makedirs(thumb_dir, exist_ok=True)
        thumb = cv2.resize(frame_rgb, (640, 480), interpolation=cv2.INTER_LINEAR)
        thumb_path = os.path.join(thumb_dir, f"THUMB_{frame_id}.jpg")
        cv2.imwrite(thumb_path, thumb, [cv2.IMWRITE_JPEG_QUALITY, 75])
    except Exception as e:
        logger.debug(f"Thumb save failed: {e}")


def _take_meter_shot(frame_index: int) -> Optional[dict]:
    """
    Capture a dedicated metering image at the fixed anchor exposure
    (anchor_shutter_s, anchor_iso). Because the camera settings never change
    between meter shots, every reading is directly comparable — no
    exposure-compensation math required, no feedback oscillation possible.

    The meter JPEG is temporary; only the histogram data is kept.
    Returns the meter result dict from push_meter_shot(), or None on failure.
    """
    if not hg.settings.enabled:
        return None
    if not picam:
        return None
    if (hg.settings.anchor_shutter_s is None or
            hg.settings.anchor_iso is None):
        return None

    try:
        # 1. Switch camera to fixed anchor settings
        anchor_controls = {
            "AeEnable":       False,
            "AwbEnable":      False,
            "ExposureTime":   max(1, int(hg.settings.anchor_shutter_s * 1_000_000)),
            "AnalogueGain":   max(1.0, hg.settings.anchor_iso / 100.0),
            "ColourGains":    (1.0, 1.0),  # neutral WB for meter shot
        }
        picam.set_controls(anchor_controls)

        # Brief settle — one preview frame to let the ISP apply the controls
        import time as _t; _t.sleep(0.25)

        # 2. Capture a JPEG preview frame (not a full DNG — fast, no disk writes)
        frame_array = picam.capture_array()
        rgb = cv2.cvtColor(frame_array, cv2.COLOR_BGR2RGB) if frame_array is not None else None

        # 3. Immediately re-apply HG controls so the creative capture isn't
        #    affected by the meter shot's fixed settings
        if _last_hg_params:
            _reapply_hg_after_capture()

        if rgb is None or rgb.size == 0:
            return None

        # 4. Push histogram to HolyGrail tracker
        from astral.sun import elevation as _se
        sun_alt = _se(hg._location.observer,
                      datetime.datetime.now(hg._tzinfo))

        result = hg.push_meter_shot(rgb, frame_index=frame_index, sun_alt=sun_alt)
        if result:
            logger.info(
                f"Meter shot frame={frame_index}: "
                f"ev={result['meter_ev']:.3f} p50={result['midtone_p50']} "
                f"hl={result['highlight_fraction']:.3f} "
                f"shadow={result['shadow_fraction']:.3f} "
                f"cond={result['condition']} K={result['kelvin']}"
            )
        return result

    except Exception as e:
        logger.warning(f"_take_meter_shot frame={frame_index}: {e}")
        # Always try to restore HG controls even if meter shot failed
        try:
            if _last_hg_params:
                _reapply_hg_after_capture()
        except Exception:
            pass
        return None


def capture_picam(frame_id: str) -> Optional[str]:
    """
    Capture a DNG still. On camera timeout/ISP error, attempts one recovery
    restart before giving up so the sequence can continue.
    """
    global _latest_shot
    dest = os.path.join(state["save_path"], f"FRAME_{frame_id}.dng")
    if not picam:
        return None

    for attempt in range(2):   # try once, recover, try once more
        try:
            still_cfg = picam.create_still_configuration(
                main={"size": (4056, 3040), "format": "RGB888"},
                raw={}
            )
            picam.switch_mode_and_capture_file(still_cfg, dest, name="raw")

            # ── CRITICAL: re-lock HG controls immediately ─────────────────────
            # switch_mode_and_capture_file reconfigures the camera twice
            # (still → preview), resetting AeEnable/AwbEnable to True each time.
            # Re-applying here means the preview is back under manual control
            # within milliseconds rather than waiting up to one full interval.
            if hg.settings.enabled:
                _reapply_hg_after_capture()

            try:
                preview_frame = picam.capture_array()
                _, buf = cv2.imencode('.jpg', preview_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                _latest_shot = buf.tobytes()
                # Save thumbnail for graph timelapse player
                _save_thumb(state["save_path"], frame_id, preview_frame)

                # HG meter shot runs in step 8 (motor move window) instead,
                # so it is guaranteed to fire on every trigger mode.
                # Nothing to do here.
            except Exception:
                pass

            logger.info(f"PiCam DNG saved: {dest}")
            return dest if os.path.exists(dest) else None

        except Exception as e:
            logger.error(f"PiCam capture attempt {attempt+1}: {e}")
            if attempt == 0:
                # First failure — try to recover the camera and retry
                recovered = _restart_picam()
                if not recovered:
                    logger.error("Camera recovery failed — skipping frame.")
                    return None
                logger.info("Camera recovered — retrying capture…")
            # Second failure — give up on this frame, sequence continues
    return None


# ─── MACRO CAPTURE HELPERS ────────────────────────────────────────────────────

async def macro_capture(slot_dir: str, frame_id: str, slot: "ExposureSlot") -> Optional[str]:
    """
    Capture one frame for a macro slot.
    Saves to slot_dir/frame_id.dng (or .ARW for Sony).
    Returns the saved file path or None on failure.
    """
    cam = state.get("active_camera", "picam")
    if cam == "picam":
        dest = os.path.join(slot_dir, f"{frame_id}.dng")
        global _latest_shot
        if not picam:
            return None
        try:
            still_cfg = picam.create_still_configuration(
                main={"size": (4056, 3040), "format": "RGB888"},
                raw={}
            )
            picam.switch_mode_and_capture_file(still_cfg, dest, name="raw")
            try:
                preview_frame = picam.capture_array()
                _, buf = cv2.imencode('.jpg', preview_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                _latest_shot = buf.tobytes()
            except Exception:
                pass
            # Write XMP sidecar with macro position metadata
            _write_macro_sidecar(dest, slot)
            logger.info(f"Macro DNG: {dest}")
            return dest if os.path.exists(dest) else None
        except Exception as e:
            logger.error(f"macro_capture picam: {e}")
            return None
    elif cam == "sony":
        dest = os.path.join(slot_dir, f"{frame_id}.ARW")
        return await asyncio.to_thread(capture_sony_to, dest)
    else:
        # S2 / aux shutter trigger
        await asyncio.to_thread(hw.trigger_camera, 0.2)
        return None


def capture_sony_to(dest: str) -> Optional[str]:
    """Capture Sony ARW to an explicit path (for macro mode)."""
    try:
        subprocess.run(
            ["gphoto2", "--port", f"ptpip:{state['sony_ip']}",
             "--capture-image-and-download", "--filename", dest],
            check=True, timeout=30)
        return dest
    except Exception as e:
        logger.error(f"Sony capture to dest: {e}")
        return None


def _write_macro_sidecar(file_path: str, slot: "ExposureSlot"):
    """XMP sidecar with macro rig position and slot metadata."""
    xmp_path = file_path.replace(".dng", ".xmp").replace(".ARW", ".xmp")
    xmp = f"""<?xpacket begin='' id='W5M0MpCehiHzreSzNTczkc9d'?>
<x:xmpmeta xmlns:x='adobe:ns:meta/'>
  <rdf:RDF xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'>
    <rdf:Description rdf:about=''
      xmlns:ps='http://ns.pislider.io/1.0/'>
      <ps:Macro_Slot>{slot.id}</ps:Macro_Slot>
      <ps:Macro_SlotLabel>{slot.label}</ps:Macro_SlotLabel>
      <ps:Macro_Relay1>{slot.relay1}</ps:Macro_Relay1>
      <ps:Macro_Relay2>{slot.relay2}</ps:Macro_Relay2>
      <ps:Macro_ISO>{slot.iso}</ps:Macro_ISO>
      <ps:Macro_Shutter>{slot.shutter_s:.6f}</ps:Macro_Shutter>
      <ps:Macro_Kelvin>{slot.kelvin}</ps:Macro_Kelvin>
      <ps:Rig_Rail_MM>{slider_axis.current_mm:.4f}</ps:Rig_Rail_MM>
      <ps:Rig_Rotation_Deg>{pan_axis.current_deg:.4f}</ps:Rig_Rotation_Deg>
      <ps:Rig_Aux_Deg>{tilt_axis.current_deg:.4f}</ps:Rig_Aux_Deg>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
<?xpacket end='w'?>"""
    try:
        with open(xmp_path, "w") as f:
            f.write(xmp)
    except Exception as e:
        logger.error(f"Macro XMP: {e}")


async def macro_apply_camera(slot: "ExposureSlot"):
    """Push a slot's camera settings to picam before capture."""
    if not _HAS_PICAM or not picam:
        return
    try:
        controls = {
            "AeEnable":  slot.ae,
            "AwbEnable": slot.awb,
        }
        if not slot.ae:
            controls["ExposureTime"] = int(slot.shutter_s * 1_000_000)
            controls["AnalogueGain"] = slot.iso / 100.0
        if not slot.awb:
            controls["ColourTemperature"] = int(slot.kelvin)
        await asyncio.to_thread(picam.set_controls, controls)
        # Let camera settle for 2 frames at the new settings
        await asyncio.to_thread(picam.capture_array)
        await asyncio.to_thread(picam.capture_array)
    except Exception as e:
        logger.error(f"macro_apply_camera: {e}")
# Frame-differencing approach:
#   1. Convert ROI crop to grayscale, apply Gaussian blur to kill sensor noise
#   2. Absolute difference between current and previous frame
#   3. Threshold the diff image → binary mask of changed pixels
#   4. Morphological close to merge nearby blobs (fills gaps in a car body)
#   5. Find contours, sum area of contours above min_contour_area
#   6. Trigger if total changed area (px²) exceeds user threshold
#   7. Temporal debounce: N consecutive trigger frames required
#
# Why this beats Lucas-Kanade for this use case:
#   - Works in any lighting, any texture (LK needs trackable corners)
#   - No feature seeding needed — instant response to any change
#   - Contour area maps directly to "how much of the ROI changed" — intuitive to tune
#   - Morphological close prevents a car being ignored because its body is smooth

_motion_prev_gray: Optional[np.ndarray] = None
_motion_consec_hits: int = 0
_MOTION_CONSEC_REQUIRED = 2      # consecutive frames required (at ~10fps = 200ms debounce)
_MOTION_MIN_CONTOUR_PX  = 150    # ignore tiny noise blobs below this area (px²)
_MOTION_BLUR_K          = 5      # Gaussian blur kernel — higher = less noise sensitivity


def _check_motion_in_roi(frame: np.ndarray) -> bool:
    """
    Frame-differencing motion detector.

    Returns True when:
    - Total area of changed-pixel blobs in the ROI exceeds the user threshold
    - This persists for _MOTION_CONSEC_REQUIRED consecutive frames

    motion_threshold in state is stored as px² directly (e.g. 2000 = 2000 px² of change).
    A typical car crossing a 50%-wide ROI at 640×480 occupies ~8000–20000 px² depending on
    distance — so default of 2000 is a conservative trigger well above noise.
    """
    global _motion_prev_gray, _motion_consec_hits

    h, w = frame.shape[:2]
    roi  = state["motion_roi"]
    x1, y1 = int(roi[0]*w), int(roi[1]*h)
    x2, y2 = int(roi[2]*w), int(roi[3]*h)
    if x2 <= x1 or y2 <= y1:
        return False

    crop = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop.copy()
    gray = cv2.GaussianBlur(gray, (_MOTION_BLUR_K, _MOTION_BLUR_K), 0)

    if _motion_prev_gray is None or _motion_prev_gray.shape != gray.shape:
        _motion_prev_gray = gray
        _motion_consec_hits = 0
        return False

    # Absolute difference
    diff = cv2.absdiff(_motion_prev_gray, gray)
    _motion_prev_gray = gray

    # Threshold — pixels that changed by more than ~15/255
    _, thresh = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)

    # Morphological close: merge blobs within ~10px of each other
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find contours and sum area of significant ones
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    changed_area = sum(cv2.contourArea(c) for c in contours if cv2.contourArea(c) >= _MOTION_MIN_CONTOUR_PX)

    threshold_px2 = state.get("motion_threshold", 2000)
    triggered = changed_area >= threshold_px2

    if triggered:
        _motion_consec_hits += 1
    else:
        _motion_consec_hits = 0

    return _motion_consec_hits >= _MOTION_CONSEC_REQUIRED


async def motion_detection_loop():
    """
    Background task: continuously check _last_frame for motion using frame differencing.
    Runs at ~10fps — slightly slower than LK but gives better sensitivity for
    distant/slow-moving subjects and works regardless of scene texture.
    """
    global _motion_triggered, _motion_prev_gray, _motion_consec_hits
    _motion_prev_gray  = None
    _motion_consec_hits = 0
    warmup = state.get("motion_warmup_frames", 10)
    count  = 0
    logger.info("Motion detection loop started (frame differencing).")
    while state["is_running"]:
        tmode = state["trigger_mode"]
        if not tmode.startswith("picam_motion"):
            await asyncio.sleep(0.1)
            continue
        frame = _last_frame
        if frame is not None:
            count += 1
            if count > warmup and not _motion_triggered:
                if _check_motion_in_roi(frame):
                    _motion_triggered = True
                    logger.info("MOTION: frame-diff trigger fired.")
            else:
                _check_motion_in_roi(frame)   # keep prev_gray current during warmup
        await asyncio.sleep(0.1)   # ~10fps — good balance of sensitivity vs CPU
    logger.info("Motion detection loop ended.")


# ─── SOFT LIMITS ──────────────────────────────────────────────────────────────
def clamp_pan(v):  return max(state["pan_min"],  min(state["pan_max"],  v))
def clamp_tilt(v): return max(state["tilt_min"], min(state["tilt_max"], v))


# ─── SEQUENCE PROGRESS ESTIMATES ──────────────────────────────────────────────
def _estimate_progress(current_frame: int, current_interval: float) -> dict:
    """
    Return estimated total_frames (if time-based) or estimated end-time
    (if frame-count based), plus current interval.
    """
    remaining = state["total_frames"] - current_frame
    secs_left = remaining * current_interval
    est_end   = datetime.datetime.now() + datetime.timedelta(seconds=secs_left)
    return {
        "current_interval": round(current_interval, 1),
        "estimated_end":    est_end.strftime("%H:%M:%S"),
        "estimated_end_ts": est_end.isoformat(),
        "secs_remaining":   int(secs_left),
    }


# ─── TIMELAPSE WORKER ─────────────────────────────────────────────────────────
async def timelapse_worker(base_interval: float):
    global _latest_shot, _motion_triggered
    try:
        await _timelapse_worker_inner(base_interval)
    except asyncio.CancelledError:
        logger.info("Timelapse worker cancelled.")
    except Exception as e:
        logger.error(f"Timelapse worker crashed: {e}", exc_info=True)
        state["is_running"] = False
        state["stop_event"].clear()
        status_leds.set_error()
        await broadcast({"type": "run_state", "running": False})
        await broadcast({"type": "log",
            "msg": f"⛔ Sequence crashed at frame {state['current_frame']}: {e}"})


async def _timelapse_worker_inner(base_interval: float):
    global _latest_shot, _motion_triggered

    state["is_running"]    = True
    state["current_frame"] = 0
    os.makedirs(state["save_path"], exist_ok=True)

    # ── Scheduled start: wait until start time if set ─────────────────────────
    schedule_start = state.get("schedule_start")
    logger.info(f"SCHEDULE_DEBUG: schedule_start={schedule_start!r}")
    await broadcast({"type": "log", "msg": f"SCHEDULE_DEBUG: schedule_start={schedule_start!r}"})
    if schedule_start:
        try:
            from zoneinfo import ZoneInfo as _ZoneInfo
            # Parse the ISO string — JS sends UTC ISO from new Date().toISOString()
            start_dt = datetime.datetime.fromisoformat(
                schedule_start.replace("Z", "+00:00"))
            now_dt   = datetime.datetime.now(datetime.timezone.utc)
            wait_s   = (start_dt - now_dt).total_seconds()
            logger.info(f"SCHEDULE_DEBUG: start_dt={start_dt} now={now_dt} wait_s={wait_s:.0f}")
            await broadcast({"type": "log",
                "msg": f"SCHEDULE_DEBUG: target={start_dt.strftime('%H:%M:%S %Z')} "
                       f"now={now_dt.strftime('%H:%M:%S %Z')} wait={wait_s:.0f}s"})
            if wait_s > 0:
                await broadcast({"type": "run_state", "running": True})
                await broadcast({"type": "log",
                    "msg": f"⏳ Scheduled start: waiting {wait_s/60:.1f} min "
                           f"until {start_dt.astimezone(_ZoneInfo(state.get('timezone','America/Winnipeg'))).strftime('%H:%M:%S %Z')}…"})
                while wait_s > 0 and not state["stop_event"].is_set():
                    await asyncio.sleep(min(5.0, wait_s))
                    # Recompute from wall clock — avoids drift in long waits
                    wait_s = (start_dt - datetime.datetime.now(datetime.timezone.utc)).total_seconds()
                    if wait_s > 0 and int(wait_s) % 300 < 6:
                        await broadcast({"type": "log",
                            "msg": f"⏳ Starting in {wait_s/60:.1f} min…"})
                if state["stop_event"].is_set():
                    state["is_running"] = False
                    await broadcast({"type": "run_state", "running": False})
                    await broadcast({"type": "log", "msg": "Scheduled start cancelled."})
                    return
                await broadcast({"type": "log",
                    "msg": "✓ Scheduled start time reached — beginning sequence."})
            else:
                await broadcast({"type": "log",
                    "msg": f"⚠ Scheduled start time already passed by {-wait_s/60:.1f} min — starting now."})
        except Exception as e:
            logger.warning(f"schedule_start error ({e}) — starting immediately.")
            await broadcast({"type": "log", "msg": f"⚠ Schedule parse error: {e} — starting immediately."})

    await broadcast({"type": "run_state", "running": True})

    # Reset HG anchor so each sequence gets a fresh calibration
    hg.settings.anchor_shutter_s = None
    hg.settings.anchor_iso       = None
    hg.settings.anchor_ev        = None

    # Clear graph history — new sequence = new graph
    _session_history.clear()
    try:
        _SESSION_HISTORY_FILE.write_text('{"frames":[]}')
    except Exception:
        pass
    await broadcast({"type": "graph_reset"})

    # ── Disk space pre-flight ──────────────────────────────────────────────────
    try:
        usage = shutil.disk_usage(state["save_path"])
        # IMX477 DNG ≈ 25 MB, ARW ≈ 30 MB, JPEG ≈ 3 MB — estimate conservatively
        cam  = state.get("active_camera", "picam")
        mb_per_frame = 30.0 if cam == "sony" else 25.0
        needed_mb  = state["total_frames"] * mb_per_frame
        free_mb    = usage.free / (1024 * 1024)
        if free_mb < needed_mb:
            msg = (f"⚠ DISK WARNING: ~{needed_mb:.0f} MB needed, "
                   f"only {free_mb:.0f} MB free. Sequence may fail before completion.")
            await broadcast({"type": "log", "msg": msg})
            logger.warning(msg)
        if free_mb < mb_per_frame * 2:
            # Less than 2 frames of space — abort
            err = f"⛔ DISK FULL: only {free_mb:.0f} MB free. Sequence aborted."
            await broadcast({"type": "disk_full", "msg": err})
            logger.error(err)
            state["is_running"] = False
            await broadcast({"type": "run_state", "running": False})
            return
    except Exception as e:
        logger.warning(f"Disk pre-flight check failed: {e}")

    # ── HG Calibration Phase ─────────────────────────────────────────────────
    # The calibration phase runs BEFORE the sequence loop and frame counter.
    # Motion does NOT start until calibration is complete.
    # Files are saved as CAL_0000.dng / CAL_0001.dng — separate from the
    # sequence FRAME_xxxx files — so they never appear in the edit timeline.
    #
    # Step 1: AE-settle shot (preview only, no file saved)
    #         → seeds EV tracker from camera metadata
    # Step 2: CAL_0000 — first real captured frame
    #         → reads EXIF aperture (if available) and refines anchor_ev
    #         → pushes full-res frame into HG sky analyser
    # Step 3: CAL_0001 — second captured frame at the now-correct exposure
    #         → further warms the tracker so frame 1 of the sequence is stable
    #
    if hg.settings.enabled:
        await broadcast({"type": "log", "msg": "HG: starting calibration…"})
        logger.info("HG: starting calibration phase (2 cal frames before sequence).")

        # Step 1 — AE settle + anchor
        ev = await asyncio.to_thread(hg_calibration_shot)
        await broadcast({"type": "log",
            "msg": f"HG cal: AE settled EV≈{ev:.2f}" if ev
                   else "HG cal: camera unavailable, starting cold."})

        # Steps 2 & 3 — two real captured frames, named CAL_xxxx
        for cal_idx in range(2):
            if state["stop_event"].is_set():
                break

            cal_params  = hg.get_next_shot_parameters()
            cal_phase   = cal_params.get("phase", "day")
            cal_shutter = cal_params.get("shutter_s", 1/125)

            # Apply HG exposure to camera
            if _HAS_PICAM and picam:
                await asyncio.to_thread(apply_picam_from_hg, cal_params)

            # Capture to CAL file — not counted, not moved by rig
            cal_dest = os.path.join(
                state["save_path"], f"CAL_{cal_idx:04d}.dng"
            )
            status_leds.set_sequence_phase("shutter", cal_shutter)
            cal_path = await asyncio.to_thread(
                _capture_cal_frame, cal_dest
            )
            status_leds.set_sequence_phase("waiting")

            if cal_path:
                # Load cal frame RGB first — needed for anchor re-computation
                cal_frame = await asyncio.to_thread(_load_cal_frame_rgb, cal_path)

                # Cal frame 0: re-anchor from actual capture pixels
                # The AE calibration shot is metered at whatever shutter AE
                # chose (may be very short at night). CAL_0000 is captured at
                # the real HG exposure (e.g. 1s). Re-computing anchor_ev from
                # its pixels gives us the correct reference for the sequence.
                if cal_idx == 0 and cal_frame is not None:
                    import numpy as _np
                    lum_arr = (0.2126 * cal_frame[:,:,0].astype(float)
                             + 0.7152 * cal_frame[:,:,1].astype(float)
                             + 0.0722 * cal_frame[:,:,2].astype(float))
                    hi = float(_np.percentile(lum_arr, 90))
                    lo = float(_np.percentile(lum_arr,  5))
                    valid = (lum_arr >= lo) & (lum_arr <= hi) & (lum_arr > 0)
                    lum_cal = float(_np.mean(lum_arr[valid])) if _np.any(valid) else 128.0
                    lum_lin = (max(lum_cal, 1.0) / 255.0) ** 2.2
                    cal_ev  = math.log2(max(lum_lin, 1e-6) / 0.18) + 12.0
                    old_anchor = hg.settings.anchor_ev
                    hg.settings.anchor_ev      = cal_ev
                    hg.settings.anchor_shutter_s = cal_params.get("shutter_s", hg.settings.anchor_shutter_s)
                    hg.settings.anchor_iso       = cal_params.get("iso", hg.settings.anchor_iso)
                    msg = (f"HG anchor re-calibrated from CAL_0000 pixels: "
                           f"EV {old_anchor:.2f}→{cal_ev:.2f} "
                           f"SS={hg.settings.anchor_shutter_s:.3f}s "
                           f"ISO={hg.settings.anchor_iso}")
                    logger.info(msg)
                    await broadcast({"type": "log", "msg": f"📷 {msg}"})

                    # Also check EXIF for aperture refinement (Sony only)
                    aperture_from_exif = await asyncio.to_thread(
                        _read_aperture_from_exif, cal_path
                    )
                    if aperture_from_exif is not None:
                        hg.settings.aperture_day   = aperture_from_exif
                        hg.settings.aperture_night = aperture_from_exif
                        await broadcast({"type": "log",
                            "msg": f"HG EXIF aperture: f/{aperture_from_exif}"})

                # Push cal frame into HG tracker (cal_frame already loaded above)
                if cal_frame is not None:
                    await asyncio.to_thread(hg.push_capture_frame, cal_frame)

                await broadcast({"type": "log",
                    "msg": f"HG cal frame {cal_idx + 1}/2 captured."})
            else:
                await broadcast({"type": "log",
                    "msg": f"HG cal frame {cal_idx + 1}/2 failed — continuing."})

            # Brief settle between cal frames
            await asyncio.sleep(0.5)

        await broadcast({"type": "log", "msg": "✓ HG calibration complete — starting sequence."})
        logger.info("HG calibration phase complete.")

    # ── Start motion detection task — AFTER calibration ──────────────────────
    motion_task = None
    if state["trigger_mode"].startswith("picam_motion"):
        _motion_triggered = False
        motion_task = asyncio.create_task(motion_detection_loop())

    save_session()
    logger.info(f"Timelapse: {state['total_frames']} frames, mode={state['trigger_mode']}")

    # ── LED: sequence started ──────────────────────────────────────────────────
    active_mode = state.get("active_mode", "timelapse")
    status_leds.set_mode("timelapse_run" if active_mode == "timelapse" else "macro_run")
    status_leds.set_sequence_phase("waiting")

    for i in range(state["total_frames"]):
        if state["stop_event"].is_set():
            break

        loop_start = asyncio.get_event_loop().time()

        # 1. HG parameters
        params   = hg.get_next_shot_parameters()
        interval = params.get("interval", base_interval)

        # Update HG phase LED
        status_leds.set_hg_phase(params.get("phase", "unknown"))

        # Update progress LED
        pct = (i / max(state["total_frames"] - 1, 1))
        status_leds.set_progress(pct)

        # 2. Apply exposure
        if hg.settings.enabled:
            await asyncio.to_thread(apply_picam_from_hg, params)

        # 3. Trigger gating — LED: waiting
        status_leds.set_sequence_phase("waiting")
        state["aux_triggered"] = False
        _motion_triggered      = False
        tmode = state["trigger_mode"]

        def _triggered():
            """True if either aux GPIO or motion detection fired."""
            return state["aux_triggered"] or _motion_triggered

        if tmode in ("aux_only", "picam_motion_only"):
            while not _triggered() and not state["stop_event"].is_set():
                await asyncio.sleep(0.05)
            if state["stop_event"].is_set():
                break

        elif tmode in ("aux_hybrid", "picam_motion_hybrid"):
            deadline = loop_start + interval
            while (not _triggered()
                   and asyncio.get_event_loop().time() < deadline
                   and not state["stop_event"].is_set()):
                await asyncio.sleep(0.05)
            if state["stop_event"].is_set():
                break

        # 4. Shoot — LED: shutter
        shutter_s = params.get("shutter_s", 1/125)
        status_leds.set_sequence_phase("shutter", shutter_s)
        state["current_frame"] = i + 1
        file_path  = None
        capture_blocked = False
        try:
            cam = state["active_camera"]
            if cam == "picam":
                file_path = await asyncio.to_thread(capture_picam, f"{i:04d}")
                capture_blocked = True
                _last_capture_time = time.time()   # ISP settle guard
            elif cam == "sony":
                file_path = await asyncio.to_thread(capture_sony,  f"{i:04d}")
            elif cam == "s2":
                await asyncio.to_thread(hw.trigger_camera, 0.2)
        except OSError as e:
            if e.errno == 28:  # ENOSPC
                err = f"⛔ DISK FULL at frame {i+1}: sequence halted. Free space and restart."
                logger.error(err)
                await broadcast({"type": "disk_full", "msg": err})
                break
            logger.error(f"Shot {i}: {e}")
        except Exception as e:
            logger.error(f"Shot {i}: {e}")

        # 5. Wait for exposure + margin (skip for PiCam — blocking capture already included it)
        if not capture_blocked:
            await asyncio.sleep(max(0.05, shutter_s + state["exp_margin"]))

        # 6. Sidecar metadata + post-capture HG tracker push + LED save status
        if file_path:
            await asyncio.to_thread(write_sidecar, file_path, params)
            await broadcast({"type": "shutter_event"})
            status_leds.set_save_ok()

            # NOTE: We do NOT push _last_frame here. The first preview frame
            # after a still capture is unreliable — the ISP hasn't settled back
            # to the manual exposure level yet. Pushing it causes a sawtooth
            # pattern (bad reading → overcorrect → correct back → repeat).
            # The preview loop pushes a correct metering frame every 2s instead.
        else:
            # File not saved — either skipped frame or actual error
            if state["current_frame"] > 1:   # don't flag before first real attempt
                status_leds.set_save_error()

        # Broadcast disc-entry warnings if sun/moon about to enter frame
        if hg.settings.enabled:
            disc = params.get("disc_entry", {})
            for body, info in disc.items():
                mins = info.get("minutes", 99)
                if mins <= 5:
                    await broadcast({"type": "log",
                        "msg": f"⚠ {body.upper()} enters frame in ~{mins:.1f} min "
                               f"(az={info['az']:.0f}° alt={info['alt']:.1f}°)"})

        # 7. Progress + telemetry broadcast
        tracker_status = hg.get_tracker_status() if hg.settings.enabled else {}
        progress = _estimate_progress(state["current_frame"], interval)
        frame_data = {
            "type":       "status",
            "frame":      state["current_frame"],
            "frame_id":   f"{i:04d}",   # actual filename ID used for THUMB_{frame_id}.jpg
            "has_thumb":  True,
            "total":      state["total_frames"],
            "hg_phase":   params.get("phase", "—"),
            "hg_sun_alt": float(params.get("sun_alt", 0.0)),
            "hg_ev":      float(params.get("ev_final",   params.get("ev_target", 0.0))),
            "hg_ev_scene": float(params.get("ev_blended", params.get("ev_target", 0.0))),
            "hg_iso":     params.get("iso", 0),
            "hg_shutter": params.get("shutter", "—"),
            "hg_shutter_s": float(params.get("shutter_s", 1/125)),
            "hg_kelvin":  params.get("kelvin", 0),
            "hg_condition":    tracker_status.get("condition", ""),
            "hg_ev_slope":     tracker_status.get("ev_slope", 0.0),
            "hg_confidence":   tracker_status.get("confidence", 0.0),
            "hg_tracker_warm": tracker_status.get("warm", False),
            **progress,
        }
        _session_history.append(frame_data)
        await broadcast(frame_data)

        # Persist frame count so reconnect shows correct progress
        state["current_frame"] = i + 1
        if (i + 1) % 10 == 0:
            save_session()
            _save_session_history()

        # 8. Move rig + meter shot (concurrent) — LED: motors
        #
        # The motor movement window is the ONLY time the camera is
        # guaranteed free regardless of trigger mode:
        #   - normal: camera is between shots
        #   - aux_only / picam_motion_only: trigger hasn't fired yet for
        #     the *next* frame, but current shot is done — motors are moving
        #   - aux_hybrid / picam_motion_hybrid: same as above
        #   - no motion: falls through to step 9 vibration settle window
        #
        # The meter shot runs concurrently with the motors on a thread-pool
        # thread. It takes ~300ms (ISP settle + frame grab) and the motor
        # move takes 1.5s, so it always completes before motors stop.
        _meter_future = None
        _has_motor_move = False

        if hg.settings.enabled and state.get("active_camera") == "picam":
            frame_idx_for_meter = i
            _meter_future = asyncio.get_event_loop().run_in_executor(
                None, lambda fi=frame_idx_for_meter: _take_meter_shot(fi)
            )

        if len(engine.keyframes) >= 2:
            try:
                if not hasattr(timelapse_worker, '_traj_cache') or \
                   timelapse_worker._traj_cache.get('n_kf') != len(engine.keyframes) or \
                   timelapse_worker._traj_cache.get('total') != state["total_frames"]:
                    traj_s, traj_p, traj_t = engine.generate_trajectory(
                        duration_s=float(state["total_frames"]),
                        fps=1,
                        easing_curve="linear"
                    )
                    timelapse_worker._traj_cache = {
                        'n_kf': len(engine.keyframes),
                        'total': state["total_frames"],
                        'traj_s': traj_s, 'traj_p': traj_p, 'traj_t': traj_t,
                    }

                idx = min(i + 1, state["total_frames"] - 1)
                cache = timelapse_worker._traj_cache
                target_s = float(cache['traj_s'][idx])
                target_p = clamp_pan(float(cache['traj_p'][idx]))
                target_t = clamp_tilt(float(cache['traj_t'][idx]))

                ds = int((target_s - slider_axis.current_mm) * slider_axis.steps_per_mm)
                dp = int((target_p - pan_axis.current_deg)   * pan_axis.steps_per_deg)
                dt = int((target_t - tilt_axis.current_deg)  * tilt_axis.steps_per_deg)
                if any([ds, dp, dt]):
                    _has_motor_move = True
                    status_leds.set_sequence_phase("motors")
                    hw.enable_motors(True)
                    await asyncio.to_thread(hw.move_axes_simultaneous, ds, dp, dt, 1.5)
                    slider_axis.current_mm = target_s
                    pan_axis.current_deg   = target_p
                    tilt_axis.current_deg  = target_t
                    hw.enable_motors(False)
                    status_leds.set_sequence_phase("waiting")
            except Exception as e:
                logger.error(f"Motion step {i}: {e}")

        # Await the meter shot future if it was launched but motors didn't move
        # (or just let it finish in the background if motors ran concurrently).
        # Either way we collect the result here so any exception is surfaced.
        if _meter_future is not None:
            try:
                await _meter_future
            except Exception as e:
                logger.warning(f"Meter shot future frame {i}: {e}")

        # 9. Anti-vibration settle — LED: waiting
        # If there was no motor move and no meter shot yet (aux_only /
        # motion_only with no keyframes), use the settle window for the
        # meter shot now. It's still guaranteed dead time.
        status_leds.set_sequence_phase("waiting")
        vibe_delay = state["vibe_delay"]

        if (not _has_motor_move
                and _meter_future is None
                and hg.settings.enabled
                and state.get("active_camera") == "picam"):
            # Run meter shot during settle — allow at least 0.5s settle after
            meter_start = asyncio.get_event_loop().time()
            frame_idx_for_meter = i
            await asyncio.get_event_loop().run_in_executor(
                None, lambda fi=frame_idx_for_meter: _take_meter_shot(fi)
            )
            meter_elapsed = asyncio.get_event_loop().time() - meter_start
            # Sleep any remaining settle time after the meter shot finishes
            remaining_settle = vibe_delay - meter_elapsed
            if remaining_settle > 0:
                await asyncio.sleep(remaining_settle)
        else:
            await asyncio.sleep(vibe_delay)

        # 10. Wait interval remainder (normal / hybrid modes)
        # Sony: grab a liveview frame during the wait for HG sky metering
        if tmode in ("normal", "aux_hybrid", "picam_motion_hybrid"):
            elapsed   = asyncio.get_event_loop().time() - loop_start
            remainder = interval - elapsed

            # Sony inter-shot sky metering (runs in background during wait)
            if (hg.settings.enabled
                    and state.get("active_camera") == "sony"
                    and remainder > 2.0):
                await asyncio.sleep(1.0)   # let camera settle after shot
                await asyncio.to_thread(meter_sony_preview_frame)
                elapsed   = asyncio.get_event_loop().time() - loop_start
                remainder = interval - elapsed

            if remainder > 0:
                await asyncio.sleep(remainder)
            elif remainder < -0.5:
                await broadcast({"type": "log",
                    "msg": f"⚠ Interval overrun at frame {i+1}: "
                           f"loop took {elapsed:.1f}s vs {interval:.1f}s target "
                           f"({abs(remainder):.1f}s over). Motor/shutter time exceeded interval."})

    # Cleanup
    if motion_task:
        motion_task.cancel()
    state["is_running"] = False
    state["stop_event"].clear()
    save_session()
    await broadcast({"type": "run_state", "running": False})
    logger.info("Timelapse complete.")
    # Return LEDs to idle breathing
    active_mode = state.get("active_mode", "timelapse")
    status_leds.set_mode("timelapse_idle" if active_mode != "macro" else "macro_idle")


# ─── BROADCAST ────────────────────────────────────────────────────────────────
# Only one active WebSocket client is allowed at a time.
# The active client is tracked here; broadcast always targets just this one.
connected_clients: set = set()   # kept for legacy broadcast() signature
_active_ws: Optional[WebSocket] = None
_graph_clients: set = set()       # read-only graph tabs — never kicked, never control

async def broadcast(payload: dict):
    """Send to the active control client and any read-only graph clients."""
    global _active_ws
    if _active_ws is not None:
        try:
            await _active_ws.send_json(payload)
        except Exception:
            _active_ws = None
    # Also forward to all graph tabs (read-only, never kicked)
    dead = set()
    for gc in _graph_clients:
        try:
            await gc.send_json(payload)
        except Exception:
            dead.add(gc)
    _graph_clients.difference_update(dead)


# ─── INIT PACKET ──────────────────────────────────────────────────────────────
def _build_init_packet() -> dict:
    """Full state packet sent to every new WS client on connect."""
    import dataclasses
    hg_d = {k: v for k, v in dataclasses.asdict(hg.settings).items()
            if not isinstance(v, datetime.datetime)}
    return {
        "type":            "init",
        "running":         state["is_running"],
        "interrupted":     state.get("_was_interrupted", False),
        "current_frame":   state["current_frame"],
        "total_frames":    state["total_frames"],
        "trigger_mode":    state["trigger_mode"],
        "pan_min":         state["pan_min"],
        "pan_max":         state["pan_max"],
        "tilt_min":        state["tilt_min"],
        "tilt_max":        state["tilt_max"],
        "save_path":       state["save_path"],
        "pan_deg":         pan_axis.current_deg,
        "tilt_deg":        tilt_axis.current_deg,
        "slider_mm":       slider_axis.current_mm,
        "hg_settings":     hg_d,
        "motion_roi":      state["motion_roi"],
        "motion_threshold": state["motion_threshold"],
        "active_camera":   state["active_camera"],
        "preview_camera":  state.get("preview_camera", state["active_camera"]),
        "vibe_delay":      state["vibe_delay"],
        "exp_margin":      state["exp_margin"],
        "manual_interval": state.get("manual_interval", 5.0),
        "active_mode":     state.get("active_mode", "timelapse"),
        "camera_orientation": state.get("camera_orientation", "landscape"),
        "cine_fps":           state.get("cine_fps", 24),
    }


# ─── WEBSOCKET ────────────────────────────────────────────────────────────────
@app.get("/api/gps")
async def api_gps():
    """
    Server-side location lookup — runs on the Pi, not the browser.
    Avoids Chrome's HTTPS-only restriction on navigator.geolocation.
    Uses ip-api.com for approximate location based on the Pi's public IP.
    Falls back to the currently configured HG lat/lon if unavailable.
    """
    try:
        import urllib.request as _ur
        import json as _json
        with _ur.urlopen("http://ip-api.com/json/?fields=lat,lon,city,timezone", timeout=4) as r:
            data = _json.loads(r.read())
        lat = round(float(data["lat"]), 5)
        lon = round(float(data["lon"]), 5)
        tz  = data.get("timezone", "America/Winnipeg")
        city = data.get("city", "")
        return JSONResponse({"lat": lat, "lon": lon, "timezone": tz, "city": city})
    except Exception as e:
        # Return currently configured values so UI doesn't error
        return JSONResponse({
            "lat": hg.settings.latitude,
            "lon": hg.settings.longitude,
            "timezone": state.get("timezone", "America/Winnipeg"),
            "city": "",
            "error": str(e),
        })


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global _active_ws, _macro_task, _cinematic_mode, _prog_task, _recording, _record_start_time
    global _active_ws

    await websocket.accept()

    # ── Single-instance enforcement ───────────────────────────────────────────
    # If another tab/window is already connected, kick it with a clear message
    # then take over as the new active client.
    if _active_ws is not None:
        try:
            await _active_ws.send_json({
                "type": "kicked",
                "msg":  "⚠ A new browser window connected — this tab has been replaced. "
                        "Close this tab; the new one is now in control."
            })
            await _active_ws.close(code=4001, reason="replaced_by_new_client")
        except Exception:
            pass   # old socket may already be dead
        logger.info("WS: previous client kicked — new client taking over.")

    _active_ws = websocket
    connected_clients.discard(websocket)   # not used for routing, kept for compatibility
    connected_clients.add(websocket)

    await websocket.send_json(_build_init_packet())

    try:
        while True:
            data = await websocket.receive_text()
            msg  = json.loads(data)
            cmd  = msg.get("command")

            # ── JOYSTICK ──────────────────────────────────────────────────────
            if cmd == "joystick":
                if state["is_running"]: continue
                vx = float(msg.get("vx", 0))
                vy = float(msg.get("vy", 0))
                vz = float(msg.get("vz", 0))   # slider axis from horizontal strip
                if vx == 0 and vy == 0 and vz == 0:
                    hw.set_tmc_velocity(0, 0); hw.set_tmc_velocity(1, 0)
                    hw.set_tmc_velocity(2, 0); hw.enable_motors(False)
                    status_leds.set_motor_speeds(0, 0, 0)
                else:
                    pan_ok  = (vx>0 and pan_axis.current_deg  < state["pan_max"]) or \
                              (vx<0 and pan_axis.current_deg  > state["pan_min"])
                    tilt_ok = (vy>0 and tilt_axis.current_deg < state["tilt_max"]) or \
                              (vy<0 and tilt_axis.current_deg > state["tilt_min"])
                    hw.enable_motors(True)
                    s_vel = int(vz * 28000)
                    p_vel = int(vx * 18000) if pan_ok  else 0
                    t_vel = int(vy * 12000) if tilt_ok else 0
                    hw.set_tmc_velocity(0, s_vel)
                    hw.set_tmc_velocity(1, p_vel)
                    hw.set_tmc_velocity(2, t_vel)
                    status_leds.set_motor_speeds(slider=s_vel, pan=p_vel, tilt=t_vel)

            # ── NUDGE ─────────────────────────────────────────────────────────
            elif cmd == "nudge_axis":
                if state["is_running"]: continue
                axis = msg.get("axis", "pan")
                deg  = float(msg.get("deg", 0))
                hw.enable_motors(True)
                if axis == "pan":
                    new = clamp_pan(pan_axis.current_deg + deg)
                    d   = new - pan_axis.current_deg
                    if abs(d) > 0.01:
                        hw.move_axes_simultaneous(0, int(d*pan_axis.steps_per_deg), 0, abs(d)*0.1+0.1)
                        pan_axis.current_deg = new
                elif axis == "tilt":
                    new = clamp_tilt(tilt_axis.current_deg + deg)
                    d   = new - tilt_axis.current_deg
                    if abs(d) > 0.01:
                        hw.move_axes_simultaneous(0, 0, int(d*tilt_axis.steps_per_deg), abs(d)*0.1+0.1)
                        tilt_axis.current_deg = new
                hw.enable_motors(False)
                save_session()
                await websocket.send_json({"type": "status",
                    "pos_p": pan_axis.current_deg, "pos_t": tilt_axis.current_deg})

            # ── SOFT LIMITS ───────────────────────────────────────────────────
            elif cmd == "set_limits":
                axis  = msg.get("axis", "pan")
                which = msg.get("which", "max")
                val   = msg.get("value", None)
                if val is None:
                    val = pan_axis.current_deg if axis == "pan" else tilt_axis.current_deg
                state[f"{axis}_{which}"] = float(val)
                save_session()
                await websocket.send_json({"type": "limits_updated",
                    "pan_min": state["pan_min"], "pan_max": state["pan_max"],
                    "tilt_min": state["tilt_min"], "tilt_max": state["tilt_max"]})

            # ── CALIBRATION ───────────────────────────────────────────────────
            elif cmd == "calibrate_origin":
                bearing = float(msg.get("bearing_deg", 0))
                state["origin_az"]   = (bearing - pan_axis.current_deg) % 360
                hg.settings.cam_az   = bearing
                hg.settings.cam_alt  = -tilt_axis.current_deg
                save_session()
                await websocket.send_json({"type": "calibration_done",
                    "origin_az": state["origin_az"],
                    "cam_az": bearing, "cam_alt": -tilt_axis.current_deg})

            # ── TRIGGERS ──────────────────────────────────────────────────────
            elif cmd == "aux_trigger":
                state["aux_triggered"] = True

            elif cmd == "set_trigger_mode":
                mode = msg.get("mode", "normal")
                valid = ("normal","aux_only","aux_hybrid",
                         "picam_motion_only","picam_motion_hybrid")
                if mode in valid:
                    state["trigger_mode"] = mode
                    save_session()

            elif cmd == "set_motion_roi":
                state["motion_roi"]       = msg.get("roi",       state["motion_roi"])
                state["motion_threshold"] = msg.get("threshold", state["motion_threshold"])
                save_session()

            # ── CAMERA ────────────────────────────────────────────────────────
            elif cmd == "set_camera":
                state["active_camera"] = msg.get("value", "picam")
                save_session()

            elif cmd == "set_mode":
                mode = msg.get("value", "timelapse")
                state["active_mode"] = mode
                # Switch picam preview aspect ratio
                if _HAS_PICAM and picam and not state["is_running"]:
                    try:
                        cfg = PREVIEW_CONFIG_169 if mode == "cinematic" else PREVIEW_CONFIG_43
                        picam.stop()
                        picam.configure(picam.create_video_configuration(main=cfg))
                        picam.start()
                        logger.info(f"Preview config switched for mode: {mode}")
                    except Exception as e:
                        logger.error(f"Preview config switch: {e}")
                # Apply per-mode microstepping
                try:
                    hw.set_mode_microstepping(mode)
                except Exception as e:
                    logger.warning(f"Microstepping set failed: {e}")
                # Auto-fan: cinematic = 60% (motors run continuously), others = 20%
                auto_fan = {"cinematic": 60, "timelapse": 20, "macro": 20}.get(mode, 20)
                mstep    = {"macro": 256, "timelapse": 16, "cinematic": 8}.get(mode, 16)
                # LED mode update
                led_mode = {"timelapse": "timelapse_idle",
                            "macro":     "macro_idle",
                            "cinematic": "cinematic"}.get(mode, "timelapse_idle")
                status_leds.set_mode(led_mode)
                try:
                    hw.set_fan(auto_fan)
                    await websocket.send_json({"type": "log",
                        "msg": f"Mode: {mode.upper()} — fan {auto_fan}%, microstepping 1/{mstep}"})
                except Exception:
                    pass
                save_session()

            elif cmd == "set_camera_orientation":
                orient = msg.get("value", "landscape")
                if orient in ("landscape", "portrait_cw", "portrait_ccw", "inverted"):
                    state["camera_orientation"] = orient
                    save_session()
                    await broadcast({"type": "camera_orientation", "value": orient})

            elif cmd == "set_cine_fps":
                fps = int(msg.get("value", 24))
                if fps in (24, 25, 30, 60):
                    state["cine_fps"] = fps
                    save_session()
                    await websocket.send_json({"type": "log",
                        "msg": f"Recording frame rate set to {fps}fps"})

            elif cmd == "set_save_path":
                state["save_path"] = msg.get("value", state["save_path"])
                save_session()

            elif cmd == "take_preview":
                if _HAS_PICAM and picam:
                    try:
                        frame = await asyncio.to_thread(picam.capture_array)
                        prev  = os.path.join(state["save_path"], "preview.jpg")
                        os.makedirs(state["save_path"], exist_ok=True)
                        _, buf = cv2.imencode('.jpg', frame)
                        with open(prev, 'wb') as f: f.write(buf.tobytes())
                        await websocket.send_json({"type":"log","msg":f"Preview: {prev}"})
                    except Exception as e:
                        await websocket.send_json({"type":"log","msg":f"Preview fail: {e}"})

            # ── PICAM MANUAL SETTINGS ─────────────────────────────────────────
            elif cmd == "set_picam_settings":
                state["picam_ae"]        = msg.get("ae",       True)
                state["picam_awb"]       = msg.get("awb",      True)
                state["picam_shutter_s"] = float(msg.get("shutter_s", 1/125))
                state["picam_iso"]       = int(  msg.get("iso",     400))
                state["picam_kelvin"]    = int(  msg.get("kelvin",  5500))
                apply_picam_settings()
                save_session()

            # ── HOLY GRAIL SETTINGS ───────────────────────────────────────────
            elif cmd == "set_hg_settings":
                try:
                    s = hg.settings
                    s_interval_sec = getattr(s, 'interval_sec', getattr(s, 'interval_day', 5.0))
                    hg.set_settings(HGSettings(
                        enabled           = msg.get("enabled",            s.enabled),
                        lat               = float(msg.get("lat",           s.lat)),
                        lon               = float(msg.get("lon",           s.lon)),
                        tz                = msg.get("tz",                  s.tz),
                        cam_az            = float(msg.get("cam_az",        s.cam_az)),
                        cam_alt           = float(msg.get("cam_alt",       s.cam_alt)),
                        hfov              = float(msg.get("hfov",          s.hfov)),
                        vfov              = float(msg.get("vfov",          s.vfov)),
                        interval_sec      = float(msg.get("interval_sec",  s_interval_sec)),
                        frames            = int(  msg.get("frames",        s.frames)),
                        vibration_delay   = float(msg.get("vibration_delay",s.vibration_delay)),
                        exposure_margin   = float(msg.get("exposure_margin",s.exposure_margin)),
                        ev_day            = float(msg.get("ev_day",        s.ev_day)),
                        ev_golden         = float(msg.get("ev_golden",     s.ev_golden)),
                        ev_twilight       = float(msg.get("ev_twilight",   s.ev_twilight)),
                        ev_night          = float(msg.get("ev_night",      s.ev_night)),
                        kelvin_day        = int(  msg.get("kelvin_day",    s.kelvin_day)),
                        kelvin_golden     = int(  msg.get("kelvin_golden", s.kelvin_golden)),
                        kelvin_twilight   = int(  msg.get("kelvin_twilight",s.kelvin_twilight)),
                        kelvin_night      = int(  msg.get("kelvin_night",  s.kelvin_night)),
                        interval_day      = float(msg.get("interval_day",  s.interval_day)),
                        interval_golden   = float(msg.get("interval_golden",s.interval_golden)),
                        interval_twilight = float(msg.get("interval_twilight",s.interval_twilight)),
                        interval_night    = float(msg.get("interval_night",s.interval_night)),
                        iso_min           = int(  msg.get("iso_min",       s.iso_min)),
                        iso_max           = int(  msg.get("iso_max",       s.iso_max)),
                        aperture_day      = float(msg.get("aperture_day",  s.aperture_day)),
                        aperture_night    = float(msg.get("aperture_night",s.aperture_night)),
                    ))
                    save_session()
                    await websocket.send_json({"type":"log","msg":"HG settings applied."})
                except Exception as e:
                    await websocket.send_json({"type":"log","msg":f"HG error: {e}"})

            # ── SEQUENCE ──────────────────────────────────────────────────────
            elif cmd == "set_total_frames":
                state["total_frames"] = int(msg.get("value", 300))
                save_session()

            elif cmd == "start_run":
                # Wait briefly for any in-flight worker to finish cleanup
                # (the worker clears stop_event as its very last act)
                deadline = asyncio.get_event_loop().time() + 3.0
                while state["stop_event"].is_set() and asyncio.get_event_loop().time() < deadline:
                    await asyncio.sleep(0.05)
                state["stop_event"].clear()   # safety clear after wait

                if not state["is_running"]:
                    state["total_frames"]    = int(  msg.get("total_frames",  state["total_frames"]))
                    state["vibe_delay"]      = float(msg.get("vibe_delay",    state["vibe_delay"]))
                    state["exp_margin"]      = float(msg.get("exp_margin",    state["exp_margin"]))
                    state["save_path"]       = msg.get("save_path",           state["save_path"])
                    state["trigger_mode"]    = msg.get("trigger_mode",        state["trigger_mode"])
                    state["manual_interval"] = float(msg.get("interval",      state.get("manual_interval", 5.0)))
                    state["schedule_start"]  = msg.get("schedule_start", None)

                    # ── Disk space preflight ───────────────────────────────────
                    # Estimate bytes needed: IMX477 DNG ~25MB, Sony ARW ~30MB
                    BYTES_PER_FRAME = 30_000_000 if state["active_camera"] == "sony" else 25_000_000
                    needed = state["total_frames"] * BYTES_PER_FRAME
                    try:
                        save_p = state["save_path"] if os.path.exists(state["save_path"]) else "/"
                        free   = shutil.disk_usage(save_p).free
                        if free < needed:
                            free_gb   = free   / 1e9
                            needed_gb = needed / 1e9
                            warn = (f"⚠ LOW DISK SPACE: {free_gb:.1f} GB free but ~{needed_gb:.1f} GB needed "
                                    f"for {state['total_frames']} frames. Sequence may fail mid-run.")
                            logger.warning(warn)
                            await websocket.send_json({"type": "disk_warn", "msg": warn,
                                                       "free_gb": round(free_gb, 2),
                                                       "needed_gb": round(needed_gb, 2)})
                            # Don't block — let user proceed with warning visible
                    except Exception as de:
                        logger.warning(f"Disk preflight check failed: {de}")

                    if hg.settings.enabled:
                        base_iv = getattr(hg.settings, 'interval_sec',
                                          getattr(hg.settings, 'interval_day', 5.0))
                    else:
                        base_iv = state["manual_interval"]

                    save_session()
                    asyncio.create_task(timelapse_worker(base_iv))

            elif cmd == "stop":
                state["stop_event"].set()
                hw.set_tmc_velocity(0, 0); hw.set_tmc_velocity(1, 0)
                hw.set_tmc_velocity(2, 0); hw.enable_motors(False)
                state["is_running"] = False
                save_session()
                await broadcast({"type": "run_state", "running": False})
                await broadcast({"type": "log", "msg": "Sequence stopped."})
                # Safety: clear stop_event after 3s in case the worker already
                # exited and won't clear it itself (camera error, task cancelled, etc.)
                async def _deferred_clear():
                    await asyncio.sleep(3.0)
                    state["stop_event"].clear()
                    logger.info("stop_event auto-cleared after stop command.")
                asyncio.create_task(_deferred_clear())

            elif cmd == "reset_session":
                    # ── Full recovery reset — works even during a running sequence ──
                    # Force-stop any running sequence first
                    if state["is_running"]:
                        state["stop_event"].set()
                        state["is_running"] = False
                        await asyncio.sleep(0.3)   # let sequence loop wake and exit

                    state["stop_event"].clear()
                    if _macro_task and not _macro_task.done():
                        _macro_task.cancel()
                    if _inertia:
                        _inertia.stop()
                    if _prog_move:
                        _prog_move.stop()
                    hw.set_tmc_velocity(0, 0); hw.set_tmc_velocity(1, 0)
                    hw.set_tmc_velocity(2, 0); hw.enable_motors(False)
                    try: hw.set_relay1(False); hw.set_relay2(False)
                    except Exception: pass

                    # Recover camera if it's in a bad state
                    if _HAS_PICAM and picam:
                        try:
                            _restart_picam()
                        except Exception:
                            pass

                    # Wipe session so restarted process starts at frame 0
                    reset_session()
                    await websocket.send_json({"type": "log",
                        "msg": "♻ Restarting server process…"})
                    await asyncio.sleep(0.3)
                    hw.cleanup()
                    import os as _os
                    _os.execv(_os.sys.executable,
                              [_os.sys.executable] + _os.sys.argv)

            # ── MOTION TEST ───────────────────────────────────────────────────
            elif cmd == "run_motion_test":
                if state["is_running"]: continue
                axis = msg.get("axis","slider"); curve = msg.get("curve","linear")
                total = float(msg.get("total",100)); intervals = int(msg.get("intervals",50))
                def _mt():
                    weights = normalize(CURVE_FUNCTIONS.get(curve, CURVE_FUNCTIONS["linear"])(intervals))
                    hw.enable_motors(True)
                    for w in weights:
                        if state["stop_event"].is_set(): break
                        if axis == "slider":
                            hw.move_axes_simultaneous(int(w*total*slider_axis.steps_per_mm),0,0,0.5)
                        elif axis == "pan":
                            hw.move_axes_simultaneous(0,int(w*total*pan_axis.steps_per_deg),0,0.5)
                        elif axis == "tilt":
                            hw.move_axes_simultaneous(0,0,int(w*total*tilt_axis.steps_per_deg),0.5)
                    hw.enable_motors(False)
                asyncio.create_task(asyncio.to_thread(_mt))

            elif cmd == "home_axis":
                await websocket.send_json({"type":"log","msg":"Homing: endstop hardware pending."})

            elif cmd == "add_node":
                engine.add_keyframe(slider_axis.current_mm, pan_axis.current_deg, tilt_axis.current_deg)
                # Invalidate trajectory cache when keyframes change
                if hasattr(timelapse_worker, '_traj_cache'):
                    del timelapse_worker._traj_cache
                await websocket.send_json({"type":"status","nodes":len(engine.keyframes)})

            elif cmd == "clear_nodes":
                engine.clear_keyframes()
                if hasattr(timelapse_worker, '_traj_cache'):
                    del timelapse_worker._traj_cache
                await websocket.send_json({"type":"status","nodes":0})

            elif cmd == "set_relay":
                relay = int(msg.get("relay",1)); on = bool(msg.get("on",False))
                if relay == 1:
                    hw.set_relay1(on)
                    status_leds.set_relay(1, on)
                elif relay == 2:
                    hw.set_relay2(on)
                    status_leds.set_relay(2, on)

            elif cmd == "set_fan":
                hw.set_fan(int(msg.get("value",0)))

            elif cmd == "build_sky_map":
                await websocket.send_json({"type":"log",
                    "msg":"Sky Map: wiring requires motor + camera connection — test pending."})

            elif cmd == "set_preview_camera":
                # Toggle preview source independently from capture camera
                cam = msg.get("camera", state["active_camera"])
                state["preview_camera"] = cam
                # Cache-bust stream — client must refresh video_feed after this
                await websocket.send_json({"type": "preview_camera_changed",
                                           "camera": cam,
                                           "label": "Sony (framing)" if cam == "sony" else "PiCam (motion zone)"})

            elif cmd == "create_folder":
                parent = msg.get("path", "/home/tim/Pictures")
                name   = msg.get("name", "").strip().replace("/", "_").replace("..", "")
                if not name:
                    await websocket.send_json({"type":"log","msg":"Create folder: name required."})
                else:
                    new_path = os.path.join(parent, name)
                    try:
                        os.makedirs(new_path, exist_ok=True)
                        await websocket.send_json({"type":"folder_created","path":new_path})
                    except Exception as e:
                        await websocket.send_json({"type":"log","msg":f"Create folder error: {e}"})

            elif cmd == "connect_sony_wifi":
                ssid     = msg.get("ssid",""); password = msg.get("password","")
                ip       = msg.get("ip","192.168.122.1")
                state["sony_ssid"] = ssid; state["sony_ip"] = ip
                async def _sc():
                    await websocket.send_json({"type":"sony_status","connected":False,"msg":f"Joining {ssid}…"})
                    try:
                        res = await asyncio.to_thread(lambda: subprocess.run(
                            ["nmcli","dev","wifi","connect",ssid,"password",password,"ifname","wlan1"],
                            capture_output=True, text=True, timeout=20))
                        if res.returncode == 0:
                            import socket as _sock
                            try:
                                c = _sock.create_connection((ip,15740),timeout=5); c.close()
                                await websocket.send_json({"type":"sony_status","connected":True,"ip":ip,"model":"Sony"})
                            except:
                                await websocket.send_json({"type":"sony_status","connected":False,"error":f"WiFi OK but camera unreachable at {ip}"})
                        else:
                            await websocket.send_json({"type":"sony_status","connected":False,"error":res.stderr.strip()})
                    except Exception as e:
                        await websocket.send_json({"type":"sony_status","connected":False,"error":str(e)})
                asyncio.create_task(_sc())

            # ── MACRO MODE ────────────────────────────────────────────────────
            elif cmd == "macro_set_soft_limits":
                # Update software travel limits for all three axes
                if "rail_min" in msg: slider_axis.soft_min = float(msg["rail_min"])
                if "rail_max" in msg: slider_axis.soft_max = float(msg["rail_max"])
                if "pan_min"  in msg: pan_axis.soft_min    = float(msg["pan_min"])
                if "pan_max"  in msg: pan_axis.soft_max    = float(msg["pan_max"])
                if "tilt_min" in msg: tilt_axis.soft_min   = float(msg["tilt_min"])
                if "tilt_max" in msg: tilt_axis.soft_max   = float(msg["tilt_max"])
                save_session()
                await websocket.send_json({"type": "log",
                    "msg": f"Soft limits updated: rail [{slider_axis.soft_min:.1f}…{slider_axis.soft_max:.1f}] mm  "
                           f"pan [{pan_axis.soft_min:.1f}…{pan_axis.soft_max:.1f}]°  "
                           f"tilt [{tilt_axis.soft_min:.1f}…{tilt_axis.soft_max:.1f}]°"})

            elif cmd == "macro_set_rail_start":
                # Mark current rail position as the focus stack start
                state["macro_rail_start_mm"] = slider_axis.current_mm
                save_session()
                await websocket.send_json({"type": "macro_rail_mark",
                    "which": "start", "mm": slider_axis.current_mm})

            elif cmd == "macro_set_rail_end":
                state["macro_rail_end_mm"] = slider_axis.current_mm
                save_session()
                await websocket.send_json({"type": "macro_rail_mark",
                    "which": "end", "mm": slider_axis.current_mm})

            elif cmd == "macro_set_rotation_start":
                state["macro_rotation_start_deg"] = pan_axis.current_deg
                save_session()
                await websocket.send_json({"type": "macro_rotation_mark",
                    "which": "start", "deg": pan_axis.current_deg})

            elif cmd == "macro_set_rotation_end":
                state["macro_rotation_end_deg"] = pan_axis.current_deg
                save_session()
                await websocket.send_json({"type": "macro_rotation_mark",
                    "which": "end", "deg": pan_axis.current_deg})

            elif cmd == "macro_set_aux_start":
                state["macro_aux_start_deg"] = tilt_axis.current_deg
                save_session()
                await websocket.send_json({"type": "macro_aux_mark",
                    "which": "start", "deg": tilt_axis.current_deg})

            elif cmd == "macro_set_aux_end":
                state["macro_aux_end_deg"] = tilt_axis.current_deg
                save_session()
                await websocket.send_json({"type": "macro_aux_mark",
                    "which": "end", "deg": tilt_axis.current_deg})

            elif cmd == "macro_calc":
                # Return live image count / storage estimate without starting
                try:
                    sess = _build_macro_session(msg)
                    await websocket.send_json({
                        "type":         "macro_calc",
                        "frames":       rail_frame_count(sess),
                        "total_images": total_image_count(sess),
                        "storage_gb":   round(estimated_storage_gb(sess), 2),
                        "travel_mm":    round(abs(sess.rail_end_mm - sess.rail_start_mm), 3),
                    })
                except Exception as e:
                    await websocket.send_json({"type":"log","msg":f"Macro calc error: {e}"})

            elif cmd == "macro_start":
                if state["is_running"]:
                    await websocket.send_json({"type":"log",
                        "msg":"⚠ Cannot start macro — another sequence is running."})
                else:
                    try:
                        sess = _build_macro_session(msg)
                        state["is_running"] = True
                        save_session()
                        await broadcast({"type": "run_state", "running": True})

                        macro_eng = MacroEngine(
                            hardware        = hw,
                            capture_fn      = macro_capture,
                            apply_camera_fn = macro_apply_camera,
                            broadcast_fn    = broadcast,
                        )
                        # Sync engine position from live axis trackers
                        macro_eng.rail_pos_mm  = slider_axis.current_mm
                        macro_eng.pan_pos_deg  = pan_axis.current_deg
                        macro_eng.tilt_pos_deg = tilt_axis.current_deg

                        async def _macro_run():
                            try:
                                await macro_eng.run(sess)
                            finally:
                                state["is_running"] = False
                                # Sync axis positions back from engine
                                slider_axis.current_mm  = macro_eng.rail_pos_mm
                                pan_axis.current_deg    = macro_eng.pan_pos_deg
                                tilt_axis.current_deg   = macro_eng.tilt_pos_deg
                                save_session()
                                await broadcast({"type": "run_state", "running": False})

                        _macro_task = asyncio.create_task(_macro_run())
                    except Exception as e:
                        state["is_running"] = False
                        await websocket.send_json({"type":"log","msg":f"Macro start error: {e}"})

            elif cmd == "macro_stop":
                if _macro_task and not _macro_task.done():
                    # Signal the engine — it will clean up and broadcast macro_done
                    state["stop_event"].set()
                    # Also flag via state for the worker loop
                    state["is_running"] = False
                    hw.set_tmc_velocity(0, 0); hw.set_tmc_velocity(1, 0)
                    hw.set_tmc_velocity(2, 0); hw.enable_motors(False)
                    await broadcast({"type": "run_state", "running": False})
                    await broadcast({"type": "log", "msg": "Macro sequence stopped."})

            elif cmd == "macro_save_lens_profile":
                state["macro_lens_profile"] = msg.get("profile", {})
                save_session()
                await websocket.send_json({"type":"log","msg":"Lens profile saved."})

            elif cmd == "macro_load_lens_profiles":
                profiles = state.get("macro_saved_lens_profiles", {})
                await websocket.send_json({"type":"macro_lens_profiles","profiles":profiles})

            elif cmd == "macro_store_lens_profile":
                name    = msg.get("name","default").strip()
                profile = msg.get("profile", {})
                if name:
                    profs = state.get("macro_saved_lens_profiles", {})
                    profs[name] = profile
                    state["macro_saved_lens_profiles"] = profs
                    save_session()
                    await websocket.send_json({"type":"log","msg":f"Lens profile '{name}' stored."})
                    await websocket.send_json({"type":"macro_lens_profiles","profiles":profs})

            # ── CINEMATIC MODE ────────────────────────────────────────────────

            elif cmd == "cinematic_set_mode":
                global _cinematic_mode
                _cinematic_mode = msg.get("value", "live")
                # Stop inertia if switching away from live
                if _cinematic_mode != "live" and _inertia:
                    _inertia.stop()

            elif cmd == "cinematic_set_rail_tilt":
                deg = float(msg.get("degrees", 0.0))
                state["rail_tilt_deg"] = deg
                _arctan.set_rail_tilt(deg)
                save_session()
                await websocket.send_json({"type": "log",
                    "msg": f"Rail tilt set to {deg:.1f}°"})

            elif cmd == "cinematic_set_high_power":
                enabled = bool(msg.get("enabled", False))
                state["high_power_mode"] = enabled
                # 16 = standard (~800mA), 24 = high power (~1.2A)
                current = 24 if enabled else 16
                for addr in (0, 1, 2):
                    hw.set_tmc_current(addr, run_current=current, hold_current=current//2)
                save_session()
                await websocket.send_json({"type": "log",
                    "msg": f"High power mode {'ON' if enabled else 'OFF'} "
                           f"({'1.2A' if enabled else '800mA'} run current)."})

            # ── SOFT LIMITS ───────────────────────────────────────────────────
            elif cmd == "cinematic_calibrate_limit":
                axis = msg.get("axis", "slider")   # slider | pan | tilt
                end  = msg.get("end",  "min")       # min | max
                ax_obj = {"slider": slider_axis, "pan": pan_axis, "tilt": tilt_axis}.get(axis)
                guard_ax = getattr(_soft_guard, axis, None)
                if ax_obj and guard_ax:
                    pos = ax_obj.current_mm if axis == "slider" else ax_obj.current_deg
                    if end == "min":
                        guard_ax.set_min(pos)
                    else:
                        guard_ax.set_max(pos)
                    save_session()
                    await broadcast({"type": "cinematic_limits",
                                     "limits": _soft_guard.status()})
                    await websocket.send_json({"type": "log",
                        "msg": f"Soft limit: {axis} {end} = {pos:.2f}"})

            elif cmd == "cinematic_clear_limit":
                axis = msg.get("axis", "slider")
                guard_ax = getattr(_soft_guard, axis, None)
                if guard_ax:
                    guard_ax.min_unit = None
                    guard_ax.max_unit = None
                    guard_ax._update_cal()
                    await broadcast({"type": "cinematic_limits",
                                     "limits": _soft_guard.status()})

            elif cmd == "cinematic_get_limits":
                await websocket.send_json({"type": "cinematic_limits",
                                           "limits": _soft_guard.status()})

            # ── LIVE / INERTIA ────────────────────────────────────────────────
            elif cmd == "cinematic_live_start":
                if not state["is_running"] and _inertia:
                    state["is_running"] = True
                    _cinematic_mode = "live"
                    _inertia.mass = float(msg.get("mass", _inertia.mass))
                    _inertia.drag = float(msg.get("drag", _inertia.drag))
                    _inertia.start()
                    hw.set_fan(80)   # boost fan during continuous motor movement
                    await broadcast({"type": "run_state", "running": True})
                    await broadcast({"type": "log", "msg": "Live cinematic mode started."})

            elif cmd == "cinematic_live_stop":
                if _inertia:
                    _inertia.stop()
                state["is_running"] = False
                hw.set_fan(60)   # back to cinematic idle fan
                save_session()
                await broadcast({"type": "run_state", "running": False})

            elif cmd == "cinematic_set_inertia":
                if _inertia:
                    preset = msg.get("preset")
                    if preset and preset in RIG_PRESETS:
                        _inertia.set_preset(preset)
                    else:
                        _inertia.set_params(
                            float(msg.get("mass", _inertia.mass)),
                            float(msg.get("drag", _inertia.drag))
                        )
                    await websocket.send_json({"type": "cinematic_inertia",
                        "mass": _inertia.mass, "drag": _inertia.drag})

            # ── ARCTAN TRACKER ────────────────────────────────────────────────
            elif cmd == "arctan_add_point":
                _arctan.add_point(
                    slider_mm = slider_axis.current_mm,
                    pan_deg   = pan_axis.current_deg,
                    tilt_deg  = tilt_axis.current_deg,
                )
                result = {
                    "type":     "arctan_status",
                    "points":   len(_arctan.points),
                    "solved":   _arctan.is_solved,
                    "residual": round(_arctan.residual_deg, 3),
                    "warning":  _arctan.warning,
                    "subject":  _arctan.subject.tolist() if _arctan.subject is not None else None,
                }
                await broadcast(result)
                await websocket.send_json({"type": "log",
                    "msg": f"Arctan: point {len(_arctan.points)} marked. "
                           + (f"Solved — RMS {_arctan.residual_deg:.2f}°" if _arctan.is_solved else "Need more points.")
                           + (f" ⚠ {_arctan.warning}" if _arctan.warning else "")})

            elif cmd == "arctan_clear":
                _arctan.clear_points()
                if _inertia:
                    _inertia.arctan_active = False
                await broadcast({"type": "arctan_status", "points": 0,
                                 "solved": False, "residual": 0, "warning": ""})

            elif cmd == "arctan_enable":
                enabled = bool(msg.get("enabled", False))
                if enabled and not _arctan.is_solved:
                    await websocket.send_json({"type": "log",
                        "msg": "⚠ Arctan lock needs calibration points first."})
                else:
                    if _inertia:
                        _inertia.arctan_active = enabled
                    if _prog_move:
                        _prog_move.tracker = _arctan if enabled else None
                    await broadcast({"type": "arctan_enabled", "enabled": enabled})
                    await websocket.send_json({"type": "log",
                        "msg": f"Arctan lock {'enabled' if enabled else 'disabled'}."})

            # ── PROGRAMMED MOVES ──────────────────────────────────────────────
            elif cmd == "cinematic_set_origin":
                if _prog_move:
                    _prog_move.set_origin(slider_axis.current_mm,
                                          pan_axis.current_deg,
                                          tilt_axis.current_deg)
                    state["cinematic_origin"] = {
                        "slider_mm": slider_axis.current_mm,
                        "pan_deg":   pan_axis.current_deg,
                        "tilt_deg":  tilt_axis.current_deg,
                    }
                    save_session()
                    await broadcast({"type": "cinematic_origin_set",
                                     "slider_mm": slider_axis.current_mm,
                                     "pan_deg":   pan_axis.current_deg,
                                     "tilt_deg":  tilt_axis.current_deg})
                    await websocket.send_json({"type": "log",
                        "msg": f"Origin set at s={slider_axis.current_mm:.1f}mm "
                               f"pan={pan_axis.current_deg:.1f}° "
                               f"tilt={tilt_axis.current_deg:.1f}°"})

            elif cmd == "cinematic_add_keyframe":
                if _prog_move:
                    idx = _prog_move.add_keyframe(
                        slider_mm  = float(msg.get("slider_mm", slider_axis.current_mm)),
                        pan_deg    = float(msg.get("pan_deg",   pan_axis.current_deg)),
                        tilt_deg   = float(msg.get("tilt_deg",  tilt_axis.current_deg)),
                        duration_s = float(msg.get("duration_s", 3.0)),
                        easing     = msg.get("easing", "gaussian"),
                    )
                    await broadcast({"type": "cinematic_keyframes",
                                     "keyframes": _keyframes_to_list()})
                    await websocket.send_json({"type": "log",
                        "msg": f"Keyframe {idx+1} added."})

            elif cmd == "cinematic_update_keyframe":
                if _prog_move:
                    idx = int(msg.get("index", 0))
                    kwargs = {k: msg[k] for k in
                              ("slider_mm","pan_deg","tilt_deg","duration_s","easing")
                              if k in msg}
                    _prog_move.update_keyframe(idx, **kwargs)
                    await broadcast({"type": "cinematic_keyframes",
                                     "keyframes": _keyframes_to_list()})

            elif cmd == "cinematic_remove_keyframe":
                if _prog_move:
                    _prog_move.remove_keyframe(int(msg.get("index", 0)))
                    await broadcast({"type": "cinematic_keyframes",
                                     "keyframes": _keyframes_to_list()})

            elif cmd == "cinematic_clear_keyframes":
                if _prog_move:
                    _prog_move.clear_keyframes()
                    await broadcast({"type": "cinematic_keyframes", "keyframes": []})

            elif cmd == "cinematic_set_preroll":
                if _prog_move:
                    _prog_move.preroll_s = float(msg.get("seconds", 3.0))

            elif cmd == "cinematic_set_loop":
                if _prog_move:
                    _prog_move.loop = bool(msg.get("enabled", False))

            elif cmd == "cinematic_play":
                global _prog_task
                if not state["is_running"] and _prog_move and len(_prog_move.keyframes) >= 2:
                    state["is_running"] = True
                    _cinematic_mode = "programmed"
                    await broadcast({"type": "run_state", "running": True})

                    async def _play_wrapper():
                        try:
                            await _prog_move.play()
                        finally:
                            state["is_running"] = False
                            save_session()
                            await broadcast({"type": "run_state", "running": False})

                    _prog_task = asyncio.create_task(_play_wrapper())
                else:
                    await websocket.send_json({"type": "log",
                        "msg": "⚠ Need at least 2 keyframes and system must be idle."})

            elif cmd == "cinematic_stop":
                if _prog_move:
                    _prog_move.stop()
                if _inertia:
                    _inertia.stop()
                state["is_running"] = False
                hw.set_tmc_velocity(0, 0); hw.set_tmc_velocity(1, 0)
                hw.set_tmc_velocity(2, 0); hw.enable_motors(False)
                save_session()
                await broadcast({"type": "run_state", "running": False})

            elif cmd == "cinematic_return_to_start":
                if _prog_move and not state["is_running"]:
                    state["is_running"] = True
                    await broadcast({"type": "run_state", "running": True})
                    async def _return():
                        try:
                            await _prog_move.return_to_start()
                        finally:
                            state["is_running"] = False
                            await broadcast({"type": "run_state", "running": False})
                    asyncio.create_task(_return())

            # ── MOVE LIBRARY ──────────────────────────────────────────────────
            elif cmd == "cinematic_save_move":
                if _prog_move and _prog_move.keyframes:
                    name  = msg.get("name", "").strip()
                    notes = msg.get("notes", "").strip()
                    if not name:
                        await websocket.send_json({"type":"log","msg":"Enter a name for the move."})
                    else:
                        try:
                            _move_library.save_move(name, _prog_move.keyframes, notes)
                            await websocket.send_json({"type": "log",
                                "msg": f"Move '{name}' saved to library."})
                            await websocket.send_json({"type": "cinematic_moves",
                                "moves": _move_library.list_moves()})
                        except Exception as e:
                            await websocket.send_json({"type":"log","msg":f"Save failed: {e}"})
                else:
                    await websocket.send_json({"type":"log","msg":"No keyframes to save."})

            elif cmd == "cinematic_load_move":
                name = msg.get("name","")
                try:
                    kfs = _move_library.load_move(name)
                    if _prog_move:
                        _prog_move.clear_keyframes()
                        for kf in kfs:
                            _prog_move.keyframes.append(kf)
                    await broadcast({"type": "cinematic_keyframes",
                                     "keyframes": _keyframes_to_list()})
                    await websocket.send_json({"type": "log",
                        "msg": f"Move '{name}' loaded ({len(kfs)} keyframes). Set origin before playing."})
                except KeyError as e:
                    await websocket.send_json({"type":"log","msg":str(e)})

            elif cmd == "cinematic_delete_move":
                name = msg.get("name","")
                _move_library.delete_move(name)
                await websocket.send_json({"type": "cinematic_moves",
                    "moves": _move_library.list_moves()})
                await websocket.send_json({"type":"log","msg":f"Move '{name}' deleted."})

            elif cmd == "cinematic_rename_move":
                old = msg.get("old_name",""); new = msg.get("new_name","").strip()
                if old and new:
                    try:
                        _move_library.rename_move(old, new)
                        await websocket.send_json({"type": "cinematic_moves",
                            "moves": _move_library.list_moves()})
                    except KeyError as e:
                        await websocket.send_json({"type":"log","msg":str(e)})

            elif cmd == "cinematic_list_moves":
                await websocket.send_json({"type": "cinematic_moves",
                    "moves": _move_library.list_moves()})

            # ── VIDEO RECORDING ───────────────────────────────────────────────
            elif cmd == "record_start":
                if _recording:
                    await websocket.send_json({"type":"log","msg":"Already recording."})
                else:
                    cam = state.get("active_camera","picam")
                    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    dest = os.path.join(state.get("save_path","/home/tim/Pictures/PiSlider"),
                                        f"CINE_{ts}.mp4")
                    os.makedirs(os.path.dirname(dest), exist_ok=True)

                    if cam == "picam":
                        fps = int(state.get("cine_fps", 24))
                        ok = await asyncio.to_thread(_start_picam_video, dest, fps)
                        if ok:
                            status_leds.set_recording(True)
                            await broadcast({"type": "record_state", "recording": True,
                                             "path": dest})
                            await websocket.send_json({"type":"log",
                                "msg":f"● Recording: {os.path.basename(dest)}"})
                        else:
                            await websocket.send_json({"type":"log",
                                "msg":"⚠ Camera recording failed — check logs."})
                    elif cam == "sony":
                        ok, msg_txt = await _toggle_sony_record()
                        if ok:
                            _recording = True
                            _record_start_time = time.time()
                            status_leds.set_recording(True)
                            await broadcast({"type": "record_state", "recording": True,
                                             "path": "Sony camera"})
                        else:
                            await websocket.send_json({"type":"log",
                                "msg":f"⚠ Sony record: {msg_txt}. Trigger manually if needed."})

            elif cmd == "record_stop":
                if not _recording:
                    await websocket.send_json({"type":"log","msg":"Not recording."})
                else:
                    cam = state.get("active_camera","picam")
                    saved_path = _video_output_path  # capture before _stop clears it
                    elapsed = time.time() - (_record_start_time or time.time())
                    if cam == "picam":
                        await asyncio.to_thread(_stop_picam_video)
                    elif cam == "sony":
                        await _toggle_sony_record()
                        _recording = False
                    status_leds.set_recording(False)
                    await broadcast({"type": "record_state", "recording": False})
                    # Report file size so user knows it was actually saved
                    size_msg = ""
                    if saved_path and os.path.exists(saved_path):
                        size_mb = os.path.getsize(saved_path) / 1_048_576
                        size_msg = f" ({size_mb:.1f}MB)"
                    elif saved_path:
                        size_msg = " ⚠ file not found — is ffmpeg installed?"
                    await websocket.send_json({"type":"log",
                        "msg":f"■ Recording stopped. {elapsed:.1f}s{size_msg}"})

            elif cmd == "cinematic_get_state":
                # Full cinematic state restore on reconnect
                await websocket.send_json({
                    "type":      "cinematic_state",
                    "limits":    _soft_guard.status(),
                    "keyframes": _keyframes_to_list(),
                    "moves":     _move_library.list_moves(),
                    "arctan":    {
                        "points":   len(_arctan.points),
                        "solved":   _arctan.is_solved,
                        "residual": round(_arctan.residual_deg, 3),
                        "warning":  _arctan.warning,
                    },
                    "inertia": {
                        "mass":   _inertia.mass if _inertia else 0.4,
                        "drag":   _inertia.drag if _inertia else 0.55,
                    },
                    "recording":   _recording,
                    "rail_tilt":   state.get("rail_tilt_deg", 0.0),
                    "high_power":  state.get("high_power_mode", False),
                    "origin":      state.get("cinematic_origin", {}),
                })

            # ── TELEMETRY TICK ────────────────────────────────────────────────
            try:
                await websocket.send_json({
                    "type":  "status",
                    "frame": state["current_frame"],
                    "total": state["total_frames"],
                    "pos_s": slider_axis.current_mm,
                    "pos_p": pan_axis.current_deg,
                    "pos_t": tilt_axis.current_deg,
                })
            except Exception:
                break
            await asyncio.sleep(0.05)

    except WebSocketDisconnect:
        logger.info("WS client disconnected (browser closed).")
    finally:
        connected_clients.discard(websocket)
        # Only clear _active_ws if it's still pointing at THIS socket.
        # If a new client already took over (kicked us), don't clobber their reference.
        if _active_ws is websocket:
            _active_ws = None
            logger.info("WS: active client cleared.")




# ─── GRAPH WEBSOCKET (read-only, never kicked) ────────────────────────────────
@app.websocket("/ws-graph")
async def websocket_graph(websocket: WebSocket):
    """
    Read-only WebSocket for the session graph tab.
    Receives all broadcasts (status, log, run_state, etc.) but:
    - never triggers the single-instance kick logic
    - never accepts control commands
    Multiple graph tabs can be open simultaneously alongside the main UI.
    """
    await websocket.accept()
    _graph_clients.add(websocket)
    # Send current frame count so the graph tab knows where we are
    await websocket.send_json({
        "type":    "init",
        "running": state["is_running"],
        "current_frame": state["current_frame"],
        "total_frames":  state["total_frames"],
    })
    try:
        while True:
            # Drain incoming messages (graph tab doesn't send commands, but
            # we need to receive to detect disconnects)
            await websocket.receive_text()
    except Exception:
        pass
    finally:
        _graph_clients.discard(websocket)

# ─── STATIC & ROOT ────────────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="web"), name="static")



@app.get("/thumbs/{frame_id}")
async def get_thumb(frame_id: str):
    """Serve a sequence thumbnail for the graph timelapse player."""
    from fastapi.responses import FileResponse, Response
    thumb_path = os.path.join(state["save_path"], "thumbs", f"THUMB_{frame_id}.jpg")
    if os.path.exists(thumb_path):
        # Cache real thumbs for 1 year — they never change once written
        return FileResponse(thumb_path, media_type="image/jpeg",
                            headers={"Cache-Control": "public, max-age=31536000"})
    # Thumb not found — return 404 so browser doesn't cache the miss
    from fastapi import HTTPException
    raise HTTPException(status_code=404, detail="Thumb not ready yet")



@app.get("/api/thumb-list")
async def thumb_list():
    """Return list of frame IDs that have saved thumbnails, for the graph strip."""
    from fastapi.responses import JSONResponse
    thumb_dir = os.path.join(state["save_path"], "thumbs")
    if not os.path.isdir(thumb_dir):
        return JSONResponse({"frame_ids": []})
    ids = sorted([
        f.replace("THUMB_", "").replace(".jpg", "")
        for f in os.listdir(thumb_dir)
        if f.startswith("THUMB_") and f.endswith(".jpg")
    ])
    return JSONResponse({"frame_ids": ids})

@app.get("/graph")
async def graph_page():
    with open("web/graph.html") as f:
        return HTMLResponse(content=f.read())


@app.get("/api/session-history")
async def session_history():
    """Return buffered frame history for graph tab initial load / reconnect."""
    from fastapi.responses import JSONResponse
    return JSONResponse({"frames": list(_session_history)})


@app.get("/")
async def index():
    with open("web/index.html") as f:
        return HTMLResponse(content=f.read())


# ─── CINEMATIC / VIDEO RECORDING ─────────────────────────────────────────────

def _start_picam_video(output_path: str, fps: int = 24) -> bool:
    """
    Record to a raw .h264 file using FileOutput (no ffmpeg quoting issues),
    then remux to .mp4 on stop using subprocess with proper arg list.
    This avoids FfmpegOutput's internal shell command breaking on spaces in paths.
    """
    global _recording, _record_start_time, _video_output_path
    if not _HAS_PICAM or not picam:
        return False
    try:
        from picamera2.encoders import H264Encoder
        from picamera2.outputs import FileOutput

        bitrate = {24: 20_000_000, 25: 20_000_000,
                   30: 18_000_000, 60: 12_000_000}.get(fps, 18_000_000)

        encoder = H264Encoder(bitrate=bitrate)

        # Write raw H264 — FileOutput takes a plain file path, no shell involved
        h264_path = output_path.replace('.mp4', '.h264')
        output = FileOutput(h264_path)
        picam.start_encoder(encoder, output)

        _recording         = True
        _record_start_time = time.time()
        _video_output_path = output_path   # final .mp4 destination
        logger.info(f"Recording started (raw H264): {h264_path} @ {bitrate//1_000_000}Mbps")
        return True
    except Exception as e:
        logger.error(f"PiCam video start: {e}")
        _recording         = False
        _record_start_time = None
        return False


def _stop_picam_video():
    """Stop encoder, remux raw H264 → MP4, verify output file."""
    global _recording, _record_start_time, _video_output_path
    if not _HAS_PICAM or not picam:
        return
    mp4_path  = _video_output_path
    h264_path = mp4_path.replace('.mp4', '.h264') if mp4_path else None

    try:
        picam.stop_encoder()
        time.sleep(0.3)   # let encoder flush final NAL units
    except Exception as e:
        logger.warning(f"PiCam stop encoder: {e}")

    _recording         = False
    _record_start_time = None
    _video_output_path = None

    if not h264_path or not os.path.exists(h264_path):
        logger.error(f"H264 source not found: {h264_path}")
        return

    h264_size = os.path.getsize(h264_path)
    logger.info(f"H264 raw file: {h264_path} ({h264_size//1024}KB) — remuxing to MP4…")

    if h264_size < 1000:
        logger.warning("H264 file suspiciously small — encoder may have failed.")
        return

    # Remux using subprocess arg list — no shell, no quoting issues with spaces
    try:
        result = subprocess.run(
            ["ffmpeg", "-y",
             "-framerate", "25",        # container framerate hint
             "-i", h264_path,
             "-c:v", "copy",            # stream copy — instant, lossless
             mp4_path],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0 and os.path.exists(mp4_path):
            mp4_size = os.path.getsize(mp4_path)
            logger.info(f"MP4 saved: {mp4_path} ({mp4_size//1024}KB)")
            os.remove(h264_path)        # clean up raw file
        else:
            logger.error(f"ffmpeg remux failed (rc={result.returncode}): {result.stderr[-300:]}")
            logger.info(f"Raw H264 kept at: {h264_path}")
    except FileNotFoundError:
        logger.error("ffmpeg not found — install with: sudo apt install ffmpeg")
        logger.info(f"Raw H264 available at: {h264_path}")
    except subprocess.TimeoutExpired:
        logger.error("ffmpeg remux timed out")


async def _toggle_sony_record():
    """Attempt to toggle Sony movie record via gphoto2 PTP."""
    try:
        result = await asyncio.to_thread(lambda: __import__('subprocess').run(
            ["gphoto2", "--port", f"ptpip:{state['sony_ip']}",
             "--set-config", "movie=1"],
            capture_output=True, text=True, timeout=10
        ))
        if result.returncode == 0:
            return True, "Sony record toggled."
        else:
            return False, result.stderr.strip()
    except Exception as e:
        return False, str(e)


async def _gamepad_event_processor():
    """
    Consume gamepad events and route to inertia engine / cinematic commands.
    Runs as a background task for the lifetime of the app.
    """
    global _inertia, _recording, _cinematic_mode

    logger.info("Gamepad event processor started.")
    while True:
        try:
            event: GamepadEvent = await asyncio.wait_for(
                _gamepad_queue.get(), timeout=1.0
            )
        except asyncio.TimeoutError:
            continue
        except Exception:
            break

        name  = event.name
        value = event.value

        # ── Axis events → inertia engine ──────────────────────────────────────
        if _inertia and _cinematic_mode == "live":
            if name == "axis_slider":
                _inertia.set_target(value, _inertia._t_pan, _inertia._t_tilt)
            elif name == "axis_pan":
                _inertia.set_target(_inertia._t_slider, value, _inertia._t_tilt)
            elif name == "axis_tilt":
                _inertia.set_target(_inertia._t_slider, _inertia._t_pan, value)
            elif name == "btn_l1":
                _inertia.set_speed_modifier(value, _inertia._r1_held)
            elif name == "btn_r1":
                _inertia.set_speed_modifier(_inertia._l1_held, value)

        # ── D-pad nudge (all modes — for positioning) ────────────────────────
        if name == "dpad_x" and value != 0 and not state["is_running"]:
            # Nudge pan
            nudge_steps = int(value * _soft_guard.pan.speed_limit * 0.1)
            nudge_steps = _soft_guard.pan.clamp_velocity(
                nudge_steps, pan_axis.current_deg)
            hw.enable_motors(True)
            hw.set_tmc_velocity(1, nudge_steps)
            await asyncio.sleep(0.15)
            hw.set_tmc_velocity(1, 0)
            pan_axis.current_deg += value * 0.5
            hw.enable_motors(False)

        elif name == "dpad_y" and value != 0 and not state["is_running"]:
            # Nudge tilt
            nudge_steps = int(value * _soft_guard.tilt.speed_limit * 0.1)
            nudge_steps = _soft_guard.tilt.clamp_velocity(
                nudge_steps, tilt_axis.current_deg)
            hw.enable_motors(True)
            hw.set_tmc_velocity(2, nudge_steps)
            await asyncio.sleep(0.15)
            hw.set_tmc_velocity(2, 0)
            tilt_axis.current_deg += value * 0.5
            hw.enable_motors(False)

        # ── Buttons ───────────────────────────────────────────────────────────
        elif name == "btn_record" and value:
            # Toggle recording
            await broadcast({"type": "gamepad_btn", "btn": "record"})

        elif name == "btn_return" and value and _prog_move:
            asyncio.create_task(_prog_move.return_to_start())

        elif name == "btn_keyframe" and value and _prog_move:
            idx = _prog_move.add_keyframe(
                slider_axis.current_mm, pan_axis.current_deg, tilt_axis.current_deg
            )
            await broadcast({"type": "cinematic_keyframe_added", "index": idx,
                             "slider_mm": slider_axis.current_mm,
                             "pan_deg":   pan_axis.current_deg,
                             "tilt_deg":  tilt_axis.current_deg})

        elif name == "btn_play" and value:
            await broadcast({"type": "gamepad_btn", "btn": "play"})

        elif name == "btn_stop" and value:
            await broadcast({"type": "gamepad_btn", "btn": "stop"})

        elif name == "btn_arctan" and value:
            await broadcast({"type": "gamepad_btn", "btn": "arctan_toggle"})

        elif name == "btn_origin" and value:
            if _prog_move:
                _prog_move.set_origin(slider_axis.current_mm,
                                      pan_axis.current_deg,
                                      tilt_axis.current_deg)
            await broadcast({"type": "cinematic_origin_set",
                             "slider_mm": slider_axis.current_mm,
                             "pan_deg":   pan_axis.current_deg,
                             "tilt_deg":  tilt_axis.current_deg})

        elif name == "gamepad_connected":
            await broadcast({"type": "gamepad_status", "connected": True})
        elif name == "gamepad_disconnected":
            await broadcast({"type": "gamepad_status", "connected": False})


@app.on_event("startup")
async def _startup():
    """Start background tasks: gamepad reader + event processor."""
    global _gamepad_reader, _gamepad_task, _inertia, _prog_move, _soft_guard

    # Initialise cinematic objects with real hardware
    _inertia = InertiaEngine(
        hardware    = hw,
        guard       = _soft_guard,
        broadcast_fn = broadcast,
        slider_axis = slider_axis,
        pan_axis    = pan_axis,
        tilt_axis   = tilt_axis,
    )
    _prog_move = ProgrammedMove(
        hardware    = hw,
        guard       = _soft_guard,
        slider_axis = slider_axis,
        pan_axis    = pan_axis,
        tilt_axis   = tilt_axis,
        broadcast_fn = broadcast,
        arctan_tracker = _arctan,
    )

    # Start gamepad reader
    _gamepad_reader = GamepadReader(_gamepad_queue)
    _gamepad_task   = asyncio.create_task(_gamepad_reader.run())

    # Start gamepad event processor
    asyncio.create_task(_gamepad_event_processor())

    # Set LEDs to correct idle mode based on restored session state
    restored_mode = state.get("active_mode", "timelapse")
    led_mode = {"timelapse": "timelapse_idle",
                "macro":     "macro_idle",
                "cinematic": "cinematic"}.get(restored_mode, "timelapse_idle")
    status_leds.set_mode(led_mode)

    logger.info("Cinematic engine and gamepad reader started.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
