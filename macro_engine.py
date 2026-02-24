#!/usr/bin/env python3
"""
macro_engine.py — Macro Focus Stack / 3D Scan Engine for PiSlider

Supports two sub-modes:
  SCAN  — even angular spacing, full 360° coverage, science-first
  ART   — easing curves, partial arcs, cinematic movement

Sequence loop (outer → inner):
  Project
    Orbit (one physical rig setup, one rotation_axis_angle)
      Rotation position (pan motor, outer loop)
        Exposure slot (relay state + camera settings)
          Focus increment (rail motor, inner loop)
            Capture

Rail always returns to start_mm at max speed between stacks.
sequence.json is written incrementally — every completed stack is marked done.
project.json is updated at orbit start and orbit completion.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Callable

import numpy as np
from distributions import CURVE_FUNCTIONS, normalize

logger = logging.getLogger("PiSlider.Macro")

LEAD_SCREW_PITCH_MM = 2.0
STEPS_PER_MM        = 100.0
STEPS_PER_DEG       = 55.5
RAIL_RETURN_VELOCITY = 50000   # TMC2209 VACTUAL units — fast return, no capture


# ─── DATACLASSES ──────────────────────────────────────────────────────────────

@dataclass
class ExposureSlot:
    """One lighting / camera treatment applied at every focus position."""
    id:               str   = "slot_A"
    label:            str   = "diffuse"
    enabled:          bool  = True
    relay1:           bool  = False
    relay2:           bool  = False
    relay_settle_ms:  int   = 0      # wait BEFORE shutter after relay fires
    relay_release_ms: int   = 0      # wait AFTER shutter before relay releases
    iso:              int   = 400
    shutter_s:        float = 1/125
    kelvin:           int   = 5500
    ae:               bool  = False  # True = let camera auto-expose this slot
    awb:              bool  = False


@dataclass
class LensProfile:
    name:               str   = "unknown"
    lens_type:          str   = "macro"   # telecentric | macro | other
    magnification:      float = 1.0
    working_distance_mm: float = 0.0
    notes:              str   = ""


@dataclass
class MacroSession:
    """Complete configuration for one orbit sequence."""

    # Identity
    project_name:   str  = "macro_project"
    orbit_label:    str  = "orbit_001"
    session_mode:   str  = "scan"   # scan | art

    # Rail (focus axis — slider motor)
    rail_start_mm:  float = 0.0
    rail_end_mm:    float = 5.0
    rail_step_um:   float = 100.0    # primary input: step size in micrometres
    rail_soft_min:  float = -999.0   # software travel limit
    rail_soft_max:  float =  999.0

    # Rotation stage (pan motor)
    rotation_mode:      str   = "full"   # full | range
    rotation_start_deg: float = 0.0
    rotation_end_deg:   float = 360.0
    num_stacks:         int   = 36
    rotation_easing:    str   = "even"   # any key from distributions.CURVE_FUNCTIONS
    rotation_axis_angle_deg: float = 90.0   # physical tilt of rotation axis (metadata)
    rotation_axis_description: str = "vertical"

    # Aux axis (tilt motor — optional creative use)
    aux_enabled:    bool  = False
    aux_label:      str   = "aux"
    aux_start_deg:  float = 0.0
    aux_end_deg:    float = 0.0
    aux_easing:     str   = "even"
    aux_soft_min:   float = -90.0
    aux_soft_max:   float =  90.0

    # Exposure slots (up to 2 for now, designed for easy extension)
    slots: List[ExposureSlot] = field(default_factory=lambda: [ExposureSlot()])

    # Timing
    vibe_delay_s:   float = 0.5    # anti-vibration settle after each motor move
    exp_margin_s:   float = 0.2    # extra wait after shutter for DNG write

    # Camera source (mirrors main app state)
    active_camera:  str  = "picam"

    # Lens metadata (saved to project.json)
    lens: LensProfile = field(default_factory=LensProfile)

    # Storage
    save_path:      str  = "/home/tim/Pictures/PiSlider"


# ─── DERIVED CALCULATIONS ─────────────────────────────────────────────────────

def rail_frame_count(session: MacroSession) -> int:
    """Number of focus steps across the rail range."""
    travel_um = abs(session.rail_end_mm - session.rail_start_mm) * 1000.0
    if session.rail_step_um <= 0:
        return 0
    return max(1, int(math.ceil(travel_um / session.rail_step_um)) + 1)


def rail_step_mm(session: MacroSession) -> float:
    return session.rail_step_um / 1000.0


def rotation_angles(session: MacroSession) -> List[float]:
    """Compute the list of rotation angles for all stacks."""
    n = session.num_stacks
    if n <= 0:
        return []
    if n == 1:
        return [session.rotation_start_deg]

    if session.rotation_mode == "full":
        # Even spacing around full 360° — last point does NOT repeat start
        return [session.rotation_start_deg + i * 360.0 / n for i in range(n)]
    else:
        # Range mode — include both endpoints
        span = session.rotation_end_deg - session.rotation_start_deg
        # Apply easing to the spacing
        weights = normalize(
            CURVE_FUNCTIONS.get(session.rotation_easing,
                                CURVE_FUNCTIONS["even"])(n)
        )
        angles = [session.rotation_start_deg]
        for w in weights[:-1]:
            angles.append(angles[-1] + w * span)
        angles[-1] = session.rotation_end_deg   # clamp last to exact end
        return angles[:n]


def aux_positions(session: MacroSession) -> List[float]:
    """Compute aux axis positions for all stacks (mirrors rotation_angles logic)."""
    n = session.num_stacks
    if not session.aux_enabled or n <= 0:
        return [0.0] * n
    if n == 1:
        return [session.aux_start_deg]
    weights = normalize(
        CURVE_FUNCTIONS.get(session.aux_easing,
                            CURVE_FUNCTIONS["even"])(n)
    )
    span = session.aux_end_deg - session.aux_start_deg
    pos = [session.aux_start_deg]
    for w in weights[:-1]:
        pos.append(pos[-1] + w * span)
    pos[-1] = session.aux_end_deg
    return pos[:n]


def total_image_count(session: MacroSession) -> int:
    enabled_slots = sum(1 for s in session.slots if s.enabled)
    return rail_frame_count(session) * session.num_stacks * max(1, enabled_slots)


def estimated_storage_gb(session: MacroSession, mb_per_frame: float = 25.0) -> float:
    return total_image_count(session) * mb_per_frame / 1024.0


# ─── FOLDER / JSON HELPERS ───────────────────────────────────────────────────

def project_folder(session: MacroSession) -> str:
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    return os.path.join(session.save_path, f"{session.project_name}_{date_str}")


def orbit_folder(proj_folder: str, orbit_label: str) -> str:
    return os.path.join(proj_folder, orbit_label)


def stack_folder(orb_folder: str, stack_idx: int,
                 rot_deg: float, aux_deg: Optional[float] = None) -> str:
    rot_str = f"rot{rot_deg:+08.3f}"
    if aux_deg is not None:
        rot_str += f"_aux{aux_deg:+07.2f}"
    return os.path.join(orb_folder, f"stack_{stack_idx+1:03d}", rot_str)


def slot_folder(stk_folder: str, slot: ExposureSlot) -> str:
    return os.path.join(stk_folder, slot.id)


def _serial(obj):
    """JSON-serialise dataclass or plain dict recursively."""
    if hasattr(obj, '__dataclass_fields__'):
        return {k: _serial(v) for k, v in asdict(obj).items()}
    if isinstance(obj, list):
        return [_serial(i) for i in obj]
    return obj


def write_sequence_json(orb_folder: str, session: MacroSession,
                        stacks_meta: List[Dict]) -> str:
    angles  = rotation_angles(session)
    aux_pos = aux_positions(session)
    doc = {
        "version":       1,
        "created":       datetime.datetime.now().isoformat(),
        "session_mode":  session.session_mode,
        "orbit_label":   session.orbit_label,
        "rotation_axis_angle_deg":        session.rotation_axis_angle_deg,
        "rotation_axis_description":      session.rotation_axis_description,
        "rail": {
            "start_mm":        session.rail_start_mm,
            "end_mm":          session.rail_end_mm,
            "step_um":         session.rail_step_um,
            "step_mm":         rail_step_mm(session),
            "frame_count":     rail_frame_count(session),
            "soft_min_mm":     session.rail_soft_min,
            "soft_max_mm":     session.rail_soft_max,
        },
        "rotation": {
            "mode":            session.rotation_mode,
            "start_deg":       session.rotation_start_deg,
            "end_deg":         session.rotation_end_deg,
            "num_stacks":      session.num_stacks,
            "easing_curve":    session.rotation_easing,
            "angles_deg":      angles,
        },
        "aux_axis": {
            "enabled":         session.aux_enabled,
            "label":           session.aux_label,
            "start_deg":       session.aux_start_deg,
            "end_deg":         session.aux_end_deg,
            "easing_curve":    session.aux_easing,
            "positions_deg":   aux_pos,
        },
        "exposure_slots":  [_serial(s) for s in session.slots],
        "timing": {
            "vibe_delay_s":    session.vibe_delay_s,
            "exp_margin_s":    session.exp_margin_s,
        },
        "total_images":    total_image_count(session),
        "stacks":          stacks_meta,
    }
    path = os.path.join(orb_folder, "sequence.json")
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
    return path


def write_project_json(proj_folder: str, session: MacroSession,
                       orbits_meta: List[Dict]) -> str:
    doc = {
        "version":       1,
        "project_name":  session.project_name,
        "created":       datetime.datetime.now().isoformat(),
        "lens_profile":  _serial(session.lens),
        "rig": {
            "lead_screw_pitch_mm": LEAD_SCREW_PITCH_MM,
            "steps_per_mm":        STEPS_PER_MM,
            "steps_per_deg":       STEPS_PER_DEG,
        },
        "orbits": orbits_meta,
    }
    path = os.path.join(proj_folder, "project.json")
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
    return path


# ─── MACRO ENGINE ─────────────────────────────────────────────────────────────

class MacroEngine:
    """
    Executes a macro focus-stack / 3D-scan sequence.

    Dependencies are injected so this module stays hardware-agnostic
    for unit testing; in production app.py passes the real objects.

    Parameters
    ----------
    hardware        HardwareController instance
    capture_fn      async callable(frame_id: str, slot: ExposureSlot) -> Optional[str]
    apply_camera_fn async callable(slot: ExposureSlot) -> None
    broadcast_fn    async callable(dict) -> None
    """

    def __init__(self, hardware, capture_fn, apply_camera_fn, broadcast_fn):
        self.hw           = hardware
        self._capture     = capture_fn
        self._apply_cam   = apply_camera_fn
        self._broadcast   = broadcast_fn
        self._stop_event  = asyncio.Event()
        self.is_running   = False

        # Runtime state (also used for resume)
        self._session:     Optional[MacroSession] = None
        self._stacks_meta: List[Dict]             = []
        self._proj_folder: str                    = ""
        self._orb_folder:  str                    = ""

        # Soft-limit tracking (live position, updated by jog commands)
        self.rail_pos_mm:  float = 0.0
        self.pan_pos_deg:  float = 0.0
        self.tilt_pos_deg: float = 0.0

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def stop(self):
        self._stop_event.set()

    def get_resume_info(self) -> Optional[Dict]:
        """
        If a partial sequence.json exists at the last orbit folder,
        return enough info for the UI to show a resume prompt.
        """
        if not self._orb_folder:
            return None
        seq_path = os.path.join(self._orb_folder, "sequence.json")
        if not os.path.exists(seq_path):
            return None
        try:
            with open(seq_path) as f:
                doc = json.load(f)
            done  = sum(1 for s in doc.get("stacks", []) if s.get("completed"))
            total = doc.get("rotation", {}).get("num_stacks", 0)
            if done < total:
                return {"done": done, "total": total, "path": seq_path}
        except Exception:
            pass
        return None

    async def run(self, session: MacroSession,
                  resume_from_stack: int = 0) -> None:
        """
        Main entry point.  Call from app.py via asyncio.create_task().

        resume_from_stack: 0 = fresh start, N = skip first N stacks
        """
        self._session    = session
        self._stop_event.clear()
        self.is_running  = True

        # Build folder tree
        self._proj_folder = project_folder(session)
        self._orb_folder  = orbit_folder(self._proj_folder, session.orbit_label)
        os.makedirs(self._orb_folder, exist_ok=True)

        # Compute full angle / aux lists once
        angles  = rotation_angles(session)
        aux_pos = aux_positions(session)

        # Load existing stacks_meta if resuming, else start fresh
        if resume_from_stack > 0:
            try:
                with open(os.path.join(self._orb_folder, "sequence.json")) as f:
                    existing = json.load(f)
                self._stacks_meta = existing.get("stacks", [])
            except Exception:
                self._stacks_meta = []
        else:
            self._stacks_meta = []

        # Write initial sequence.json and project.json
        write_sequence_json(self._orb_folder, session, self._stacks_meta)
        self._update_project_json(completed=False)

        await self._broadcast({
            "type": "macro_progress",
            "stack": resume_from_stack,
            "total_stacks": session.num_stacks,
            "frame": 0,
            "total_frames": rail_frame_count(session),
            "msg": f"Starting macro sequence — {session.num_stacks} stacks × "
                   f"{rail_frame_count(session)} frames × "
                   f"{sum(1 for s in session.slots if s.enabled)} slots"
        })

        try:
            await self._run_sequence(session, angles, aux_pos, resume_from_stack)
        finally:
            self.is_running = False
            self._stop_event.clear()
            # Final writes
            write_sequence_json(self._orb_folder, session, self._stacks_meta)
            self._update_project_json(completed=not self._stop_event.is_set())
            # Safe motor stop
            self.hw.set_tmc_velocity(0, 0)
            self.hw.set_tmc_velocity(1, 0)
            self.hw.set_tmc_velocity(2, 0)
            self.hw.enable_motors(False)
            await self._broadcast({
                "type": "macro_done",
                "interrupted": self._stop_event.is_set(),
                "msg": "Macro sequence complete." if not self._stop_event.is_set()
                       else "Macro sequence stopped by user."
            })

    # ──────────────────────────────────────────────────────────────────────────
    # Internal sequence execution
    # ──────────────────────────────────────────────────────────────────────────

    async def _run_sequence(self, session: MacroSession,
                            angles: List[float], aux_pos: List[float],
                            resume_from: int) -> None:

        frames_per_stack = rail_frame_count(session)
        step_mm          = rail_step_mm(session)
        enabled_slots    = [s for s in session.slots if s.enabled]

        for stack_idx in range(resume_from, session.num_stacks):
            if self._stop_event.is_set():
                break

            rot_deg  = angles[stack_idx]
            a_deg    = aux_pos[stack_idx] if session.aux_enabled else None

            await self._broadcast({
                "type":         "macro_progress",
                "stack":        stack_idx + 1,
                "total_stacks": session.num_stacks,
                "frame":        0,
                "total_frames": frames_per_stack,
                "rotation_deg": rot_deg,
                "msg": f"Stack {stack_idx+1}/{session.num_stacks} — "
                       f"rot {rot_deg:+.1f}°"
            })

            # ── 1. Move rotation stage ────────────────────────────────────────
            await self._move_rotation(rot_deg, a_deg, session)

            # ── 2. Move rail to start ─────────────────────────────────────────
            await self._rail_to(session.rail_start_mm, session, fast=False)

            stack_meta = {
                "stack_id":    f"stack_{stack_idx+1:03d}",
                "rotation_deg": rot_deg,
                "aux_deg":      a_deg,
                "folder":       os.path.relpath(
                    stack_folder(self._orb_folder, stack_idx, rot_deg,
                                 a_deg if session.aux_enabled else None),
                    self._orb_folder),
                "frame_count":  frames_per_stack,
                "completed":    False,
            }

            # ── 3. Step through focus rail ────────────────────────────────────
            direction = 1 if session.rail_end_mm >= session.rail_start_mm else -1
            target_positions = [
                session.rail_start_mm + i * step_mm * direction
                for i in range(frames_per_stack)
            ]
            # Clamp last position exactly to end
            if target_positions:
                target_positions[-1] = session.rail_end_mm

            frame_global_base = stack_idx * frames_per_stack

            for frame_idx, target_mm in enumerate(target_positions):
                if self._stop_event.is_set():
                    break

                # Move rail to this focus position
                await self._rail_to(target_mm, session, fast=False)

                # Anti-vibe settle
                await asyncio.sleep(session.vibe_delay_s)

                # ── 4. Fire each enabled exposure slot ────────────────────────
                for slot in enabled_slots:
                    if self._stop_event.is_set():
                        break

                    frame_id = (f"stack{stack_idx+1:03d}_"
                                f"f{frame_idx+1:04d}_{slot.id}")

                    # Set relay states
                    self.hw.set_relay1(slot.relay1)
                    self.hw.set_relay2(slot.relay2)

                    # Settle for relay
                    if slot.relay_settle_ms > 0:
                        await asyncio.sleep(slot.relay_settle_ms / 1000.0)

                    # Apply camera settings for this slot
                    await self._apply_cam(slot)

                    # Build output path and capture
                    slot_dir = slot_folder(
                        stack_folder(self._orb_folder, stack_idx, rot_deg,
                                     a_deg if session.aux_enabled else None),
                        slot
                    )
                    os.makedirs(slot_dir, exist_ok=True)

                    file_path = await self._capture(
                        slot_dir, frame_id, slot
                    )

                    # Exposure wait (only for non-picam cameras)
                    await asyncio.sleep(
                        max(0.02, slot.shutter_s + session.exp_margin_s)
                    )

                    # Release relay + post-settle
                    if slot.relay_release_ms > 0:
                        await asyncio.sleep(slot.relay_release_ms / 1000.0)
                    self.hw.set_relay1(False)
                    self.hw.set_relay2(False)

                    await self._broadcast({
                        "type":         "macro_progress",
                        "stack":        stack_idx + 1,
                        "total_stacks": session.num_stacks,
                        "frame":        frame_idx + 1,
                        "total_frames": frames_per_stack,
                        "slot":         slot.id,
                        "rotation_deg": rot_deg,
                        "rail_mm":      target_mm,
                        "msg": f"Stack {stack_idx+1}/{session.num_stacks}  "
                               f"Frame {frame_idx+1}/{frames_per_stack}  "
                               f"[{slot.label}]"
                    })

            # ── 5. Return rail to start at high speed ─────────────────────────
            if not self._stop_event.is_set():
                await self._rail_to(session.rail_start_mm, session, fast=True)

            # ── 6. Mark stack complete ────────────────────────────────────────
            stack_meta["completed"] = not self._stop_event.is_set()
            # Update or append
            if stack_idx < len(self._stacks_meta):
                self._stacks_meta[stack_idx] = stack_meta
            else:
                self._stacks_meta.append(stack_meta)

            # Incremental save after every stack
            write_sequence_json(self._orb_folder, self._session, self._stacks_meta)

            await self._broadcast({
                "type":         "macro_stack_complete",
                "stack":        stack_idx + 1,
                "total_stacks": session.num_stacks,
                "rotation_deg": rot_deg,
                "completed":    stack_meta["completed"],
            })

    # ──────────────────────────────────────────────────────────────────────────
    # Motor helpers
    # ──────────────────────────────────────────────────────────────────────────

    async def _rail_to(self, target_mm: float, session: MacroSession,
                       fast: bool = False) -> None:
        """Move rail to absolute position, respecting soft limits."""
        target_mm = max(session.rail_soft_min,
                        min(session.rail_soft_max, target_mm))
        delta_mm  = target_mm - self.rail_pos_mm
        if abs(delta_mm) < 0.001:
            return

        steps = int(delta_mm * STEPS_PER_MM)
        if steps == 0:
            return

        self.hw.enable_motors(True)

        if fast:
            # High-speed return via VACTUAL — direction from sign, then stop
            vel = RAIL_RETURN_VELOCITY if steps > 0 else -RAIL_RETURN_VELOCITY
            self.hw.set_tmc_velocity(0, vel)
            # Wait based on distance at max speed (rough estimate)
            est_time = abs(delta_mm) / (RAIL_RETURN_VELOCITY / STEPS_PER_MM) * 0.9
            wait_step = 0.05
            elapsed   = 0.0
            while elapsed < est_time and not self._stop_event.is_set():
                await asyncio.sleep(wait_step)
                elapsed += wait_step
            self.hw.set_tmc_velocity(0, 0)
        else:
            # Precision move via Bresenham stepper
            duration = max(0.2, abs(delta_mm) * 0.15)   # ~150ms per mm
            await asyncio.to_thread(
                self.hw.move_axes_simultaneous,
                steps, 0, 0, duration
            )

        self.rail_pos_mm = target_mm
        self.hw.enable_motors(False)

    async def _move_rotation(self, rot_deg: float, aux_deg: Optional[float],
                             session: MacroSession) -> None:
        """Move pan (and optionally tilt/aux) to target angles."""
        delta_pan  = rot_deg  - self.pan_pos_deg
        delta_aux  = (aux_deg - self.tilt_pos_deg) if aux_deg is not None else 0.0

        pan_steps  = int(delta_pan * STEPS_PER_DEG)
        tilt_steps = int(delta_aux * STEPS_PER_DEG)

        if pan_steps == 0 and tilt_steps == 0:
            return

        # Clamp to soft limits
        new_pan  = self.pan_pos_deg + delta_pan
        new_tilt = self.tilt_pos_deg + delta_aux

        duration = max(0.3, abs(delta_pan) * 0.02 + abs(delta_aux) * 0.02)

        self.hw.enable_motors(True)
        await asyncio.to_thread(
            self.hw.move_axes_simultaneous,
            0, pan_steps, tilt_steps, duration
        )
        self.hw.enable_motors(False)

        self.pan_pos_deg  = new_pan
        self.tilt_pos_deg = new_tilt

        await asyncio.sleep(session.vibe_delay_s)

    # ──────────────────────────────────────────────────────────────────────────
    # Project JSON maintenance
    # ──────────────────────────────────────────────────────────────────────────

    def _update_project_json(self, completed: bool) -> None:
        if not self._session or not self._proj_folder:
            return
        done = sum(1 for s in self._stacks_meta if s.get("completed"))
        orbit_entry = {
            "orbit_id":   self._session.orbit_label,
            "label":      self._session.orbit_label,
            "rotation_axis_angle_deg": self._session.rotation_axis_angle_deg,
            "rotation_axis_description": self._session.rotation_axis_description,
            "folder":     os.path.basename(self._orb_folder),
            "num_stacks": self._session.num_stacks,
            "stacks_done": done,
            "completed":  completed,
        }

        # Load existing project.json if present, update orbits list
        proj_path = os.path.join(self._proj_folder, "project.json")
        orbits = []
        if os.path.exists(proj_path):
            try:
                with open(proj_path) as f:
                    existing = json.load(f)
                orbits = existing.get("orbits", [])
            except Exception:
                pass

        # Replace or append this orbit's entry
        found = False
        for i, o in enumerate(orbits):
            if o.get("orbit_id") == orbit_entry["orbit_id"]:
                orbits[i] = orbit_entry
                found = True
                break
        if not found:
            orbits.append(orbit_entry)

        write_project_json(self._proj_folder, self._session, orbits)
