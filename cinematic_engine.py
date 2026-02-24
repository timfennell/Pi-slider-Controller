#!/usr/bin/env python3
"""
cinematic_engine.py — Cinema motion control engine for PiSlider.

Four main classes:
  SoftLimitGuard    — per-axis travel limits with velocity-aware decel ramp.
                      Blocks high-speed movement until both ends calibrated.
  InertiaEngine     — physics simulation: mass + fluid drag model at 50 Hz.
                      Consumes gamepad axis events, outputs TMC2209 velocities.
  ArcTanTracker     — 3D least-squares subject solve from N calibration points.
                      Outputs real-time (pan_deg, tilt_deg) for any slider pos.
  ProgrammedMove    — keyframe store with per-segment timing + cubic spline
                      playback via existing TrajectoryPlayer infrastructure.

MoveLibrary        — named keyframe sequence persistence (~/.pislider_moves.json).
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Tuple, Any

import numpy as np

logger = logging.getLogger("PiSlider.Cinematic")

# ─── HARDWARE CONSTANTS ───────────────────────────────────────────────────────
STEPS_PER_MM  = 100.0
STEPS_PER_DEG = 55.5

# Maximum TMC2209 VACTUAL values (steps/s) — physical limits
MAX_VEL_SLIDER = 80000   # ~800 mm/s
MAX_VEL_PAN    = 40000   # ~720 deg/s
MAX_VEL_TILT   = 40000

# Maximum deceleration (steps/s²) — chosen to never lose steps
MAX_DECEL_SLIDER = 120000
MAX_DECEL_PAN    =  60000
MAX_DECEL_TILT   =  60000

# Crawl speed (used before soft limits are calibrated)
CRAWL_VEL_SLIDER = 4000   # ~40 mm/s
CRAWL_VEL_PAN    = 2000
CRAWL_VEL_TILT   = 2000

# Inertia loop rate
INERTIA_HZ = 50
INERTIA_DT = 1.0 / INERTIA_HZ

MOVES_FILE = os.path.expanduser("~/.pislider_moves.json")


# ─── SOFT LIMIT GUARD ────────────────────────────────────────────────────────

class AxisLimit:
    """Per-axis soft limit state and deceleration guard."""

    CAL_NONE  = 0   # uncalibrated — crawl only
    CAL_ONE   = 1   # one end calibrated — half speed
    CAL_BOTH  = 2   # both ends calibrated — full speed

    def __init__(self, name: str, max_vel: int, max_decel: int,
                 crawl_vel: int, steps_per_unit: float):
        self.name           = name
        self.max_vel        = max_vel
        self.max_decel      = max_decel
        self.crawl_vel      = crawl_vel
        self.steps_per_unit = steps_per_unit

        self.min_unit: Optional[float] = None   # in mm or deg
        self.max_unit: Optional[float] = None
        self.cal_state: int = self.CAL_NONE
        self.current_vel: float = 0.0           # steps/s (signed)

    @property
    def speed_limit(self) -> int:
        """Effective max velocity given calibration state."""
        if self.cal_state == self.CAL_NONE:
            return self.crawl_vel
        if self.cal_state == self.CAL_ONE:
            return self.max_vel // 2
        return self.max_vel

    def set_min(self, unit: float):
        self.min_unit = unit
        self._update_cal()

    def set_max(self, unit: float):
        self.max_unit = unit
        self._update_cal()

    def _update_cal(self):
        both = self.min_unit is not None and self.max_unit is not None
        one  = self.min_unit is not None or  self.max_unit is not None
        self.cal_state = self.CAL_BOTH if both else (self.CAL_ONE if one else self.CAL_NONE)

    def clamp_velocity(self, desired_vel: int, current_pos_unit: float) -> int:
        """
        Return safe velocity respecting:
        1. Speed limit from calibration state
        2. Deceleration ramp when approaching soft limits
        3. Hard clamp at limit boundaries
        """
        # Speed cap
        limit = self.speed_limit
        vel = max(-limit, min(limit, desired_vel))

        # Deceleration ramp toward limits
        if vel != 0 and self.cal_state == self.CAL_BOTH:
            vel = self._apply_ramp(vel, current_pos_unit)

        return int(vel)

    def _apply_ramp(self, vel: int, pos: float) -> int:
        """
        Reduce velocity if stopping distance would exceed gap to limit.
        stopping_distance = v² / (2 * max_decel)  [in steps]
        """
        if vel > 0 and self.max_unit is not None:
            gap_units = self.max_unit - pos
            gap_steps = gap_units * self.steps_per_unit
            stop_dist = (vel ** 2) / (2.0 * self.max_decel)
            if stop_dist >= gap_steps > 0:
                # Scale velocity proportionally
                safe_vel = math.sqrt(max(0, 2.0 * self.max_decel * gap_steps))
                vel = min(vel, int(safe_vel))

        elif vel < 0 and self.min_unit is not None:
            gap_units = pos - self.min_unit
            gap_steps = gap_units * self.steps_per_unit
            stop_dist = (vel ** 2) / (2.0 * self.max_decel)
            if stop_dist >= gap_steps > 0:
                safe_vel = math.sqrt(max(0, 2.0 * self.max_decel * gap_steps))
                vel = max(vel, -int(safe_vel))

        return vel


class SoftLimitGuard:
    """Wraps all three axes with AxisLimit instances."""

    def __init__(self):
        self.slider = AxisLimit("slider", MAX_VEL_SLIDER, MAX_DECEL_SLIDER,
                                CRAWL_VEL_SLIDER, STEPS_PER_MM)
        self.pan    = AxisLimit("pan",    MAX_VEL_PAN,    MAX_DECEL_PAN,
                                CRAWL_VEL_PAN,    STEPS_PER_DEG)
        self.tilt   = AxisLimit("tilt",   MAX_VEL_TILT,   MAX_DECEL_TILT,
                                CRAWL_VEL_TILT,   STEPS_PER_DEG)

    def clamp(self, v_slider: int, v_pan: int, v_tilt: int,
              pos_slider: float, pos_pan: float, pos_tilt: float
              ) -> Tuple[int, int, int]:
        return (
            self.slider.clamp_velocity(v_slider, pos_slider),
            self.pan.clamp_velocity(v_pan,       pos_pan),
            self.tilt.clamp_velocity(v_tilt,     pos_tilt),
        )

    def status(self) -> Dict:
        def _ax(ax: AxisLimit):
            return {
                "cal_state": ax.cal_state,
                "min":       ax.min_unit,
                "max":       ax.max_unit,
                "speed_pct": int(ax.speed_limit / ax.max_vel * 100),
            }
        return {
            "slider": _ax(self.slider),
            "pan":    _ax(self.pan),
            "tilt":   _ax(self.tilt),
        }


# ─── INERTIA ENGINE ───────────────────────────────────────────────────────────

# Preset rigs (mass, drag)
RIG_PRESETS = {
    "light":    (0.15, 0.80),   # GoPro / mirrorless on light head
    "standard": (0.40, 0.55),   # Standard mirrorless + lens
    "heavy":    (0.90, 0.30),   # Cinema camera on fluid head
    "custom":   None,
}


class InertiaEngine:
    """
    Physics simulation for live camera movement.

    Model:
        acceleration = (stick_force - drag * velocity) / mass
        velocity    += acceleration * dt
        position    += velocity * dt

    stick_force is proportional to stick deflection (non-linear: cube root
    gives fine control near center, full speed at extremes).

    Both mass and drag are user-adjustable. Three presets provided.
    """

    def __init__(self, hardware, guard: SoftLimitGuard, broadcast_fn,
                 slider_axis, pan_axis, tilt_axis):
        self.hw         = hardware
        self.guard      = guard
        self.broadcast  = broadcast_fn
        self._slider    = slider_axis
        self._pan       = pan_axis
        self._tilt      = tilt_axis

        # Physics state (steps/s)
        self._v_slider: float = 0.0
        self._v_pan:    float = 0.0
        self._v_tilt:   float = 0.0

        # Target velocities from gamepad (normalized [-1, 1])
        self._t_slider: float = 0.0
        self._t_pan:    float = 0.0
        self._t_tilt:   float = 0.0

        # Physics params
        self.mass: float = 0.40   # seconds to reach full speed
        self.drag: float = 0.55   # viscosity coefficient

        # Speed multiplier from L1/R1 buttons
        self._speed_mult: float = 1.0
        self._l1_held: bool = False
        self._r1_held: bool = False

        # Arctan lock
        self.arctan_active: bool = False
        self._tracker: Optional[ArcTanTracker] = None

        self._running: bool = False
        self._task: Optional[asyncio.Task] = None

    def set_preset(self, name: str):
        if name in RIG_PRESETS and RIG_PRESETS[name]:
            self.mass, self.drag = RIG_PRESETS[name]
            logger.info(f"Inertia preset: {name} (mass={self.mass}, drag={self.drag})")

    def set_params(self, mass: float, drag: float):
        self.mass = max(0.05, min(3.0, mass))
        self.drag = max(0.05, min(1.50, drag))

    def set_target(self, slider: float, pan: float, tilt: float):
        """Called from gamepad event handler with normalized [-1, 1] stick values."""
        self._t_slider = slider
        self._t_pan    = pan
        self._t_tilt   = tilt

    def set_speed_modifier(self, l1: bool, r1: bool):
        self._l1_held = l1
        self._r1_held = r1
        if l1:
            self._speed_mult = 0.25
        elif r1:
            self._speed_mult = 2.0
        else:
            self._speed_mult = 1.0

    def set_arctan_tracker(self, tracker: Optional["ArcTanTracker"]):
        self._tracker = tracker

    def start(self):
        if not self._running:
            self._running = True
            self._task = asyncio.create_task(self._loop())

    def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
        self.hw.set_tmc_velocity(0, 0)
        self.hw.set_tmc_velocity(1, 0)
        self.hw.set_tmc_velocity(2, 0)

    async def _loop(self):
        """50 Hz physics + motor command loop."""
        self.hw.enable_motors(True)
        try:
            while self._running:
                t0 = asyncio.get_event_loop().time()
                self._tick()
                elapsed = asyncio.get_event_loop().time() - t0
                await asyncio.sleep(max(0, INERTIA_DT - elapsed))
        finally:
            self.hw.set_tmc_velocity(0, 0)
            self.hw.set_tmc_velocity(1, 0)
            self.hw.set_tmc_velocity(2, 0)
            self.hw.enable_motors(False)

    def _tick(self):
        """One physics integration step."""
        # Non-linear stick mapping: cube-root gives fine control near centre
        def _map(t: float, max_v: float) -> float:
            sign = 1 if t >= 0 else -1
            return sign * (abs(t) ** 0.6) * max_v * self._speed_mult

        # Target velocities in steps/s
        target_slider = _map(self._t_slider, self.guard.slider.speed_limit)
        target_pan    = _map(self._t_pan,    self.guard.pan.speed_limit)
        target_tilt   = _map(self._t_tilt,   self.guard.tilt.speed_limit)

        # Physics: F = (target - drag*v) / mass
        dt = INERTIA_DT
        inv_mass = 1.0 / max(0.01, self.mass)

        self._v_slider += (target_slider - self.drag * self._v_slider) * inv_mass * dt
        self._v_pan    += (target_pan    - self.drag * self._v_pan)    * inv_mass * dt
        self._v_tilt   += (target_tilt   - self.drag * self._v_tilt)   * inv_mass * dt

        # Integrate position (update axis trackers)
        self._slider.current_mm  += self._v_slider * dt / STEPS_PER_MM
        self._pan.current_deg    += self._v_pan    * dt / STEPS_PER_DEG
        self._tilt.current_deg   += self._v_tilt   * dt / STEPS_PER_DEG

        # Arctan override: if locked, recompute pan+tilt from slider position
        if self.arctan_active and self._tracker and self._tracker.is_solved:
            pan_t, tilt_t = self._tracker.get_angles(self._slider.current_mm)
            # Drive pan/tilt toward computed angles (fast, 1-tick convergence at cinematic speeds)
            self._pan.current_deg  = pan_t
            self._tilt.current_deg = tilt_t
            # Override pan/tilt velocities to match the required motion
            self._v_pan  = (pan_t  - (self._pan.current_deg  - self._v_pan  * dt / STEPS_PER_DEG)) * STEPS_PER_DEG / dt
            self._v_tilt = (tilt_t - (self._tilt.current_deg - self._v_tilt * dt / STEPS_PER_DEG)) * STEPS_PER_DEG / dt

        # Soft limit clamping
        vs, vp, vt = self.guard.clamp(
            int(self._v_slider), int(self._v_pan), int(self._v_tilt),
            self._slider.current_mm, self._pan.current_deg, self._tilt.current_deg
        )

        # Send to hardware
        self.hw.set_tmc_velocity(0, vs)
        self.hw.set_tmc_velocity(1, vp)
        self.hw.set_tmc_velocity(2, vt)

        # Update velocities with clamped values for next tick's physics
        self._v_slider = float(vs)
        self._v_pan    = float(vp)
        self._v_tilt   = float(vt)


# ─── ARCTAN TRACKER ──────────────────────────────────────────────────────────

@dataclass
class CalibPoint:
    """One calibration measurement: gantry position + where camera pointed."""
    slider_mm:  float
    pan_deg:    float
    tilt_deg:   float


class ArcTanTracker:
    """
    3D subject position solver using N calibration points.

    The user directly measures (pan_deg, tilt_deg) pointing at the subject
    from each slider position. Rail tilt is irrelevant — it's already encoded
    in the tilt readings. We just need to find the 3D point that best explains
    all the measured ray directions.

    Model:
      Each calibration point gives a unit ray in camera-local space:
        ray = [cos(tilt)*cos(pan), cos(tilt)*sin(pan), sin(tilt)]

      Camera position along rail (1D): x = slider_mm (horizontal component
      only needed for parallax; the ball head stays level so pan/tilt are
      world-space regardless of rail angle).

      We solve for subject (X, Y, Z) in world mm such that for each point i:
        pan_i  = atan2(Y - 0,        X - slider_mm_i)   [horizontal plane]
        tilt_i = atan2(Z,            sqrt((X-s_i)^2 + Y^2))

      This is a straightforward nonlinear least-squares in 3 unknowns.
    """

    WARN_RESIDUAL_DEG = 1.5
    MIN_POINTS = 3

    def __init__(self):
        self.points: List[CalibPoint] = []
        self.subject: Optional[np.ndarray] = None   # [X, Y, Z] world mm
        self.residual_deg: float = 0.0
        self.is_solved: bool = False
        self.warning: str = ""

    def add_point(self, slider_mm: float, pan_deg: float, tilt_deg: float):
        self.points.append(CalibPoint(slider_mm, pan_deg, tilt_deg))
        logger.info(f"ArcTan: added point {len(self.points)}: "
                    f"s={slider_mm:.1f}mm pan={pan_deg:.2f}° tilt={tilt_deg:.2f}°")
        if len(self.points) >= self.MIN_POINTS:
            self.solve()

    def clear_points(self):
        self.points = []
        self.subject = None
        self.is_solved = False
        self.warning = ""

    def solve(self) -> bool:
        """
        Least-squares solve for subject [X, Y, Z].
        No rail tilt needed — pan/tilt measurements are already in world space
        because the ball head keeps the motors level.
        """
        if len(self.points) < self.MIN_POINTS:
            return False

        # Camera X positions (1D along rail horizontal projection)
        cam_xs  = np.array([p.slider_mm  for p in self.points])
        pan_r   = np.array([math.radians(p.pan_deg)  for p in self.points])
        tilt_r  = np.array([math.radians(p.tilt_deg) for p in self.points])

        # Initial guess: subject 500mm ahead of midpoint, same height
        subj = np.array([float(np.mean(cam_xs)) + 500.0, 500.0, 0.0])

        for _ in range(40):
            dx    = subj[0] - cam_xs
            dy    = np.full_like(dx, subj[1])
            dz    = np.full_like(dx, subj[2])
            horiz = np.sqrt(dx**2 + dy**2)

            pred_pan  = np.arctan2(dy, dx)
            pred_tilt = np.arctan2(dz, horiz)

            res_pan  = pan_r  - pred_pan
            res_tilt = tilt_r - pred_tilt

            r2 = dx**2 + dy**2
            r3 = horiz**2 + dz**2

            J = np.zeros((2 * len(self.points), 3))
            J[0::2, 0] = -dy / r2
            J[0::2, 1] =  dx / r2
            J[0::2, 2] =  0
            J[1::2, 0] = -dx * dz / (horiz * r3)
            J[1::2, 1] = -dy * dz / (horiz * r3)
            J[1::2, 2] =  horiz / r3

            residuals       = np.empty(2 * len(self.points))
            residuals[0::2] = res_pan
            residuals[1::2] = res_tilt

            try:
                delta, _, _, _ = np.linalg.lstsq(J, residuals, rcond=None)
            except np.linalg.LinAlgError:
                break
            subj += delta
            if np.linalg.norm(delta) < 1e-4:
                break

        # RMS residual
        dx    = subj[0] - cam_xs
        dy    = np.full_like(dx, subj[1])
        dz    = np.full_like(dx, subj[2])
        horiz = np.sqrt(dx**2 + dy**2)
        pred_pan  = np.degrees(np.arctan2(dy, dx))
        pred_tilt = np.degrees(np.arctan2(dz, horiz))
        meas_pan  = np.array([p.pan_deg  for p in self.points])
        meas_tilt = np.array([p.tilt_deg for p in self.points])
        rms = float(np.sqrt(np.mean(
            (pred_pan - meas_pan)**2 + (pred_tilt - meas_tilt)**2
        )))

        self.subject      = subj
        self.residual_deg = rms
        self.is_solved    = True

        slider_span = max(p.slider_mm for p in self.points) - min(p.slider_mm for p in self.points)

        self.warning = ""
        if rms > self.WARN_RESIDUAL_DEG:
            self.warning = (f"High residual ({rms:.2f}°) — re-aim more carefully "
                            f"or add more calibration points.")
        elif slider_span < 50:
            self.warning = ("Points too close together — spread them further "
                            "along the rail for accuracy.")

        logger.info(f"ArcTan solved: subject={subj.round(1)}, RMS={rms:.3f}° "
                    f"warning='{self.warning}'")
        return True

    def get_angles(self, slider_mm: float) -> Tuple[float, float]:
        """Return (pan_deg, tilt_deg) for a given slider position."""
        if not self.is_solved or self.subject is None:
            return 0.0, 0.0
        sx, sy, sz = self.subject
        dx    = sx - slider_mm
        dy    = sy
        dz    = sz
        horiz = math.sqrt(dx**2 + dy**2)
        pan   = math.degrees(math.atan2(dy, dx))
        tilt  = math.degrees(math.atan2(dz, horiz))
        return pan, tilt


# ─── PROGRAMMED MOVE ─────────────────────────────────────────────────────────

@dataclass
class Keyframe:
    slider_mm: float
    pan_deg:   float
    tilt_deg:  float
    duration_s: float = 3.0          # time to reach NEXT keyframe
    easing:     str   = "gaussian"   # from distributions.CURVE_FUNCTIONS


class ProgrammedMove:
    """
    Manages a list of keyframes and plays them back via TrajectoryPlayer.

    Segment model: each keyframe specifies the duration and easing to reach
    the NEXT keyframe. The last keyframe's duration/easing are ignored.

    Origin system: positions stored as ABSOLUTE (mm/deg). At playback,
    they are offset by the session origin so the move plays relative to
    wherever the user parked the rig.
    """

    def __init__(self, hardware, guard: SoftLimitGuard,
                 slider_axis, pan_axis, tilt_axis,
                 broadcast_fn, arctan_tracker: Optional[ArcTanTracker] = None):
        self.hw         = hardware
        self.guard      = guard
        self._slider    = slider_axis
        self._pan       = pan_axis
        self._tilt      = tilt_axis
        self.broadcast  = broadcast_fn
        self.tracker    = arctan_tracker

        self.keyframes: List[Keyframe] = []
        self.preroll_s: float = 3.0
        self.loop: bool = False

        # Origin offset (set by user "Set Origin" action)
        self.origin_slider: float = 0.0
        self.origin_pan:    float = 0.0
        self.origin_tilt:   float = 0.0

        self._running: bool = False
        self._stop_event    = asyncio.Event()

    def set_origin(self, slider_mm: float, pan_deg: float, tilt_deg: float):
        self.origin_slider = slider_mm
        self.origin_pan    = pan_deg
        self.origin_tilt   = tilt_deg
        logger.info(f"Origin set: s={slider_mm:.2f}mm p={pan_deg:.2f}° t={tilt_deg:.2f}°")

    def add_keyframe(self, slider_mm: float, pan_deg: float, tilt_deg: float,
                     duration_s: float = 3.0, easing: str = "gaussian") -> int:
        kf = Keyframe(slider_mm, pan_deg, tilt_deg, duration_s, easing)
        self.keyframes.append(kf)
        return len(self.keyframes) - 1

    def update_keyframe(self, index: int, **kwargs):
        if 0 <= index < len(self.keyframes):
            for k, v in kwargs.items():
                setattr(self.keyframes[index], k, v)

    def remove_keyframe(self, index: int):
        if 0 <= index < len(self.keyframes):
            self.keyframes.pop(index)

    def clear_keyframes(self):
        self.keyframes = []

    def stop(self):
        self._stop_event.set()

    async def return_to_start(self, speed_fraction: float = 0.5):
        """Move rig to first keyframe (world position) at reduced speed."""
        if not self.keyframes:
            return
        kf0 = self.keyframes[0]
        target_s = kf0.slider_mm + self.origin_slider
        target_p = kf0.pan_deg   + self.origin_pan
        target_t = kf0.tilt_deg  + self.origin_tilt

        await self.broadcast({"type": "cinematic_status",
                               "msg": "Returning to start position…"})
        await self._move_to(target_s, target_p, target_t,
                            duration_s=max(2.0, kf0.duration_s * 0.5))

    async def play(self):
        """Execute the programmed move once (or loop until stopped)."""
        if len(self.keyframes) < 2:
            await self.broadcast({"type": "cinematic_status",
                                   "msg": "Need at least 2 keyframes to play."})
            return

        self._stop_event.clear()
        self._running = True

        try:
            take = 0
            while not self._stop_event.is_set():
                take += 1
                await self.broadcast({"type": "cinematic_status",
                                       "msg": f"Pre-roll {self.preroll_s:.1f}s… (Take {take})"})

                # Pre-roll delay
                for _ in range(int(self.preroll_s * 10)):
                    if self._stop_event.is_set():
                        break
                    await asyncio.sleep(0.1)

                if self._stop_event.is_set():
                    break

                await self.broadcast({"type": "cinematic_play_start", "take": take})
                await self._execute_move()

                if not self.loop or self._stop_event.is_set():
                    break

                # Return to start for next take
                await self.return_to_start()

        finally:
            self._running = False
            self.hw.set_tmc_velocity(0, 0)
            self.hw.set_tmc_velocity(1, 0)
            self.hw.set_tmc_velocity(2, 0)
            self.hw.enable_motors(False)
            await self.broadcast({"type": "cinematic_play_done"})

    async def _execute_move(self):
        """
        Play segments one by one. Each segment interpolates from keyframe[i]
        to keyframe[i+1] using its own duration and easing curve.
        """
        from scipy.interpolate import CubicSpline
        from distributions import CURVE_FUNCTIONS, normalize

        FPS = 60
        self.hw.enable_motors(True)

        for i in range(len(self.keyframes) - 1):
            if self._stop_event.is_set():
                break

            kf_a = self.keyframes[i]
            kf_b = self.keyframes[i + 1]
            dur  = max(0.1, kf_a.duration_s)
            n_frames = max(2, int(dur * FPS))

            # Easing time array
            curve_fn = CURVE_FUNCTIONS.get(kf_a.easing, CURVE_FUNCTIONS["gaussian"])
            weights  = normalize(curve_fn(n_frames))
            t_arr    = np.cumsum(weights)
            t_arr    = np.insert(t_arr, 0, 0.0)[:-1]
            t_arr[-1] = 1.0

            # World-space targets (absolute + origin offset)
            s_a = kf_a.slider_mm + self.origin_slider
            s_b = kf_b.slider_mm + self.origin_slider
            p_a = kf_a.pan_deg   + self.origin_pan
            p_b = kf_b.pan_deg   + self.origin_pan
            t_a = kf_a.tilt_deg  + self.origin_tilt
            t_b = kf_b.tilt_deg  + self.origin_tilt

            # Linear interpolation within segment (CubicSpline per full move
            # is applied at the multi-keyframe level if user chains them)
            traj_s = s_a + t_arr * (s_b - s_a)
            traj_p = p_a + t_arr * (p_b - p_a)
            traj_t = t_a + t_arr * (t_b - t_a)

            # If arctan active, override pan+tilt for each slider position
            if self.tracker and self.tracker.is_solved:
                for j, s_pos in enumerate(traj_s):
                    pan_t, tilt_t = self.tracker.get_angles(s_pos)
                    traj_p[j] = pan_t
                    traj_t[j] = tilt_t

            # Stream velocity commands at FPS
            dt = 1.0 / FPS
            start_t = asyncio.get_event_loop().time()

            for frame in range(n_frames - 1):
                if self._stop_event.is_set():
                    break

                delta_s = traj_s[frame + 1] - traj_s[frame]
                delta_p = traj_p[frame + 1] - traj_p[frame]
                delta_t = traj_t[frame + 1] - traj_t[frame]

                v_s = int(delta_s / dt * STEPS_PER_MM)
                v_p = int(delta_p / dt * STEPS_PER_DEG)
                v_t = int(delta_t / dt * STEPS_PER_DEG)

                # Update position trackers
                self._slider.current_mm  = traj_s[frame + 1]
                self._pan.current_deg    = traj_p[frame + 1]
                self._tilt.current_deg   = traj_t[frame + 1]

                # Guard clamp
                v_s, v_p, v_t = self.guard.clamp(
                    v_s, v_p, v_t,
                    self._slider.current_mm,
                    self._pan.current_deg,
                    self._tilt.current_deg,
                )

                self.hw.set_tmc_velocity(0, v_s)
                self.hw.set_tmc_velocity(1, v_p)
                self.hw.set_tmc_velocity(2, v_t)

                # Progress broadcast every 30 frames (~0.5s)
                if frame % 30 == 0:
                    progress = (i + t_arr[frame]) / (len(self.keyframes) - 1)
                    await self.broadcast({
                        "type":     "cinematic_progress",
                        "segment":  i + 1,
                        "segments": len(self.keyframes) - 1,
                        "progress": round(progress, 3),
                        "pos_s":    round(self._slider.current_mm, 2),
                        "pos_p":    round(self._pan.current_deg, 2),
                        "pos_t":    round(self._tilt.current_deg, 2),
                    })

                # Precision timing
                next_t = start_t + (frame + 1) * dt
                sleep  = next_t - asyncio.get_event_loop().time() - 0.001
                if sleep > 0:
                    await asyncio.sleep(sleep)
                while asyncio.get_event_loop().time() < next_t:
                    pass

        self.hw.set_tmc_velocity(0, 0)
        self.hw.set_tmc_velocity(1, 0)
        self.hw.set_tmc_velocity(2, 0)

    async def _move_to(self, target_s: float, target_p: float, target_t: float,
                       duration_s: float = 3.0):
        """Simple smooth move to absolute position."""
        FPS   = 60
        n     = max(2, int(duration_s * FPS))
        dt    = 1.0 / FPS

        cur_s = self._slider.current_mm
        cur_p = self._pan.current_deg
        cur_t = self._tilt.current_deg

        # Smootherstep for the return move
        def _ss(x):
            t = max(0.0, min(1.0, x))
            return t * t * t * (t * (6 * t - 15) + 10)

        self.hw.enable_motors(True)
        start_t = asyncio.get_event_loop().time()

        for frame in range(n - 1):
            if self._stop_event.is_set():
                break
            frac = _ss(frame / (n - 1))
            next_frac = _ss((frame + 1) / (n - 1))

            s_now = cur_s + frac      * (target_s - cur_s)
            s_nxt = cur_s + next_frac * (target_s - cur_s)
            p_now = cur_p + frac      * (target_p - cur_p)
            p_nxt = cur_p + next_frac * (target_p - cur_p)
            t_now = cur_t + frac      * (target_t - cur_t)
            t_nxt = cur_t + next_frac * (target_t - cur_t)

            v_s = int((s_nxt - s_now) / dt * STEPS_PER_MM)
            v_p = int((p_nxt - p_now) / dt * STEPS_PER_DEG)
            v_t = int((t_nxt - t_now) / dt * STEPS_PER_DEG)

            self._slider.current_mm = s_nxt
            self._pan.current_deg   = p_nxt
            self._tilt.current_deg  = t_nxt

            v_s, v_p, v_t = self.guard.clamp(v_s, v_p, v_t, s_nxt, p_nxt, t_nxt)
            self.hw.set_tmc_velocity(0, v_s)
            self.hw.set_tmc_velocity(1, v_p)
            self.hw.set_tmc_velocity(2, v_t)

            next_t = start_t + (frame + 1) * dt
            sleep  = next_t - asyncio.get_event_loop().time() - 0.001
            if sleep > 0:
                await asyncio.sleep(sleep)

        self.hw.set_tmc_velocity(0, 0)
        self.hw.set_tmc_velocity(1, 0)
        self.hw.set_tmc_velocity(2, 0)
        self.hw.enable_motors(False)


# ─── MOVE LIBRARY ─────────────────────────────────────────────────────────────

@dataclass
class SavedMove:
    name:        str
    created:     str
    notes:       str
    total_duration_s: float
    extents:     Dict   # slider_range_mm, pan_range_deg, tilt_range_deg
    keyframes:   List[Dict]   # serialised Keyframe dicts


class MoveLibrary:
    """
    Named keyframe sequence persistence.

    Moves are stored absolute (mm/deg). At playback, the session origin
    is applied as an offset so the move is always relative to the user's
    parked home position.
    """

    def __init__(self, path: str = MOVES_FILE):
        self._path = path
        self._moves: Dict[str, SavedMove] = {}
        self._load()

    def _load(self):
        if os.path.exists(self._path):
            try:
                with open(self._path) as f:
                    raw = json.load(f)
                for name, data in raw.items():
                    self._moves[name] = SavedMove(**data)
                logger.info(f"MoveLibrary: loaded {len(self._moves)} moves from {self._path}")
            except Exception as e:
                logger.warning(f"MoveLibrary load failed: {e}")

    def _save(self):
        try:
            with open(self._path, "w") as f:
                json.dump({n: asdict(m) for n, m in self._moves.items()}, f, indent=2)
        except Exception as e:
            logger.error(f"MoveLibrary save failed: {e}")

    def save_move(self, name: str, keyframes: List[Keyframe], notes: str = "") -> SavedMove:
        import datetime
        if not keyframes:
            raise ValueError("Cannot save empty keyframe list.")

        total_dur = sum(kf.duration_s for kf in keyframes[:-1])
        s_vals = [kf.slider_mm for kf in keyframes]
        p_vals = [kf.pan_deg   for kf in keyframes]
        t_vals = [kf.tilt_deg  for kf in keyframes]

        move = SavedMove(
            name     = name,
            created  = datetime.datetime.now().isoformat(),
            notes    = notes,
            total_duration_s = round(total_dur, 2),
            extents  = {
                "slider_min_mm":  round(min(s_vals), 2),
                "slider_max_mm":  round(max(s_vals), 2),
                "pan_min_deg":    round(min(p_vals), 2),
                "pan_max_deg":    round(max(p_vals), 2),
                "tilt_min_deg":   round(min(t_vals), 2),
                "tilt_max_deg":   round(max(t_vals), 2),
            },
            keyframes = [asdict(kf) for kf in keyframes],
        )
        self._moves[name] = move
        self._save()
        logger.info(f"MoveLibrary: saved '{name}' ({len(keyframes)} keyframes, {total_dur:.1f}s)")
        return move

    def load_move(self, name: str) -> List[Keyframe]:
        if name not in self._moves:
            raise KeyError(f"Move '{name}' not found.")
        return [Keyframe(**kf) for kf in self._moves[name].keyframes]

    def delete_move(self, name: str):
        if name in self._moves:
            del self._moves[name]
            self._save()

    def list_moves(self) -> List[Dict]:
        return [
            {
                "name":     m.name,
                "created":  m.created,
                "notes":    m.notes,
                "duration": m.total_duration_s,
                "extents":  m.extents,
                "keyframes": len(m.keyframes),
            }
            for m in self._moves.values()
        ]

    def rename_move(self, old_name: str, new_name: str):
        if old_name not in self._moves:
            raise KeyError(f"Move '{old_name}' not found.")
        move = self._moves.pop(old_name)
        move.name = new_name
        self._moves[new_name] = move
        self._save()
