#!/usr/bin/env python3
"""
holygrail.py — Holy Grail exposure brain for PiSlider  v3.0

Architecture
────────────
Three layers blend together each frame, with weights that shift
dynamically based on phase, tracker confidence, and conditions:

  1. ASTRONOMICAL MODEL  (_compute_astro)
     Sun/moon positions -> phase classification -> EV/Kelvin/interval
     priors. Includes ambient moonlight model (full moon +2.5 stops vs
     new moon), disc-in-frame geometry, and look-ahead anticipation
     (begins pre-adjusting EV before sun/moon crosses into frame).

  2. DNG CAPTURE TRACKER  (AdaptiveEVTracker + push_capture_ev)
     After every saved frame, app.py reads luminance from the thumbnail
     and calls push_capture_ev(). Builds a rolling window of 20 frames.
     Fits weighted linear regression -> slope (stops/frame) + R².
     Works at night - reads from the capture, not the preview.

  3. DYNAMIC BLEND WEIGHT  (_blend_weight)
     Pixel vs astro trust shifts each frame:
     - Deep stable night  -> 15% pixel / 85% astro
     - Active transition  -> 50/50
     - Day with clouds    -> 75% pixel / 25% astro
     - High R² (clean trend) -> more pixel weight
     - Moon rising/setting   -> more astro weight
"""

from __future__ import annotations

import math
import time
import datetime
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple

import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

from zoneinfo import ZoneInfo
from astral import LocationInfo
from astral.sun import elevation as sun_elevation, azimuth as sun_azimuth
from astral.moon import elevation as moon_elevation, azimuth as moon_azimuth
from astral.moon import phase as moon_phase

logger = logging.getLogger("PiSlider.HG")


# ─────────────────────────────────────────────────────────────────────────────
# HGSettings
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HGSettings:
    enabled: bool = True

    # Location & time
    lat: float = 49.8951
    lon: float = -97.1384
    tz:  str   = "America/Winnipeg"

    start_dt:        Optional[datetime.datetime] = None
    interval_sec:    float = 5.0
    frames:          int   = 1000
    vibration_delay: float = 1.0
    exposure_margin: float = 0.2

    # Camera geometry
    cam_az:  float = 180.0
    cam_alt: float = 0.0
    hfov:    float = 60.0
    vfov:    float = 40.0

    # ── EV targets per phase ─────────────────────────────────────────────────
    ev_day:      float = 13.0
    ev_golden:   float = 10.0
    ev_twilight: float =  6.0
    ev_night:    float =  3.0

    # Kelvin targets per phase
    kelvin_day:      int = 5500
    kelvin_golden:   int = 5200
    kelvin_twilight: int = 4800
    kelvin_night:    int = 3800

    # Per-phase intervals (s)
    interval_day:      float = 5.0
    interval_golden:   float = 7.0
    interval_twilight: float = 10.0
    interval_night:    float = 20.0

    # Aperture & ISO
    aperture_day:   float = 5.6
    aperture_night: float = 2.8
    iso_min:        int   = 100
    iso_max:        int   = 3200
    iso_max_night:  int   = 3200

    # ── Anchor exposure ───────────────────────────────────────────────────────
    # Set once by calibration shot. ALL meter shots use these exact settings
    # so every measurement is directly comparable — no compensation math needed.
    anchor_shutter_s: Optional[float] = None
    anchor_iso:       Optional[int]   = None
    anchor_ev:        Optional[float] = None

    # Shutter limits
    shutter_max_night:     float =  8.0
    shutter_max_twilight:  float =  4.0
    night_prefer_low_iso:  bool  = True

    # ── Histogram targets (0–255 luminance in meter JPEG) ────────────────────
    # User-facing controls for the look of the timelapse.
    # The system steers toward these using clean anchor-exposure meter shots.

    # Highlight protection: if this fraction of pixels exceeds clip_level,
    # pull exposure down regardless of trend. Prevents blown daylight skies.
    highlight_clip_level:    int   = 245    # pixel value = blown
    highlight_clip_limit:    float = 0.005  # 0.5% blown pixels = pull down

    # Midtone target: where we want the P50 of the histogram.
    # Night: push brighter to gather more light and show stars.
    # Lightroom/LRTimelapse normalises brightness in post anyway.
    midtone_target_day:   int   = 110
    midtone_target_night: int   = 80
    midtone_percentile:   float = 0.50  # steer this percentile toward target

    # Shadow floor: if too many pixels are crushed to black, boost exposure.
    shadow_floor_level: int   = 18     # below this = crushed black
    shadow_floor_limit: float = 0.60   # >60% crushed → boost needed

    # ── Per-phase agility (max stops/frame the output is allowed to change) ──
    # This is the primary "butter" control. Low = smooth. High = responsive.
    # Transitions (golden, twilight) get more agility. Stable phases get less.
    agility_day:      float = 0.008   # stable sun — almost no movement
    agility_golden:   float = 0.030   # golden/sunset — fast ramp allowed
    agility_twilight: float = 0.025   # still ramping, slightly slower
    agility_night:    float = 0.010   # stable darkness — tiny corrections only

    # Extra multiplier near the horizon (±15°) for moon rise/set events.
    horizon_agility_boost: float = 1.8

    # ── Tracker tuning ────────────────────────────────────────────────────────
    adaptive_weight:      float = 0.75
    ev_max_delta_flat:    float = 0.04
    ev_max_delta_fast:    float = 0.35
    kelvin_max_delta:     int   = 60
    anomaly_threshold_ev: float = 1.5
    tracker_window:       int   = 20
    tracker_warmup:       int   = 5
    slope_ma_window:      int   = 12   # frames to average for slope_ma
    # How strongly ev_smooth drifts toward astro_ev per frame in stable conditions.
    # Prevents multi-hour drift without visible jerky corrections.
    drift_pull_strength:  float = 0.0003

    # ── Celestial disc tuning ─────────────────────────────────────────────────
    disc_lookahead_min:   float = 10.0
    sun_weight:           float = 1.0
    moon_weight:          float = 0.4
    moon_phase_weight:    float = 1.0
    moonlight_ev_max:     float = 2.5


# ─────────────────────────────────────────────────────────────────────────────
# MeterShot — result of a clean anchor-exposure meter capture
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MeterShot:
    """
    Luminance histogram extracted from a JPEG captured at fixed anchor
    settings (anchor_shutter_s, anchor_iso). Because settings never change,
    every MeterShot is directly comparable — no compensation math needed.

    Fields that matter to the tracker:
      meter_ev          — EV computed from P50 luminance (the reference signal)
      highlight_fraction — fraction of pixels at or above clip_level
      shadow_fraction   — fraction of pixels at or below floor_level
      midtone_p50       — raw P50 luminance (0–255)
      condition         — 'clear' | 'hazy' | 'overcast' from histogram variance
      kelvin            — colour temperature (day only; astro ramp at night)
    """
    timestamp:          float
    frame_index:        int
    meter_ev:           float   # EV at anchor exposure — scene luminance signal
    midtone_p50:        int     # P50 luminance (0–255)
    highlight_fraction: float   # fraction of pixels >= clip_level
    shadow_fraction:    float   # fraction of pixels <= floor_level
    hist_std:           float   # stddev of luminance — sky variance / cloud indicator
    kelvin:             float
    condition:          str     # 'clear' | 'hazy' | 'overcast'
    is_anomaly:         bool  = False
    weight:             float = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# CaptureMeasurement — kept for push_capture_ev compatibility
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CaptureMeasurement:
    timestamp:    float
    frame_index:  int
    pixel_ev:     float
    kelvin:       float
    sky_fraction: float
    condition:    str
    is_anomaly:   bool  = False
    weight:       float = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# SkyMeasurement  (preview / day use)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SkyMeasurement:
    timestamp:    float
    ev:           float
    rg_ratio:     float
    bg_ratio:     float
    lum_mean:     float
    sky_fraction: float
    condition:    str
    source:       str   = 'preview'
    is_anomaly:   bool  = False
    weight:       float = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# SkyAnalyser
# ─────────────────────────────────────────────────────────────────────────────

class SkyAnalyser:
    _SKY_HSV_RANGES = [
        ((90,  20,  60), (130, 255, 255)),
        ((0,   0,  130), (180,  55, 230)),
        ((5,   40,  80), ( 35, 255, 255)),
        ((95,  10,  40), (140,  80, 200)),
    ]

    def __init__(self):
        self._prev_mask: Optional[np.ndarray] = None

    def analyse(
        self,
        frame_rgb: np.ndarray,
        cam_alt:   float = 0.0,
        sun_az:    float = 0.0,
        sun_alt:   float = 0.0,
        cam_az:    float = 0.0,
        hfov:      float = 60.0,
        vfov:      float = 40.0,
        moon_az:   float = 0.0,
        moon_alt:  float = 0.0,
        camera_ev: Optional[float] = None,
    ) -> Optional[SkyMeasurement]:
        if not _HAS_CV2 or frame_rgb is None:
            return None
        h, w = frame_rgb.shape[:2]
        if cam_alt < -25.0:
            return None

        hsv  = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
        mask = np.zeros((h, w), dtype=np.uint8)
        for lo, hi in self._SKY_HSV_RANGES:
            mask |= cv2.inRange(hsv,
                                np.array(lo, dtype=np.uint8),
                                np.array(hi, dtype=np.uint8))

        sky_top = max(1, int(h * 0.65))
        pos_mask = np.zeros((h, w), dtype=np.uint8)
        pos_mask[:sky_top, :] = 255
        mask &= pos_mask

        if self._prev_mask is not None and self._prev_mask.shape == mask.shape:
            mask = cv2.addWeighted(mask, 0.7, self._prev_mask, 0.3, 0).astype(np.uint8)
        self._prev_mask = mask.copy()

        sky_pixels   = frame_rgb[mask > 127]
        sky_fraction = float(np.sum(mask > 127)) / (h * w)
        if sky_fraction < 0.02 or len(sky_pixels) < 50:
            return None

        lum_arr = (0.2126 * sky_pixels[:, 0].astype(float) +
                   0.7152 * sky_pixels[:, 1].astype(float) +
                   0.0722 * sky_pixels[:, 2].astype(float))
        thresh   = np.percentile(lum_arr, 90)
        good     = lum_arr[lum_arr <= thresh]
        if len(good) == 0:
            good = lum_arr
        lum_mean = float(np.mean(good))
        lum_safe = max(lum_mean, 1.0)
        ev = math.log2((lum_safe / 255.0) ** 2.2 / 0.18) + 12.0

        rg = float(np.mean(sky_pixels[:, 0])) / max(float(np.mean(sky_pixels[:, 1])), 1.0)
        bg = float(np.mean(sky_pixels[:, 2])) / max(float(np.mean(sky_pixels[:, 1])), 1.0)

        hsv_sky  = hsv[mask > 127]
        sat_mean = float(np.mean(hsv_sky[:, 1])) if len(hsv_sky) > 0 else 0
        if sat_mean > 60:
            condition = 'clear'
        elif sat_mean > 25:
            condition = 'hazy'
        else:
            condition = 'overcast'

        weight = 1.0
        if camera_ev is not None and abs(ev - camera_ev) > 2.0:
            weight = 0.3

        return SkyMeasurement(
            timestamp    = time.time(),
            ev           = ev,
            rg_ratio     = rg,
            bg_ratio     = bg,
            lum_mean     = lum_mean,
            sky_fraction = sky_fraction,
            condition    = condition,
            source       = 'preview',
            weight       = weight,
        )


# ─────────────────────────────────────────────────────────────────────────────
# AdaptiveEVTracker
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveEVTracker:
    """
    Rolling window of meter-shot EV measurements.
    Each measurement comes from a dedicated anchor-exposure capture, so
    the EV values are directly comparable — no compensation math, no
    "infected" samples from exposure changes.

    Weighted linear regression gives slope = rate of EV change (stops/frame).
    slope_ma is the moving average of recent slope estimates — only sustained
    trends affect the output. A single cloudy frame cannot dominate.

    Highlight and shadow fractions from the histogram allow hard-limit
    overrides: if highlights are clipping, pull down regardless of trend.
    """

    def __init__(self, window_size: int = 20, warmup: int = 5,
                 slope_ma_window: int = 12):
        self.window_size     = window_size
        self.warmup          = warmup
        self._window: deque  = deque(maxlen=window_size)
        self._last_ev:     Optional[float] = None
        self._last_kelvin: Optional[float] = None
        self.ev_slope:     float = 0.0
        self.kelvin_slope: float = 0.0
        self.r_squared:    float = 0.0
        self.condition:    str   = 'unknown'
        self.n_frames:     int   = 0
        self._slope_history: deque[float] = deque(maxlen=slope_ma_window)
        self.slope_ma:       float = 0.0
        # meas_ev / meas_kelvin: what the SCENE is doing (from regression).
        # _last_ev / _last_kelvin: what we TOLD the camera (output state).
        # These are deliberately separate — output only moves by max_step.
        self.meas_ev:        Optional[float] = None
        self.meas_kelvin:    Optional[float] = None
        # Latest histogram stats (from most recent MeterShot)
        self.highlight_fraction: float = 0.0
        self.shadow_fraction:    float = 0.0
        self.midtone_p50:        int   = 128
        self.hist_std:           float = 0.0

    @property
    def is_warm(self) -> bool:
        return len(self._window) >= self.warmup

    def push_meter_shot(self, m: MeterShot) -> None:
        """
        Primary feedback path. Called after every anchor-exposure meter shot.
        m.meter_ev is already at a fixed reference — no normalization needed.
        """
        # Store histogram stats for highlight/shadow override logic
        self.highlight_fraction = m.highlight_fraction
        self.shadow_fraction    = m.shadow_fraction
        self.midtone_p50        = m.midtone_p50
        self.hist_std           = m.hist_std

        # Anomaly detection on the clean meter EV stream
        if len(self._window) >= 3:
            recent_evs = [x.meter_ev for x in self._window]
            median_ev  = float(np.median(recent_evs))
            if abs(m.meter_ev - median_ev) > 1.0:
                m.is_anomaly = True
                m.weight     = 0.15
                # Two consecutive anomalies in the same direction = real event
                if len(self._window) >= 2:
                    last = self._window[-1]
                    if (getattr(last, 'is_anomaly', False) and
                            (m.meter_ev - median_ev) * (last.meter_ev - median_ev) > 0):
                        m.weight = 1.0

        self._window.append(m)
        self.n_frames += 1
        self.condition = m.condition
        self._refit_meter()

    def push_capture(self, m: CaptureMeasurement) -> None:
        """Legacy path — converts CaptureMeasurement to minimal MeterShot."""
        ms = MeterShot(
            timestamp          = m.timestamp,
            frame_index        = m.frame_index,
            meter_ev           = m.pixel_ev,
            midtone_p50        = 128,
            highlight_fraction = 0.0,
            shadow_fraction    = 0.0,
            hist_std           = 20.0,
            kelvin             = m.kelvin,
            condition          = m.condition,
            weight             = m.weight,
        )
        self.push_meter_shot(ms)

    def seed(self, ev: float, kelvin: float) -> None:
        self._last_ev     = ev
        self._last_kelvin = kelvin

    def current_ev(self) -> Optional[float]:
        return self._last_ev

    def current_kelvin(self) -> Optional[float]:
        return self._last_kelvin

    def predict_ev(self, frames_ahead: float = 1.0) -> Optional[float]:
        if not self.is_warm or self._last_ev is None:
            return None
        return self._last_ev + self.ev_slope * frames_ahead

    def predict_kelvin(self, frames_ahead: float = 1.0) -> Optional[float]:
        if not self.is_warm or self._last_kelvin is None:
            return None
        return self._last_kelvin + self.kelvin_slope * frames_ahead

    def smooth_ev(
        self, ev_target: float,
        max_flat: float = 0.12, max_fast: float = 0.35,
    ) -> float:
        if self._last_ev is None:
            self._last_ev = ev_target
            return ev_target
        diff = ev_target - self._last_ev
        if abs(diff) > 1.5:
            # Emergency recovery — large drift, allow up to max_fast per frame
            max_delta = max_fast
        else:
            # Scale with slope, capped at 0.5 stops/frame
            capped_slope = min(abs(self.ev_slope), 0.5)
            slope_factor = min(1.0, capped_slope * 10.0)
            # Also scale with r_squared: low confidence = tighter limit
            # This kills transient dips (person, bird, cloud edge) that don't
            # produce a sustained trend. r_squared near 0 = noisy/unsustained.
            confidence_factor = max(0.1, self.r_squared)
            max_delta = max_flat * confidence_factor + (max_fast - max_flat) * slope_factor * confidence_factor
            max_delta = max(max_flat * 0.3, max_delta)   # floor: always allow tiny creep
        diff = max(-max_delta, min(max_delta, diff))
        self._last_ev += diff
        return self._last_ev

    def smooth_kelvin(self, kelvin_target: float, max_delta: int = 60) -> int:
        if self._last_kelvin is None:
            self._last_kelvin = float(kelvin_target)
            return int(kelvin_target)
        diff = kelvin_target - self._last_kelvin
        diff = max(-max_delta, min(max_delta, diff))
        self._last_kelvin += diff
        return int(self._last_kelvin)

    def get_status(self) -> Dict[str, Any]:
        return {
            "warm":        self.is_warm,
            "n_frames":    self.n_frames,
            "window_used": len(self._window),
            "ev_slope":    round(self.ev_slope, 4),
            "kelvin_slope":round(self.kelvin_slope, 2),
            "r_squared":   round(self.r_squared, 3),
            "condition":   self.condition,
            "last_ev":     round(self._last_ev, 3) if self._last_ev is not None else None,
            "last_kelvin": int(self._last_kelvin) if self._last_kelvin is not None else None,
        }

    def _refit_meter(self) -> None:
        """Refit regression over clean meter_ev measurements."""
        meas = list(self._window)
        if len(meas) < 2:
            self.ev_slope = self.kelvin_slope = self.r_squared = 0.0
            return

        xs  = np.array([m.frame_index for m in meas], dtype=float)
        xs -= xs[0]
        evs = np.array([m.meter_ev for m in meas])
        ws  = np.array([m.weight   for m in meas])
        ks  = np.array([m.kelvin   for m in meas])

        self.ev_slope     = _weighted_slope(xs, evs, ws)
        self.kelvin_slope = _weighted_slope(xs, ks,  ws)

        if len(xs) >= 3:
            ev_pred = evs[0] + self.ev_slope * xs
            ss_res  = float(np.sum(ws * (evs - ev_pred) ** 2))
            ev_mean = float(np.average(evs, weights=ws))
            ss_tot  = float(np.sum(ws * (evs - ev_mean) ** 2))
            self.r_squared = max(0.0, min(1.0, 1.0 - ss_res / max(ss_tot, 1e-9)))
        else:
            self.r_squared = 0.3

        # meas_ev = regression fitted value at latest frame.
        # This is the MEASUREMENT trend — what the scene is doing.
        # It is NOT written to _last_ev because _last_ev is the OUTPUT
        # (what we last told the camera). The output only moves by max_step
        # in _compute_params. Separating these two is the key to smooth output.
        if len(meas) >= 2:
            self.meas_ev = float(evs[0] + self.ev_slope * xs[-1])
        else:
            self.meas_ev = float(evs[-1])

        # Slope moving average — the stable trend rate.
        # Transient blips produce one out-of-family slope estimate that gets
        # averaged away across slope_ma_window frames. Only a sustained real
        # trend shifts slope_ma enough to drive meaningful output movement.
        self._slope_history.append(self.ev_slope)
        self.slope_ma = float(np.mean(self._slope_history))

        n    = min(3, len(meas))
        rw   = ws[-n:]
        wsum = float(np.sum(rw))
        if wsum > 0:
            self.meas_kelvin = float(np.average(ks[-n:], weights=rw))


# ─────────────────────────────────────────────────────────────────────────────
# HolyGrailController
# ─────────────────────────────────────────────────────────────────────────────

class HolyGrailController:

    def __init__(self, settings: Optional[HGSettings] = None):
        self.settings  = settings or HGSettings()
        self._tzinfo   = self._make_tzinfo(self.settings.tz)
        self._location = self._make_location()
        self._analyser = SkyAnalyser()
        self._tracker  = AdaptiveEVTracker(
            window_size    = self.settings.tracker_window,
            warmup         = self.settings.tracker_warmup,
            slope_ma_window= self.settings.slope_ma_window,
        )
        self._plan:        Optional[List[Dict[str, Any]]] = None
        self._frame_index: int = 0
        self._last_phase:  str = 'unknown'   # detect phase transitions

    # ── Configuration ─────────────────────────────────────────────────────────

    def set_settings(self, settings) -> None:
        import dataclasses as _dc
        if isinstance(settings, dict):
            valid = {f.name for f in _dc.fields(HGSettings)}
            clean = {k: v for k, v in settings.items() if k in valid}
            self.settings = HGSettings(**clean)
        else:
            self.settings = settings
        self._tzinfo   = self._make_tzinfo(self.settings.tz)
        self._location = self._make_location()
        self._tracker  = AdaptiveEVTracker(
            window_size    = self.settings.tracker_window,
            warmup         = self.settings.tracker_warmup,
            slope_ma_window= self.settings.slope_ma_window,
        )
        self._plan = None
        self._frame_index = 0
        self._last_phase  = 'unknown' 

    def get_settings_dict(self) -> Dict[str, Any]:
        import dataclasses as _dc
        return _dc.asdict(self.settings)

    # ── Capture feedback (primary closed-loop input) ──────────────────────────

    def push_capture_ev(
        self,
        pixel_ev:     float,
        kelvin:       float,
        frame_index:  int,
        sky_fraction: float = 0.5,
        condition:    str   = 'unknown',
    ) -> None:
        """
        Push EV measured from a real saved DNG/thumbnail.
        Primary closed-loop feedback. Works day and night.

        pixel_ev must use same formula as anchor_ev:
            ev = log2((lum/255)^2.2 / 0.18) + 12
        where lum is mean luminance of non-blown pixels.
        """
        m = CaptureMeasurement(
            timestamp    = time.time(),
            frame_index  = frame_index,
            pixel_ev     = pixel_ev,
            kelvin       = kelvin,
            sky_fraction = sky_fraction,
            condition    = condition,
        )
        self._tracker.push_capture(m)
        logger.debug(
            f"HG capture push: frame={frame_index} ev={pixel_ev:.3f} "
            f"K={kelvin:.0f} slope={self._tracker.ev_slope:.4f} "
            f"R2={self._tracker.r_squared:.2f}"
        )

    # ── Meter shot feedback (primary clean-signal path) ──────────────────────

    def push_meter_shot(
        self,
        jpeg_rgb:    np.ndarray,
        frame_index: int,
        sun_alt:     Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        PRIMARY feedback path. Call after every dedicated anchor-exposure
        meter JPEG. Because the camera is always at anchor_shutter_s /
        anchor_iso, every call is directly comparable — no math compensation.

        jpeg_rgb: uint8 RGB array decoded from the meter JPEG.
        Returns the MeterShot dict for logging, or None on failure.
        """
        s = self.settings
        if jpeg_rgb is None or jpeg_rgb.size == 0:
            return None

        try:
            h, w   = jpeg_rgb.shape[:2]
            lum    = (0.2126 * jpeg_rgb[:, :, 0].astype(float)
                    + 0.7152 * jpeg_rgb[:, :, 1].astype(float)
                    + 0.0722 * jpeg_rgb[:, :, 2].astype(float))
            lum_flat = lum.flatten()
            n_pixels = len(lum_flat)

            # ── Histogram stats ──────────────────────────────────────────────
            midtone_p50        = int(np.percentile(lum_flat, s.midtone_percentile * 100))
            highlight_fraction = float(np.mean(lum_flat >= s.highlight_clip_level))
            shadow_fraction    = float(np.mean(lum_flat <= s.shadow_floor_level))
            hist_std           = float(np.std(lum_flat))

            # ── meter_ev from median luminance ───────────────────────────────
            # We use P50 (not mean) so a few bright street-lights at night
            # don't dominate the measurement.
            lum_safe  = max(float(midtone_p50), 1.0)
            meter_ev  = math.log2(max((lum_safe / 255.0) ** 2.2, 1e-9) / 0.18) + 12.0

            # ── Condition from histogram variance ────────────────────────────
            # High std = lots of tonal contrast = clear sky with clouds.
            # Low std = flat = deep overcast.
            condition = ('clear' if hist_std > 45 else
                         'hazy'  if hist_std > 18 else 'overcast')

            # ── Kelvin (day only — night Kelvin from astro ramp) ─────────────
            if sun_alt is None:
                from astral.sun import elevation as _se
                sun_alt = _se(self._location.observer,
                              datetime.datetime.now(self._tzinfo))

            if sun_alt >= 0.0:
                r_mean = float(np.mean(jpeg_rgb[:, :, 0]))
                g_mean = max(float(np.mean(jpeg_rgb[:, :, 1])), 1.0)
                b_mean = float(np.mean(jpeg_rgb[:, :, 2]))
                rg     = r_mean / g_mean
                bg     = b_mean / g_mean
                kelvin = float(max(2500, min(10000,
                    5500 - (rg - 1.0) * 2000 + (bg - 1.0) * 1600)))
            else:
                kelvin = float(self._kelvin_for_phase(sun_alt))

            ms = MeterShot(
                timestamp          = time.time(),
                frame_index        = frame_index,
                meter_ev           = meter_ev,
                midtone_p50        = midtone_p50,
                highlight_fraction = highlight_fraction,
                shadow_fraction    = shadow_fraction,
                hist_std           = hist_std,
                kelvin             = kelvin,
                condition          = condition,
            )
            self._tracker.push_meter_shot(ms)
            logger.debug(
                f"MeterShot frame={frame_index}: ev={meter_ev:.3f} "
                f"p50={midtone_p50} hl={highlight_fraction:.3f} "
                f"shadow={shadow_fraction:.3f} cond={condition} K={kelvin:.0f}"
            )
            return {
                "meter_ev": round(meter_ev, 3),
                "midtone_p50": midtone_p50,
                "highlight_fraction": round(highlight_fraction, 4),
                "shadow_fraction": round(shadow_fraction, 4),
                "hist_std": round(hist_std, 1),
                "condition": condition,
                "kelvin": int(kelvin),
            }
        except Exception as e:
            logger.warning(f"push_meter_shot failed: {e}")
            return None

    # ── Preview feedback (daylight only, supplementary) ───────────────────────

    def push_preview_frame(
        self,
        frame_rgb:  np.ndarray,
        camera_ev:  Optional[float] = None,
    ) -> Optional[SkyMeasurement]:
        """Preview metering — day only (sun > -6 deg). Night = None immediately."""
        s   = self.settings
        now = datetime.datetime.now(self._tzinfo)
        obs = self._location.observer
        sun_alt = sun_elevation(obs, now)
        if sun_alt < -6.0:
            return None

        sun_az   = sun_azimuth(obs, now)
        moon_alt = moon_elevation(obs, now)
        moon_az  = moon_azimuth(obs, now)

        m = self._analyser.analyse(
            frame_rgb,
            cam_alt=s.cam_alt, sun_az=sun_az, sun_alt=sun_alt,
            cam_az=s.cam_az, hfov=s.hfov, vfov=s.vfov,
            moon_az=moon_az, moon_alt=moon_alt, camera_ev=camera_ev,
        )
        if m is not None:
            self.push_capture_ev(
                pixel_ev     = m.ev,
                kelvin       = float(_rg_bg_to_kelvin(m.rg_ratio, m.bg_ratio, m.lum_mean)),
                frame_index  = self._frame_index,
                sky_fraction = m.sky_fraction,
                condition    = m.condition,
            )
            # Preview is less reliable than a real DNG — reduce weight
            if self._tracker._window:
                self._tracker._window[-1].weight *= 0.4
        return m

    def push_capture_frame(self, frame_rgb: np.ndarray) -> Optional[SkyMeasurement]:
        return self.push_preview_frame(frame_rgb)

    def seed_from_calibration(self, ev: float, kelvin: int) -> None:
        self._tracker.seed(ev, float(kelvin))
        logger.info(f"HG tracker seeded: EV={ev:.2f} K={kelvin}")

    # ── Main API ──────────────────────────────────────────────────────────────

    def get_next_shot_parameters(
        self,
        now: Optional[datetime.datetime] = None,
    ) -> Dict[str, Any]:
        if not self.settings.enabled:
            return {
                "mode": "manual", "iso": self.settings.iso_min,
                "shutter": "1/125", "shutter_s": 1/125,
                "kelvin": self.settings.kelvin_day,
                "interval": self.settings.interval_sec,
            }
        if now is None:
            now = datetime.datetime.now(self._tzinfo)
        else:
            now = self._ensure_tz(now)
        return self._compute_params(now)

    # ── Core per-frame computation ────────────────────────────────────────────

    def _compute_params(self, now: datetime.datetime) -> Dict[str, Any]:
        s   = self.settings
        obs = self._location.observer

        # 1. Astronomical model
        sun_alt      = sun_elevation(obs, now)
        sun_az       = sun_azimuth(obs, now)
        moon_alt     = moon_elevation(obs, now)
        moon_az      = moon_azimuth(obs, now)
        moon_ph_days = moon_phase(now)
        moon_ph      = max(0.0, min(1.0, moon_ph_days / 29.53))

        phase         = _phase_for_alt(sun_alt)
        astro_ev      = self._ev_for_phase(sun_alt)
        astro_kelvin  = self._kelvin_for_phase(sun_alt)
        interval_base = self._interval_for_phase(phase)

        # Flush tracker window on phase transition so stale data from the
        # previous phase (e.g. 700 night frames) doesn't corrupt regression
        # at the first bright dawn frame.
        if phase != self._last_phase and self._last_phase != 'unknown':
            logger.info(f"HG phase transition {self._last_phase}→{phase}: flushing tracker window")
            self._tracker._window.clear()
            self._tracker.ev_slope     = 0.0
            self._tracker.kelvin_slope = 0.0
            self._tracker.r_squared    = 0.0
            # Keep _last_ev and _last_kelvin so smooth_ev has a starting point
        self._last_phase = phase

        # 2. Ambient moonlight EV contribution
        # Full moon at zenith raises ambient by ~2.5 stops vs new moon.
        moonlight_ev = 0.0
        if phase == 'night' and moon_alt > 0.0:
            phase_factor = math.sin(moon_ph * math.pi) ** 0.5
            alt_factor   = math.sin(math.radians(max(0.0, moon_alt)))
            moonlight_ev = s.moonlight_ev_max * phase_factor * alt_factor
            if moonlight_ev > 0.3:
                astro_ev += moonlight_ev * 0.6

        # 3. Disc-in-frame geometric offset
        disc_ev_offset = self._disc_ev_offset(
            sun_az, sun_alt, moon_az, moon_alt, moon_ph
        )

        # 4. Look-ahead anticipatory ramp
        disc_entry = self.next_disc_entry()
        disc_anticipation_ev = 0.0
        if disc_entry:
            for body, sign in [("sun", -s.sun_weight),
                                ("moon", -s.moon_weight * moon_ph)]:
                if body in disc_entry:
                    mins_away = disc_entry[body]["minutes"]
                    if mins_away < s.disc_lookahead_min:
                        ramp = _smootherstep(1.0 - mins_away / s.disc_lookahead_min)
                        disc_anticipation_ev += sign * ramp

        total_astro_ev = astro_ev + disc_ev_offset + disc_anticipation_ev

        # 5. Dynamic blend weights
        pixel_w, astro_w = self._blend_weight(sun_alt, moon_alt, moon_ph)

        # 6. Tracker predictions
        tracker_ev     = self._tracker.predict_ev(frames_ahead=1.0)
        tracker_kelvin = self._tracker.predict_kelvin(frames_ahead=1.0)

        # 7. Blend
        if tracker_ev is not None:
            blended_ev = total_astro_ev * astro_w + tracker_ev * pixel_w
            # Kelvin: only blend pixel Kelvin during daylight (sun > 0).
            if sun_alt >= 0 and tracker_kelvin is not None:
                kelvin_pixel_w = pixel_w * min(1.0, sun_alt / 10.0)
                blended_kelvin = astro_kelvin * (1.0 - kelvin_pixel_w) + tracker_kelvin * kelvin_pixel_w
            else:
                blended_kelvin = float(astro_kelvin)
        else:
            blended_ev     = total_astro_ev
            blended_kelvin = float(astro_kelvin)
            pixel_w        = 0.0
            astro_w        = 1.0

        # ── 7b. Phase-variable agility ─────────────────────────────────────────
        #
        # max_step is the maximum EV change per frame. It is phase-dependent:
        # large during golden/twilight (fast ramp needed), small at stable day
        # or deep night (butter smooth). Additionally scaled up near the
        # horizon (±15°) for moon rise/set and civil-twilight events.
        #
        # This is the primary "butter" control. The slope-driven output is then
        # clamped to max_step, so no single frame can make a large jump.
        phase_agility = {
            'day':      s.agility_day,
            'golden':   s.agility_golden,
            'twilight': s.agility_twilight,
            'night':    s.agility_night,
        }.get(phase, s.agility_golden)

        # Horizon boost: near sunrise/sunset or moon crossing the horizon,
        # allow more agility so we don't lag behind fast lighting changes.
        horizon_factor = max(0.0, min(1.0, (15.0 - abs(sun_alt)) / 15.0))
        max_step = phase_agility * (1.0 + (s.horizon_agility_boost - 1.0) * horizon_factor)

        # ── 8. EV output path ─────────────────────────────────────────────────
        anchor_set   = (s.anchor_ev is not None and
                        s.anchor_shutter_s is not None and
                        s.anchor_iso is not None)
        tracker_warm = self._tracker.is_warm
        last_ev      = self._tracker._last_ev

        highlight_override = False
        shadow_override    = False

        if anchor_set and not tracker_warm:
            # Cold start — hold at calibration anchor, no movement yet.
            # Seed _last_ev so first warm frame has a clean starting point.
            ev_smooth = s.anchor_ev
            if self._tracker._last_ev is None:
                self._tracker._last_ev = s.anchor_ev

        elif tracker_warm and last_ev is not None:
            # ── Slope-driven output (primary path) ────────────────────────────
            #
            # KEY DESIGN: slope_ma is the rate of change of the SCENE
            # (from meter shot measurements). We apply a fraction of it
            # each frame, clamped to max_step. The output (ev_smooth) is
            # a moving average of previous outputs, not a jump to the
            # current measurement. This is what makes it "butter smooth".
            #
            # Think of it like a ship's rudder: the slope_ma tells us
            # which direction the light is going; max_step limits how
            # fast we turn the wheel. We never jerk to match the reading —
            # we steer toward it gradually.

            slope_step = max(-max_step, min(max_step,
                             self._tracker.slope_ma * pixel_w))

            # Drift pull: gently steer toward meas_ev (the regression
            # measurement) so the output doesn't wander away from reality
            # during flat/stable conditions. Scaled by how far we've drifted.
            meas_ev = self._tracker.meas_ev
            if meas_ev is not None:
                drift_gap  = meas_ev - last_ev
                # Pull proportional to gap, but soft: capped at max_step/4.
                # This means even a 1-stop gap closes in ~50 frames, not 1.
                drift_pull = max(-max_step * 0.25,
                                 min(max_step * 0.25,
                                     drift_gap * s.drift_pull_strength * 10.0))
            else:
                drift_pull = (total_astro_ev - last_ev) * s.drift_pull_strength

            ev_smooth = last_ev + slope_step + drift_pull

            # ── Hard histogram overrides ──────────────────────────────────────
            # These fire AFTER the slope calculation and can push ev_smooth
            # further than max_step would allow, because they are hard user
            # limits — not noise responses.

            # 1. Highlight protection: pull down if blown
            if self._tracker.highlight_fraction > s.highlight_clip_limit:
                excess     = self._tracker.highlight_fraction - s.highlight_clip_limit
                correction = -min(0.5, excess * 20.0)
                ev_smooth  = min(ev_smooth, last_ev + correction)
                highlight_override = True
                logger.debug(
                    f"HG highlight override: hl={self._tracker.highlight_fraction:.3f}"
                    f" correction={correction:.3f}")

            # 2. Shadow boost: push up if too many pixels crushed to black
            elif (self._tracker.shadow_fraction > s.shadow_floor_limit
                  and phase in ('night', 'twilight')):
                excess     = self._tracker.shadow_fraction - s.shadow_floor_limit
                correction = min(0.4, excess * 3.0)
                ev_smooth  = max(ev_smooth, last_ev + correction)
                shadow_override = True
                logger.debug(
                    f"HG shadow boost: shadow={self._tracker.shadow_fraction:.3f}"
                    f" correction={correction:.3f}")

            # Write ev_smooth back as the new output state.
            # CRITICAL: this is what next frame's slope_step builds on.
            # Without this write-back, every frame starts from the same
            # _last_ev and the smoothing does nothing.
            self._tracker._last_ev = ev_smooth

        else:
            # Fully cold and no anchor — follow astro model with rate limiting
            ev_smooth = self._tracker.smooth_ev(
                total_astro_ev,
                max_flat = s.ev_max_delta_flat,
                max_fast = s.ev_max_delta_fast,
            )

        # ── Kelvin output ─────────────────────────────────────────────────────
        k_min = {'day': 4000, 'golden': 3800, 'twilight': 3500, 'night': 3000}.get(phase, 3500)
        k_max = {'day': 7500, 'golden': 6500, 'twilight': 5500, 'night': 5000}.get(phase, 6000)
        blended_kelvin = max(k_min, min(k_max, blended_kelvin))
        if self._tracker._last_kelvin is not None:
            self._tracker._last_kelvin = max(k_min, min(k_max, self._tracker._last_kelvin))
        kelvin_smooth = self._tracker.smooth_kelvin(blended_kelvin, s.kelvin_max_delta)

        # 9. EV -> ISO + shutter
        iso, shutter_s = self._ev_to_exposure(ev_smooth, phase)

        # 10. Interval floor
        required = shutter_s + s.vibration_delay + s.exposure_margin
        interval = max(interval_base, required)

        condition = self._tracker.condition if self._tracker.is_warm else 'prior'

        return {
            "mode":               "holygrail",
            "phase":              phase,
            "condition":          condition,
            "sun_alt":            round(sun_alt, 2),
            "sun_az":             round(sun_az, 2),
            "moon_alt":           round(moon_alt, 2),
            "moon_az":            round(moon_az, 2),
            "moon_phase":         round(moon_ph, 3),
            "moonlight_ev":       round(moonlight_ev, 3),
            "ev_astro":           round(total_astro_ev, 3),
            "ev_disc_offset":     round(disc_ev_offset, 3),
            "ev_anticipation":    round(disc_anticipation_ev, 3),
            "ev_tracker":         round(tracker_ev, 3) if tracker_ev is not None else None,
            "ev_blended":         round(blended_ev, 3),
            "ev_final":           round(ev_smooth, 3),
            "ev_target":          round(total_astro_ev, 3),
            "ev_offset":          round(disc_ev_offset, 3),
            "pixel_weight":       round(pixel_w, 2),
            "astro_weight":       round(astro_w, 2),
            "tracker_warm":       self._tracker.is_warm,
            "ev_slope":           round(self._tracker.ev_slope, 4),
            "slope_ma":           round(self._tracker.slope_ma, 4),
            "r_squared":          round(self._tracker.r_squared, 3),
            "max_step":           round(max_step, 4),
            "highlight_override": highlight_override,
            "shadow_override":    shadow_override,
            "highlight_fraction": round(self._tracker.highlight_fraction, 4),
            "shadow_fraction":    round(self._tracker.shadow_fraction, 4),
            "midtone_p50":        self._tracker.midtone_p50,
            "hist_std":           round(self._tracker.hist_std, 1),
            "iso":                iso,
            "shutter":            _format_shutter(shutter_s),
            "shutter_s":          shutter_s,
            "kelvin":             kelvin_smooth,
            "interval":           interval,
            "disc_entry":         disc_entry,
        }

    # ── Dynamic blend weight ──────────────────────────────────────────────────

    def _blend_weight(
        self, sun_alt: float, moon_alt: float, moon_ph: float,
    ) -> Tuple[float, float]:
        if not self._tracker.is_warm:
            return 0.0, 1.0

        # Base pixel weight from phase
        if sun_alt > 10:
            base_pixel = 0.75
        elif sun_alt > 0:
            t = (10 - sun_alt) / 10.0
            base_pixel = 0.75 - t * 0.25
        elif sun_alt > -6:
            t = (-sun_alt) / 6.0
            base_pixel = 0.50 - t * 0.20
        elif sun_alt > -12:
            t = (-6 - sun_alt) / 6.0
            base_pixel = 0.30 - t * 0.15
        else:
            base_pixel = 0.15

        # R² modulation: confident trend -> more pixel weight
        r2     = self._tracker.r_squared
        r2_mod = max(-0.15, min(0.20, (r2 - 0.3) / 0.7 * 0.20))

        # Condition modulation
        cond_mod = {'clear': +0.05, 'overcast': -0.10}.get(
            self._tracker.condition, 0.0)

        # Moon near horizon -> rely more on astro
        moon_mod = 0.0
        if -5 < moon_alt < 15 and moon_ph > 0.3:
            moon_mod = -0.08

        pixel_w = max(0.05, min(0.90, base_pixel + r2_mod + cond_mod + moon_mod))
        return pixel_w, 1.0 - pixel_w

    # ── Celestial forecast ────────────────────────────────────────────────────

    def get_celestial_forecast(
        self,
        minutes_ahead: float = 60.0,
        step_min:      float = 1.0,
    ) -> List[Dict[str, Any]]:
        now = datetime.datetime.now(self._tzinfo)
        obs = self._location.observer
        s   = self.settings
        results = []
        for i in range(int(minutes_ahead / step_min)):
            dt       = now + datetime.timedelta(minutes=i * step_min)
            sun_alt  = sun_elevation(obs, dt);  sun_az  = sun_azimuth(obs, dt)
            moon_alt = moon_elevation(obs, dt); moon_az = moon_azimuth(obs, dt)
            sun_in,  sun_dist  = _is_in_frame(sun_az,  sun_alt,
                                               s.cam_az, s.cam_alt, s.hfov, s.vfov)
            moon_in, moon_dist = _is_in_frame(moon_az, moon_alt,
                                               s.cam_az, s.cam_alt, s.hfov, s.vfov)
            results.append({
                "minutes_from_now": i * step_min,
                "sun_alt":     round(sun_alt, 2),  "sun_az":   round(sun_az, 2),
                "sun_in_frame": sun_in,             "sun_dist": round(sun_dist, 3),
                "moon_alt":    round(moon_alt, 2),  "moon_az":  round(moon_az, 2),
                "moon_in_frame": moon_in,           "moon_dist":round(moon_dist, 3),
            })
        return results

    def next_disc_entry(self) -> Dict[str, Any]:
        forecast = self.get_celestial_forecast(
            minutes_ahead=self.settings.disc_lookahead_min, step_min=0.5)
        result = {}
        for entry in forecast:
            if entry["sun_in_frame"]  and "sun"  not in result:
                result["sun"]  = {"minutes": entry["minutes_from_now"],
                                   "alt": entry["sun_alt"],  "az": entry["sun_az"]}
            if entry["moon_in_frame"] and "moon" not in result:
                result["moon"] = {"minutes": entry["minutes_from_now"],
                                   "alt": entry["moon_alt"], "az": entry["moon_az"]}
            if "sun" in result and "moon" in result:
                break
        return result

    def simulate_plan(self) -> List[Dict[str, Any]]:
        s = self.settings
        start_dt = (datetime.datetime.now(self._tzinfo)
                    if not s.start_dt else self._ensure_tz(s.start_dt))
        plan = []
        dt = start_dt
        for i in range(s.frames):
            p = self._compute_astro(dt)
            p["index"] = i; p["time"] = dt.isoformat()
            plan.append(p)
            dt += datetime.timedelta(seconds=p["interval"])
        self._plan = plan; self._frame_index = 0
        return plan

    def get_tracker_status(self) -> Dict[str, Any]:
        return self._tracker.get_status()

    # ── Pure astro (no tracker) ───────────────────────────────────────────────

    def _compute_astro(self, dt: datetime.datetime) -> Dict[str, Any]:
        s   = self.settings
        obs = self._location.observer
        sun_alt  = sun_elevation(obs, dt);  sun_az  = sun_azimuth(obs, dt)
        moon_alt = moon_elevation(obs, dt); moon_az = moon_azimuth(obs, dt)
        moon_ph  = max(0.0, min(1.0, moon_phase(dt) / 29.53))
        phase         = _phase_for_alt(sun_alt)
        astro_ev      = self._ev_for_phase(sun_alt)
        astro_kelvin  = self._kelvin_for_phase(sun_alt)
        interval_base = self._interval_for_phase(phase)
        disc_offset   = self._disc_ev_offset(sun_az, sun_alt, moon_az, moon_alt, moon_ph)
        ev_target     = astro_ev + disc_offset
        iso, shutter_s = self._ev_to_exposure(ev_target, phase)
        required = shutter_s + s.vibration_delay + s.exposure_margin
        return {
            "mode": "holygrail", "phase": phase,
            "sun_alt": round(sun_alt, 2), "sun_az": round(sun_az, 2),
            "moon_alt": round(moon_alt, 2), "moon_az": round(moon_az, 2),
            "moon_phase": round(moon_ph, 3),
            "ev_target": round(ev_target, 3), "ev_final": round(ev_target, 3),
            "ev_offset": round(disc_offset, 3),
            "iso": iso, "shutter": _format_shutter(shutter_s), "shutter_s": shutter_s,
            "kelvin": astro_kelvin, "interval": max(interval_base, required),
        }

    # ── Phase blending ────────────────────────────────────────────────────────

    def _ev_for_phase(self, sun_alt: float) -> float:
        s = self.settings
        p0, p1, t = _phase_pair(sun_alt)
        m = {"day": s.ev_day, "golden": s.ev_golden,
             "twilight": s.ev_twilight, "night": s.ev_night}
        e0, e1 = m.get(p0, s.ev_day), m.get(p1, s.ev_day)
        return e0 if p0 == p1 else e0 + (e1 - e0) * t

    def _kelvin_for_phase(self, sun_alt: float) -> int:
        s = self.settings
        p0, p1, t = _phase_pair(sun_alt)
        m = {"day": s.kelvin_day, "golden": s.kelvin_golden,
             "twilight": s.kelvin_twilight, "night": s.kelvin_night}
        k0, k1 = m.get(p0, s.kelvin_day), m.get(p1, s.kelvin_day)
        return int(k0) if p0 == p1 else int(k0 + (k1 - k0) * t)

    def _interval_for_phase(self, phase: str) -> float:
        s = self.settings
        return {"day": s.interval_day, "golden": s.interval_golden,
                "twilight": s.interval_twilight, "night": s.interval_night
                }.get(phase, s.interval_day)

    def _disc_ev_offset(
        self, sun_az: float, sun_alt: float,
        moon_az: float, moon_alt: float, moon_ph: float,
    ) -> float:
        s = self.settings
        offset = 0.0
        sun_in,  sd = _is_in_frame(sun_az,  sun_alt,  s.cam_az, s.cam_alt, s.hfov, s.vfov)
        moon_in, md = _is_in_frame(moon_az, moon_alt, s.cam_az, s.cam_alt, s.hfov, s.vfov)
        if sun_in:
            offset += -s.sun_weight * (1.0 - sd)
        if moon_in:
            offset += -0.3 * s.moon_weight * moon_ph * s.moon_phase_weight * (1.0 - md)
        return offset

    def _ev_to_exposure(self, ev: float, phase: str) -> Tuple[int, float]:
        s = self.settings
        if (s.anchor_shutter_s is not None
                and s.anchor_iso is not None
                and s.anchor_ev is not None):
            ev_delta = ev - s.anchor_ev
            if phase == 'night':
                slo, shi = 1.0, s.shutter_max_night
            elif phase == 'twilight':
                slo, shi = 1/500, s.shutter_max_twilight
            else:
                slo, shi = 1/8000, 1.0
            iso_max = s.iso_max_night if phase in ('twilight', 'night') else s.iso_max
            new_s = s.anchor_shutter_s / (2 ** ev_delta)
            if slo <= new_s <= shi:
                return s.anchor_iso, new_s
            new_s   = max(slo, min(shi, new_s))
            s_ev    = math.log2(s.anchor_shutter_s / new_s)
            remain  = ev_delta - s_ev
            new_iso = s.anchor_iso / (2 ** remain)
            new_iso = max(s.iso_min, min(iso_max, int(round(new_iso / 100) * 100)))
            return new_iso, new_s

        # Fallback aperture-based
        aperture = s.aperture_night if phase in ('twilight', 'night') else s.aperture_day
        iso_max  = s.iso_max_night  if phase in ('twilight', 'night') else s.iso_max
        if phase == 'night':
            slo, shi = 1.0, s.shutter_max_night
            iso_ord  = ([800,1600,3200,400,200,100] if s.night_prefer_low_iso
                        else [3200,1600,800,400,200,100])
        elif phase == 'twilight':
            slo, shi = 1/500, s.shutter_max_twilight
            iso_ord  = [1600,3200,800,400,200,100]
        else:
            slo, shi = 1/8000, 1.0
            iso_ord  = [100,200,400,800,1600,3200]
        iso_ord = [x for x in iso_ord if s.iso_min <= x <= iso_max] or [s.iso_min]
        def _t(iso): return (aperture**2) / (2**(ev + math.log2(iso/100.0)))
        for iso in iso_ord:
            t = _t(iso)
            if slo <= t <= shi:
                return iso, t
        mid = math.sqrt(slo * shi)
        best_iso, best_err = iso_ord[0], float("inf")
        for iso in iso_ord:
            err = abs(math.log(max(_t(iso),1e-9)) - math.log(mid))
            if err < best_err:
                best_err, best_iso = err, iso
        return best_iso, max(slo, min(_t(best_iso), shi))

    # ── Location helpers ──────────────────────────────────────────────────────

    def _make_location(self) -> LocationInfo:
        s = self.settings
        return LocationInfo(name="UserLocation", region="",
                            timezone=s.tz, latitude=s.lat, longitude=s.lon)

    @staticmethod
    def _make_tzinfo(tz_name: str) -> ZoneInfo:
        try:    return ZoneInfo(tz_name)
        except: return ZoneInfo("UTC")

    def _ensure_tz(self, dt: datetime.datetime) -> datetime.datetime:
        return dt.replace(tzinfo=self._tzinfo) if dt.tzinfo is None else dt.astimezone(self._tzinfo)


# ─────────────────────────────────────────────────────────────────────────────
# Pure helper functions
# ─────────────────────────────────────────────────────────────────────────────

def _wrap_angle(a: float) -> float:
    return (a + 180) % 360 - 180

def _smootherstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * t * (t * (6 * t - 15) + 10)

def _phase_for_alt(sun_alt: float) -> str:
    if sun_alt > 10:  return "day"
    if sun_alt >  0:  return "golden"
    if sun_alt > -6:  return "twilight"
    return "night"

def _phase_pair(sun_alt: float) -> Tuple[str, str, float]:
    if sun_alt > 10:           return ("day",      "day",      0.0)
    if 0   < sun_alt <= 10:    return ("day",      "golden",   _smootherstep((10 - sun_alt) / 10.0))
    if -6  < sun_alt <= 0:     return ("golden",   "twilight", _smootherstep(-sun_alt / 6.0))
    if -12 < sun_alt <= -6:    return ("twilight", "night",    _smootherstep((-6 - sun_alt) / 6.0))
    return ("night", "night", 0.0)

def _is_in_frame(
    obj_az: float, obj_alt: float,
    cam_az: float, cam_alt: float,
    hfov: float, vfov: float,
) -> Tuple[bool, float]:
    d_az  = _wrap_angle(obj_az - cam_az)
    d_alt = obj_alt - cam_alt
    if abs(d_az) <= hfov/2 and abs(d_alt) <= vfov/2:
        nx   = abs(d_az)  / (hfov/2) if hfov > 0 else 0.0
        ny   = abs(d_alt) / (vfov/2) if vfov > 0 else 0.0
        return True, min(1.0, math.sqrt((nx*nx + ny*ny) / 2.0))
    return False, 1.0

def _format_shutter(t: float) -> str:
    if t >= 1.0: return f"{round(t,3)}s"
    return f"1/{max(1, int(round(1.0/t)))}"

def _rg_bg_to_kelvin(rg: float, bg: float, luminance: float = 128.0) -> int:
    kelvin_raw = 5500 - (rg - 1.0) * 2000 + (bg - 1.0) * 1600
    trust  = max(0.0, min(1.0, (luminance - 10.0) / 50.0))
    kelvin = trust * kelvin_raw + (1.0 - trust) * 5000.0
    return int(max(2500, min(10000, kelvin)))

def _weighted_slope(t: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    if len(t) < 2: return 0.0
    w_sum = float(np.sum(w))
    if w_sum == 0: return 0.0
    t_mean = float(np.average(t, weights=w))
    y_mean = float(np.average(y, weights=w))
    num = float(np.sum(w * (t - t_mean) * (y - y_mean)))
    den = float(np.sum(w * (t - t_mean) ** 2))
    return num / den if abs(den) > 1e-10 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random
    hg = HolyGrailController()
    print("Cold:", hg.get_next_shot_parameters())
    random.seed(42)
    base_ev = 6.0
    for i in range(25):
        ev   = base_ev + i * 0.05 + random.gauss(0, 0.08)
        kelv = 4800.0 + random.gauss(0, 50)
        hg.push_capture_ev(ev, kelv, i, sky_fraction=0.4, condition="clear")
    p = hg.get_next_shot_parameters()
    print(f"Warm: EV={p['ev_final']:.3f} shutter={p['shutter']} "
          f"ISO={p['iso']} K={p['kelvin']} "
          f"pixel_w={p['pixel_weight']} astro_w={p['astro_weight']} "
          f"slope={p['ev_slope']:.4f} R2={p['r_squared']:.3f}")
