#!/usr/bin/env python3
"""
sidecar.py — XMP sidecar generator for PiSlider timelapse frames.

Purpose
-------
For Holy Grail timelapse, the camera cannot always hit the *ideal* exposure
exactly (e.g., limited to specific shutter or ISO increments). To achieve
buttery-smooth transitions in post (Lightroom / LRTimelapse), we record the
difference between the *ideal* EV (HolyGrail) and the *actual* camera exposure
in an .XMP sidecar file.

Lightroom interprets:
    crs:Exposure2012 = exposure offset in stops

So we compute:
    delta_ev = ideal_ev - actual_ev

and write that to the sidecar.
"""

from __future__ import annotations

import os
import math
from typing import Optional, Dict, Any


# ---------------------------------------------------------------------------
# EV computation helpers
# ---------------------------------------------------------------------------

def parse_shutter_to_seconds(s: Any) -> float:
    """
    Convert a shutter spec into seconds.
    Accepts: 0.5, "0.5", "1/125", 1/125
    """
    if isinstance(s, (int, float)):
        return float(s)
    try:
        s_str = str(s)
        if "/" in s_str:
            n, d = s_str.split("/")
            return float(n) / float(d)
        return float(s_str)
    except Exception:
        return 0.0


def compute_ev(iso: float, shutter_s: float, aperture: float) -> Optional[float]:
    """
    Compute EV at ISO 100 for given ISO, shutter, aperture.
    Formula: EV = log2(N^2 / t * 100 / ISO)
    """
    try:
        iso = float(iso)
        shutter_s = float(shutter_s)
        aperture = float(aperture)
        if iso <= 0 or shutter_s <= 0 or aperture <= 0:
            return None

        # Standard Photometry calculation
        ev = math.log2((aperture ** 2) / shutter_s * (100.0 / iso))
        return ev
    except Exception:
        return None


# ---------------------------------------------------------------------------
# XMP builder
# ---------------------------------------------------------------------------

def _build_xmp_content(
    exposure_offset_ev: float,
    temperature: Optional[int] = None,
    extra_tags: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Build a minimal Lightroom-compatible XMP sidecar content string.
    """
    exp_str = f"{exposure_offset_ev:.3f}"

    if temperature is None:
        temperature = 0
    temp_str = str(int(temperature))

    # Basic XMP skeleton with Camera Raw (crs) namespace.
    # This is valid for Lightroom & Adobe Camera Raw.
    xmp = f"""<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
   xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/">

  <rdf:Description rdf:about=""
    crs:Version="13.0"
    crs:ProcessVersion="11.0"
    crs:Exposure2012="{exp_str}"
    crs:Temperature="{temp_str}"
    crs:Tint="0"
    crs:Exposure2012Auto="False">
  </rdf:Description>

 </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>"""
    return xmp


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def write_xmp_sidecar(
    raw_path: str,
    hg_params: Optional[Dict[str, Any]] = None,
    camera_settings: Optional[Dict[str, Any]] = None,
    extra_tags: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Write an XMP sidecar next to a RAW/DNG file with exposure delta info.

    Parameters
    ----------
    raw_path : str
        Path to the RAW / DNG file (e.g., Frame_0001.DNG).
    hg_params : dict
        Parameters from HolyGrailController.get_next_shot_parameters()
    camera_settings : dict
        Actual camera settings used (iso, shutter, aperture, kelvin).
    """
    try:
        if not raw_path:
            return None

        base, _ = os.path.splitext(raw_path)
        sidecar_path = base + ".xmp"

        # 1. Gather ideal EV from HolyGrail
        ideal_ev = None
        if hg_params is not None:
            ideal_ev = hg_params.get("ev_target", hg_params.get("ev", None))

        # 2. Gather actual settings
        settings = {}
        settings.update(hg_params or {})
        settings.update(camera_settings or {})

        iso = settings.get("iso", 100)
        shutter = settings.get("shutter_s", settings.get("shutter", 0.1))
        aperture = settings.get("aperture", 2.8)
        kelvin = settings.get("kelvin", None)

        shutter_s = parse_shutter_to_seconds(shutter)
        actual_ev = compute_ev(iso, shutter_s, aperture)

        # 3. Compute delta EV (ideal - actual), defaulting to 0
        if ideal_ev is not None and actual_ev is not None:
            exposure_offset_ev = ideal_ev - actual_ev
        else:
            exposure_offset_ev = 0.0

        xmp_content = _build_xmp_content(
            exposure_offset_ev=exposure_offset_ev,
            temperature=kelvin,
            extra_tags=extra_tags,
        )

        # 4. Write XMP to disk
        with open(sidecar_path, "w", encoding="utf-8") as f:
            f.write(xmp_content)

        return sidecar_path

    except Exception:
        # Fail-safe: don't crash capture pipeline because sidecar failed.
        return None