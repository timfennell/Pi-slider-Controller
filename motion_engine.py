#!/usr/bin/env python3
"""
motion_engine.py — Unified Cinematic Trajectory Generator for PiSlider.

Fuses:
1. Spatial Paths (Keyframes -> scipy.interpolate.CubicSpline)
2. Temporal Easing (Math curves from distributions.py)

Outputs frame-by-frame absolute position targets.
"""

import json
import logging
import numpy as np
from scipy.interpolate import CubicSpline
from distributions import CURVE_FUNCTIONS, normalize

def generate_time_array(total_frames: int, curve_name: str = "linear") -> np.ndarray:
    """
    Converts a mathematical curve from distributions.py into 
    a normalized temporal array (t) from 0.0 to 1.0.
    """
    if curve_name not in CURVE_FUNCTIONS:
        logging.warning(f"Curve '{curve_name}' not found. Defaulting to 'linear'.")
        curve_name = "linear"
        
    # 1. Get raw distribution weights
    raw_weights = CURVE_FUNCTIONS[curve_name](total_frames)
    
    # 2. Normalize so they sum exactly to 1.0
    normalized_weights = normalize(raw_weights)
    
    # 3. Cumulative sum turns intervals into a timeline (0.0 -> 1.0)
    t_array = np.cumsum(normalized_weights)
    
    # 4. Clamp start and end points to avoid float precision drift
    t_array = np.insert(t_array, 0, 0.0)[:-1] 
    t_array[-1] = 1.0
    
    return t_array


class MotionEngine:
    """
    Manages keyframes, tracking math, and trajectory generation.
    """
    def __init__(self):
        self.keyframes = []

    def add_keyframe(self, slider_mm: float, pan_deg: float, tilt_deg: float = 0.0):
        """Appends a spatial waypoint to the path."""
        self.keyframes.append({
            "slider_mm": slider_mm,
            "pan_deg": pan_deg,
            "tilt_deg": tilt_deg
        })

    def clear_keyframes(self):
        self.keyframes = []

    def generate_trajectory(self, duration_s: float, fps: int = 60, easing_curve: str = "linear"):
        """
        Fuses the spatial keyframes and temporal easing math into frame-by-frame targets.
        
        Returns:
            traj_slider, traj_pan, traj_tilt (np.ndarray of targets per frame)
        """
        if len(self.keyframes) < 2:
            raise ValueError("MotionEngine requires at least 2 keyframes (Start and End).")

        total_frames = int(duration_s * fps)
        
        # 1. Generate Temporal Timeline (The Speed Profile)
        t_normalized = generate_time_array(total_frames, easing_curve)
        
        # 2. Extract Spatial Data (The Path)
        slider_pts = [kf["slider_mm"] for kf in self.keyframes]
        pan_pts = [kf["pan_deg"] for kf in self.keyframes]
        tilt_pts = [kf["tilt_deg"] for kf in self.keyframes]
        
        # Map keyframes evenly across a 0.0 to 1.0 grid
        t_keyframes = np.linspace(0, 1, len(self.keyframes))
        
        # 3. Create the 3D Splines (clamped prevents overshoot past final keyframes)
        spline_slider = CubicSpline(t_keyframes, slider_pts, bc_type='clamped')
        spline_pan = CubicSpline(t_keyframes, pan_pts, bc_type='clamped')
        spline_tilt = CubicSpline(t_keyframes, tilt_pts, bc_type='clamped')
        
        # 4. Map the Speed Profile onto the Path
        traj_slider = spline_slider(t_normalized)
        traj_pan = spline_pan(t_normalized)
        traj_tilt = spline_tilt(t_normalized)
        
        return traj_slider, traj_pan, traj_tilt