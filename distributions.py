#!/usr/bin/env python3
"""
distributions.py — Motion distribution curves for PiSlider.

This module generates normalized distribution arrays that describe movement 
increments per interval. Arrays always sum to 1.0, allowing them to be 
scaled by total travel distance (mm or degrees).

Used for:
- Spatial easing in Cinematic Video (Splines).
- Step distribution in Move-Shoot-Move Timelapse.
- Object Tracking / Parallax triangulation logic.
"""

import numpy as np
import math

# ----------------------------------------------------------------------
# Utility: safe normalization
# ----------------------------------------------------------------------
def normalize(arr):
    """Normalize array so sum(arr) == 1.0."""
    if arr is None or len(arr) == 0:
        return np.array([])
    s = float(np.sum(arr))
    if s <= 0:
        # Fallback to uniform if array is zeros or negative
        return np.ones(len(arr)) / len(arr)
    return arr / s


# ----------------------------------------------------------------------
# Standard distribution curves (Parody of previous set)
# ----------------------------------------------------------------------

def catenary_distribution(n):
    """Hyperbolic cosine curve (Chain-like easing)."""
    if n <= 0: return np.array([])
    return np.cosh(np.linspace(-1.5, 1.5, n)) - 1

def inverted_catenary_distribution(n):
    s = catenary_distribution(n)
    return (np.max(s) if s.size else 0) - s

def gaussian_distribution(n):
    """Bell curve easing (Very smooth start and stop)."""
    if n <= 0: return np.array([])
    x = np.linspace(-2.5, 2.5, n)
    return np.exp(-x**2 / 2)

def inverted_gaussian_distribution(n):
    s = gaussian_distribution(n)
    return (np.max(s) if s.size else 0) - s

def ellipsoidal_distribution(n):
    if n <= 0: return np.array([])
    x = np.linspace(-1, 1, n)
    return np.sqrt(1 - x**2)

def inverted_ellipsoidal_distribution(n):
    s = ellipsoidal_distribution(n)
    return (np.max(s) if s.size else 0) - s

def parabolic_distribution(n):
    """Polynomial easing (Smooth but punchy acceleration)."""
    if n <= 0: return np.array([])
    x = np.linspace(-1, 1, n)
    return 1 - x**2

def inverted_parabolic_distribution(n):
    s = parabolic_distribution(n)
    return (np.max(s) if s.size else 0) - s

def cycloid_distribution(n):
    """Based on the path of a point on a rolling circle."""
    if n <= 0: return np.array([])
    return 1 - np.cos(np.linspace(0, 2 * np.pi, n))

def inverted_cycloid_distribution(n):
    s = cycloid_distribution(n)
    return (np.max(s) if s.size else 0) - s

def lame_curve_distribution(n, exponent=4):
    """Lamé curve / superellipse (Sharp square-like easing)."""
    if n <= 0: return np.array([])
    x = np.linspace(-1, 1, n)
    return (1 - np.abs(x)**exponent)**(1/exponent)

def inverted_lame_curve_distribution(n, exponent=4):
    s = lame_curve_distribution(n, exponent)
    return (np.max(s) if s.size else 0) - s

def linear_distribution(n):
    """Constant acceleration profile."""
    if n <= 0: return np.array([])
    return np.linspace(0.01, 1, n)

def inverted_linear_distribution(n):
    if n <= 0: return np.array([])
    return np.linspace(1, 0.01, n)

def even_distribution(n):
    """Constant speed profile (No easing)."""
    if n <= 0: return np.array([])
    return np.ones(n)


# ----------------------------------------------------------------------
# Specialized Feature: Object Tracking / Parallax logic
# ----------------------------------------------------------------------

def object_tracking_rotation(
    n,
    total_rotation_angle_deg,
    steps_distribution,
    belt_pitch_mm=2.0,
    pulley_teeth=20,
    microstepping=16,
    steps_per_rev=200,
):
    """
    Triangulation math: Computes the exact pan rotation increments needed 
    to keep a subject centered while the slider moves laterally according 
    to a provided distribution.
    
    This retains the '2-point vector triangulation' logic from your design.
    """
    if n <= 0 or steps_distribution.size != n:
        return np.zeros(n)

    if abs(total_rotation_angle_deg) < 1e-3:
        return np.zeros(n)

    # 1. Physical math: Slider mm movement per motor step
    dist_per_step_mm = (belt_pitch_mm * pulley_teeth) / (microstepping * steps_per_rev)

    # 2. Convert raw steps distribution -> Physical mm movement
    slider_mm_intervals = steps_distribution * dist_per_step_mm
    total_slider_mm = np.sum(slider_mm_intervals)

    # 3. Geometry: Calculate distance to subject based on total rotation requested
    # Trig: dist = (HalfSlider) / tan(HalfAngle)
    dist_to_obj_mm = (total_slider_mm / 2.0) / math.tan(
        math.radians(abs(total_rotation_angle_deg)) / 2.0
    )

    rotation_intervals = np.zeros(n)
    cum_slider = 0.0

    # 4. Iterative triangulation per interval
    prev_angle = math.atan2(-total_slider_mm / 2.0, dist_to_obj_mm)

    for i in range(n):
        cum_slider += slider_mm_intervals[i]
        x = cum_slider - (total_slider_mm / 2.0)
        curr_angle = math.atan2(x, dist_to_obj_mm)
        rotation_intervals[i] = math.degrees(curr_angle - prev_angle)
        prev_angle = curr_angle

    return rotation_intervals


# ----------------------------------------------------------------------
# Public Registry for Motion Engine & Web UI
# ----------------------------------------------------------------------
CURVE_FUNCTIONS = {
    "linear": linear_distribution,
    "inverted_linear": inverted_linear_distribution,
    "even": even_distribution,
    "parabolic": parabolic_distribution,
    "inverted_parabolic": inverted_parabolic_distribution,
    "gaussian": gaussian_distribution,
    "inverted_gaussian": inverted_gaussian_distribution,
    "catenary": catenary_distribution,
    "inverted_catenary": inverted_catenary_distribution,
    "ellipsoidal": ellipsoidal_distribution,
    "inverted_ellipsoidal": inverted_ellipsoidal_distribution,
    "cycloid": cycloid_distribution,
    "inverted_cycloid": inverted_cycloid_distribution,
    "lame": lame_curve_distribution,
    "inverted_lame": inverted_lame_curve_distribution,
}