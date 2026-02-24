#!/usr/bin/env python3
"""
slider.py — High-level motion control for PiSlider.

Contains:
1. LinearAxis & RotationAxis (High-level trackers for discrete timelapse moves)
2. TrajectoryPlayer (Executes motion_engine splines at 60fps via UART)
"""

import time
import logging

class TrajectoryPlayer:
    """
    Executes a high-speed cinematic motion trajectory by streaming 
    velocity commands to the TMC2209 drivers via UART.
    """
    def __init__(self, hardware, steps_per_mm: float = 100.0, steps_per_deg: float = 55.5):
        self.hardware = hardware
        self.steps_per_mm = steps_per_mm
        self.steps_per_deg = steps_per_deg

    def play(self, traj_slider, traj_pan, traj_tilt, fps: int = 60):
        """
        Plays back spatial arrays smoothly.
        """
        total_frames = len(traj_slider)
        if total_frames < 2:
            logging.error("Trajectory too short to play.")
            return

        dt = 1.0 / fps
        logging.info(f"Playing Trajectory: {total_frames} frames at {fps} FPS.")

        self.hardware.enable_motors(True)
        start_time = time.perf_counter()

        for i in range(total_frames - 1):
            # 1. Calculate physical delta for this frame
            delta_slider = traj_slider[i+1] - traj_slider[i]
            delta_pan = traj_pan[i+1] - traj_pan[i]
            delta_tilt = traj_tilt[i+1] - traj_tilt[i]

            # 2. Convert to velocity (Units per second)
            v_slider_units_s = delta_slider / dt
            v_pan_units_s = delta_pan / dt
            v_tilt_units_s = delta_tilt / dt

            # 3. Convert physical velocity to Motor Steps/sec
            v_slider_steps = int(v_slider_units_s * self.steps_per_mm)
            v_pan_steps = int(v_pan_units_s * self.steps_per_deg)
            v_tilt_steps = int(v_tilt_units_s * self.steps_per_deg)

            # 4. Stream to Hardware
            self.hardware.set_tmc_velocity(addr=0, velocity=v_slider_steps)
            self.hardware.set_tmc_velocity(addr=1, velocity=v_pan_steps)
            self.hardware.set_tmc_velocity(addr=2, velocity=v_tilt_steps)

            # 5. Precision Timing Engine (Avoids Linux OS Jitter)
            next_frame_time = start_time + ((i + 1) * dt)
            
            # Sleep to free CPU, but wake up 2ms early
            sleep_time = next_frame_time - time.perf_counter() - 0.002 
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            # Microsecond-perfect lock for the end of the frame
            while time.perf_counter() < next_frame_time:
                pass

        # 6. Stop perfectly at the end of the trajectory
        self.hardware.set_tmc_velocity(0, 0)
        self.hardware.set_tmc_velocity(1, 0)
        self.hardware.set_tmc_velocity(2, 0)
        logging.info("Playback complete.")


class LinearAxis:
    """
    High-level wrapper for the slider motor (used for discrete timelapse moves).
    """
    def __init__(self, hardware, steps_per_mm=100.0, max_mm=300.0, settle_time=0.5):
        self.hw = hardware
        self.steps_per_mm = steps_per_mm
        self.max_mm = max_mm
        self.settle_time = settle_time
        self.current_mm = 0.0

    def move_to_mm(self, target_mm: float, duration_s: float = 1.0) -> bool:
        target_mm = max(0.0, min(target_mm, self.max_mm))
        delta_mm = target_mm - self.current_mm
        steps = int(delta_mm * self.steps_per_mm)
        
        # Uses the timelapse Bresenham loop for discrete movement
        self.hw.move_axes_simultaneous(slider_steps=steps, pan_steps=0, tilt_steps=0, duration_s=duration_s)
        self.current_mm = target_mm
        time.sleep(self.settle_time)
        return True

    def get_position_mm(self) -> float:
        return self.current_mm


class RotationAxis:
    """
    High-level wrapper for rotation (pan/tilt) motors.
    """
    def __init__(self, hardware, addr=1, steps_per_deg=55.5, settle_time=0.5):
        self.hw = hardware
        self.addr = addr
        self.steps_per_deg = steps_per_deg
        self.settle_time = settle_time
        self.current_deg = 0.0

    def move_to_deg(self, target_deg: float, duration_s: float = 1.0) -> bool:
        delta_deg = target_deg - self.current_deg
        steps = int(delta_deg * self.steps_per_deg)
        
        if self.addr == 1: # Pan
            self.hw.move_axes_simultaneous(0, steps, 0, duration_s)
        elif self.addr == 2: # Tilt
            self.hw.move_axes_simultaneous(0, 0, steps, duration_s)
            
        self.current_deg = target_deg
        time.sleep(self.settle_time)
        return True

    def get_position_deg(self) -> float:
        return self.current_deg

    def sweep_degrees(self, start_deg: float, end_deg: float, frames: int):
        if frames < 2:
            frames = 2
        positions = []
        step_deg = (end_deg - start_deg) / (frames - 1)
        for i in range(frames):
            positions.append(start_deg + i * step_deg)
        return positions