#!/usr/bin/env python3
"""
hardware.py — Unified Hardware Controller for PiSlider (Raspberry Pi 5)

SOURCE OF TRUTH: Matches wiring diagram rev 2026-02-17
─────────────────────────────────────────────────────
Pi Header (BCM) → Function
─────────────────────────────────────────────────────
GP2  (Pin  3) → Slider STEP
GP3  (Pin  5) → Slider DIR
GP4  (Pin  7) → Pan STEP
GP14 (Pin  8) → UART TX  (via 1kΩ resistor to TMC2209 RX bus)
GP9  (Pin 21) → UART RX
GP17 (Pin 11) → Pan DIR
GP27 (Pin 13) → Tilt STEP
GP22 (Pin 15) → Tilt DIR
GP10 (Pin 19) → ENABLE    (Active LOW)
GP9  (Pin 21) → Endstop   ← shared with UART RX; endstop reads are
                             only valid when UART is idle
GP11 (Pin 23) → (Unused / reserved)
GP8  (Pin 24) → Camera Trigger S2 (via optocoupler adaptor board)
GP6  (Pin 31) → Relay 1   (Ring Light)
GP12 (Pin 32) → Fan PWM
GP13 (Pin 33) → Aux Trigger / Relay 2 (Laser Projector)
─────────────────────────────────────────────────────
TMC2209 UART addressing (MS1/MS2 strap):
  Addr 0 → Slider
  Addr 1 → Pan
  Addr 2 → Tilt
─────────────────────────────────────────────────────
"""

import time
import struct
import logging
import serial
import crcmod
import lgpio

logger = logging.getLogger("PiSlider.HW")

# =============================================================================
# GPIO PINOUT — BCM numbers
# =============================================================================
PIN_SLIDER_STEP = 2
PIN_SLIDER_DIR  = 3
PIN_PAN_STEP    = 4
PIN_PAN_DIR     = 17
PIN_TILT_STEP   = 27
PIN_TILT_DIR    = 22

PIN_ENABLE      = 10   # Active LOW — pulls all TMC2209 EN pins
PIN_ENDSTOP     = 9    # Shared with UART RX; read only when UART idle
PIN_CAMERA      = 8    # S2 shutter trigger via optocoupler
PIN_RELAY1      = 6    # Relay 1 → Ring Light (macro mode)
PIN_FAN         = 12   # PWM fan control
PIN_AUX         = 13   # Relay 2 → Laser Projector (macro mode) / Aux

# =============================================================================
# TMC2209 UART
# =============================================================================
UART_PORT  = "/dev/ttyAMA0"   # Pi 5 hardware UART
BAUDRATE   = 115200
TMC_SYNC   = 0x05

# Register map (relevant subset)
REG_GCONF    = 0x00
REG_IHOLD_IRUN = 0x10
REG_TPOWERDOWN = 0x11
REG_CHOPCONF = 0x6C
REG_XACTUAL  = 0x21   # Hardware position odometer (read)
REG_VACTUAL  = 0x22   # Velocity command (write)
REG_TSTEP    = 0x12   # Time between steps (read)
REG_DRV_STATUS = 0x6F # Driver status flags (read)

# CRC8 for TMC2209 (poly 0x07, no reflection)
crc8_func = crcmod.mkCrcFun(0x107, initCrc=0x00, rev=False, xorOut=0x00)


class HardwareController:
    """
    Unified hardware interface for:
      • TMC2209 UART velocity/position control (cinematic + joystick)
      • Coordinated Bresenham pulse stepping (timelapse MSM)
      • Camera shutter trigger (S2 optocoupler)
      • Relay control (ring light, laser projector)
      • Fan PWM
      • Endstop reading
    """

    def __init__(self, gpio_chip_index: int = 0):
        self.gpio_chip_index = gpio_chip_index
        try:
            self.gpio_chip = lgpio.gpiochip_open(gpio_chip_index)
            self.uart = serial.Serial(UART_PORT, BAUDRATE, timeout=0.1)
            self._init_gpio()
            self.enable_motors(False)   # Safety: start disabled
            logger.info("HardwareController initialised. Pinout matches wiring rev 2026-02-17.")
        except lgpio.error as e:
            if 'busy' in str(e).lower():
                # GPIO lines still claimed by a previous process — force-free and retry
                logger.warning(f"GPIO busy on first attempt — freeing chip and retrying in 1s…")
                try:
                    lgpio.gpiochip_close(self.gpio_chip)
                except Exception:
                    pass
                import time as _t; _t.sleep(1.0)
                try:
                    self.gpio_chip = lgpio.gpiochip_open(gpio_chip_index)
                    self._init_gpio()
                    self.enable_motors(False)
                    logger.info("HardwareController initialised (after GPIO retry).")
                except Exception as e2:
                    logger.error(f"Hardware init failed after retry: {e2}")
                    raise
            else:
                logger.error(f"Hardware init failed: {e}")
                raise
        except Exception as e:
            logger.error(f"Hardware init failed: {e}")
            raise

    # -------------------------------------------------------------------------
    # GPIO INITIALISATION
    # -------------------------------------------------------------------------
    def _init_gpio(self):
        outputs = [
            PIN_SLIDER_STEP, PIN_SLIDER_DIR,
            PIN_PAN_STEP,    PIN_PAN_DIR,
            PIN_TILT_STEP,   PIN_TILT_DIR,
            PIN_ENABLE,
            PIN_CAMERA,
            PIN_RELAY1,
            PIN_FAN,
            PIN_AUX,
        ]
        for pin in outputs:
            lgpio.gpio_claim_output(self.gpio_chip, pin)
            lgpio.gpio_write(self.gpio_chip, pin, 0)

        lgpio.gpio_claim_input(self.gpio_chip, PIN_ENDSTOP)

    # -------------------------------------------------------------------------
    # TMC2209 UART — LOW LEVEL
    # -------------------------------------------------------------------------
    def _send_tmc_uart(self, addr: int, reg: int, data: int):
        """Write 32-bit value to TMC2209 register over single-wire UART."""
        packet = bytearray([
            TMC_SYNC,
            addr,
            reg | 0x80,          # Write flag
            (data >> 24) & 0xFF,
            (data >> 16) & 0xFF,
            (data >> 8)  & 0xFF,
            data         & 0xFF,
        ])
        packet.append(crc8_func(packet))
        self.uart.write(packet)
        self.uart.flush()
        self.uart.read(len(packet))   # Discard single-wire echo

    def _read_tmc_uart(self, addr: int, reg: int) -> int:
        """Read 32-bit value from TMC2209 register."""
        packet = bytearray([TMC_SYNC, addr, reg])
        packet.append(crc8_func(packet))

        self.uart.flushInput()
        self.uart.write(packet)
        self.uart.flush()
        self.uart.read(len(packet))   # Discard echo

        reply = self.uart.read(8)
        if len(reply) == 8 and reply[0] == TMC_SYNC:
            value = struct.unpack('>I', reply[4:8])[0]
            return value
        return 0

    # -------------------------------------------------------------------------
    # TMC2209 — HIGH LEVEL
    # -------------------------------------------------------------------------
    def set_tmc_velocity(self, addr: int, velocity: int):
        """
        Write VACTUAL. Positive = forward, negative = reverse.
        TMC2209 accepts a signed 24-bit integer.
        """
        if velocity < 0:
            velocity = (1 << 24) + velocity
        self._send_tmc_uart(addr, REG_VACTUAL, velocity)

    def get_tmc_position(self, addr: int) -> int:
        """Read XACTUAL (signed 32-bit microstep odometer)."""
        pos = self._read_tmc_uart(addr, REG_XACTUAL)
        if pos & 0x80000000:
            pos -= 0x100000000
        return pos

    def get_tmc_driver_status(self, addr: int) -> dict:
        """
        Read DRV_STATUS register and decode key fault flags.
        Useful for thermal monitoring and stall detection.
        """
        raw = self._read_tmc_uart(addr, REG_DRV_STATUS)
        return {
            "ot":      bool(raw & (1 << 1)),   # Overtemperature shutdown
            "otpw":    bool(raw & (1 << 0)),   # Overtemperature pre-warning
            "s2ga":    bool(raw & (1 << 27)),  # Short to GND phase A
            "s2gb":    bool(raw & (1 << 28)),  # Short to GND phase B
            "ola":     bool(raw & (1 << 29)),  # Open load phase A
            "olb":     bool(raw & (1 << 30)),  # Open load phase B
            "stst":    bool(raw & (1 << 31)),  # Standstill
            "raw":     raw,
        }

    def set_tmc_current(self, addr: int, run_current: int = 16, hold_current: int = 8):
        """
        Set motor run/hold current via IHOLD_IRUN.
        Values are 0–31 (TMC2209 rms scale).
        Default: run=16 (~50% of max), hold=8 (~25%).
        """
        ihold_irun = (hold_current & 0x1F) | ((run_current & 0x1F) << 8) | (3 << 16)
        self._send_tmc_uart(addr, REG_IHOLD_IRUN, ihold_irun)

    def init_tmc_drivers(self):
        """
        Send initial configuration to all three TMC2209 drivers.
        Call once after enable_motors(True).
        """
        for addr in (0, 1, 2):
            # GCONF: Enable UART control (PDN_DISABLE=1, MSTEP_REG_SELECT=1)
            self._send_tmc_uart(addr, REG_GCONF, 0b01000000_00000000_00000000_01000000)
            self.set_tmc_current(addr, run_current=16, hold_current=6)
            logger.info(f"TMC2209 addr={addr} initialised.")

    # -------------------------------------------------------------------------
    # COORDINATED BRESENHAM STEPPING (Move-Shoot-Move timelapse)
    # -------------------------------------------------------------------------
    def move_axes_simultaneous(
        self,
        slider_steps: int,
        pan_steps: int,
        tilt_steps: int,
        duration_s: float,
    ):
        """
        Pulse all three motors concurrently using Bresenham interpolation
        so they arrive at their targets at exactly the same time.

        Steps are signed (positive = forward direction per DIR pin logic).
        Duration controls overall speed — longer = slower.
        """
        lgpio.gpio_write(self.gpio_chip, PIN_SLIDER_DIR, 1 if slider_steps >= 0 else 0)
        lgpio.gpio_write(self.gpio_chip, PIN_PAN_DIR,    1 if pan_steps    >= 0 else 0)
        lgpio.gpio_write(self.gpio_chip, PIN_TILT_DIR,   1 if tilt_steps   >= 0 else 0)

        s_tgt = abs(slider_steps)
        p_tgt = abs(pan_steps)
        t_tgt = abs(tilt_steps)
        max_steps = max(s_tgt, p_tgt, t_tgt)

        if max_steps == 0:
            return

        step_delay = duration_s / max_steps
        s_err = p_err = t_err = 0

        for _ in range(max_steps):
            s_err += s_tgt
            if s_err >= max_steps:
                lgpio.gpio_write(self.gpio_chip, PIN_SLIDER_STEP, 1)
                s_err -= max_steps

            p_err += p_tgt
            if p_err >= max_steps:
                lgpio.gpio_write(self.gpio_chip, PIN_PAN_STEP, 1)
                p_err -= max_steps

            t_err += t_tgt
            if t_err >= max_steps:
                lgpio.gpio_write(self.gpio_chip, PIN_TILT_STEP, 1)
                t_err -= max_steps

            time.sleep(0.0001)   # ~100µs pulse width
            lgpio.gpio_write(self.gpio_chip, PIN_SLIDER_STEP, 0)
            lgpio.gpio_write(self.gpio_chip, PIN_PAN_STEP,    0)
            lgpio.gpio_write(self.gpio_chip, PIN_TILT_STEP,   0)

            time.sleep(max(0, step_delay - 0.0001))

    # -------------------------------------------------------------------------
    # PERIPHERALS
    # -------------------------------------------------------------------------
    def enable_motors(self, enable: bool):
        """Enable or disable all TMC2209 drivers (EN pin, active LOW)."""
        lgpio.gpio_write(self.gpio_chip, PIN_ENABLE, 0 if enable else 1)

    def trigger_camera(self, duration_s: float = 0.2):
        """Fire the S2 shutter trigger via optocoupler on GP8."""
        lgpio.gpio_write(self.gpio_chip, PIN_CAMERA, 1)
        time.sleep(duration_s)
        lgpio.gpio_write(self.gpio_chip, PIN_CAMERA, 0)

    def set_fan(self, percent: int):
        """Set cooling fan speed 0–100% via hardware PWM on GP12."""
        percent = max(0, min(100, int(percent)))
        lgpio.tx_pwm(self.gpio_chip, PIN_FAN, 100, percent)   # 100 Hz

    def set_relay1(self, on: bool):
        """
        Relay 1 (GP6) — Ring Light for macro/focus-stack mode.
        True = energise relay = light ON.
        """
        lgpio.gpio_write(self.gpio_chip, PIN_RELAY1, 1 if on else 0)

    def set_relay2(self, on: bool):
        """
        Relay 2 / Aux (GP13) — Laser Structured-Light Projector.
        True = energise relay = laser ON.
        """
        lgpio.gpio_write(self.gpio_chip, PIN_AUX, 1 if on else 0)

    def read_endstop(self) -> bool:
        """
        Read the endstop input (GP9).
        Returns True when triggered (active LOW — pulled high normally).
        Note: only valid when no UART transaction is in progress.
        """
        return lgpio.gpio_read(self.gpio_chip, PIN_ENDSTOP) == 0

    # -------------------------------------------------------------------------
    # MICROSTEPPING
    # -------------------------------------------------------------------------
    def set_microstepping(self, addr: int, mstep: int):
        """
        Set microstepping resolution for one TMC2209 driver via CHOPCONF.
        mstep: 256 | 128 | 64 | 32 | 16 | 8 | 4 | 2 | 1
        """
        mres_map = {256: 0, 128: 1, 64: 2, 32: 3,
                    16: 4,   8: 5,  4: 6,  2: 7, 1: 8}
        mres = mres_map.get(mstep, 4)  # default 1/16
        # Read-modify-write CHOPCONF: mask out bits [27:24], set new MRES
        base = 0x10000053          # stealthChop defaults
        chopconf = (base & ~(0xF << 24)) | (mres << 24)
        self._send_tmc_uart(addr, REG_CHOPCONF, chopconf)
        logger.info(f"TMC2209 addr={addr} microstepping set to 1/{mstep} (MRES={mres})")

    def set_mode_microstepping(self, mode: str):
        """Apply per-mode microstepping presets to all three drivers."""
        presets = {
            'cinematic':  8,    # max torque/speed
            'timelapse':  16,   # balance
            'macro':      256,  # finest resolution
        }
        mstep = presets.get(mode, 16)
        for addr in (0, 1, 2):
            self.set_microstepping(addr, mstep)
        logger.info(f"Mode microstepping: {mode} → 1/{mstep}")

    # -------------------------------------------------------------------------
    # SAFE SHUTDOWN
    # -------------------------------------------------------------------------
    def cleanup(self):
        """Gracefully stop all motion and release GPIO."""
        try:
            self.set_tmc_velocity(0, 0)
            self.set_tmc_velocity(1, 0)
            self.set_tmc_velocity(2, 0)
            self.enable_motors(False)
            self.set_relay1(False)
            self.set_relay2(False)
            self.set_fan(0)
        except Exception:
            pass
        try:
            self.uart.close()
        except Exception:
            pass
        try:
            lgpio.gpiochip_close(self.gpio_chip)
        except Exception:
            pass
        logger.info("HardwareController shutdown complete.")
