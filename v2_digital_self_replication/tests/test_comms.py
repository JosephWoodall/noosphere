"""
Hardware communication layer tests.

Validates — without any physical hardware — that:
  1. Conversion math is correct and roundtrips cleanly
  2. Serial protocol bytes are exactly what the Pico expects
  3. Feedback parsing correctly recovers DOF position from angle reports
  4. StepperSim state machine (= Pico firmware logic) is correct
  5. FakeSerial end-to-end: bridge sends command → stepper moves → feedback recovered

Run:
    pytest v2_digital_self_replication/tests/test_comms.py -v
"""

from __future__ import annotations

import threading
import time
from io import BytesIO
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from v2_digital_self_replication.comms.arduino_bridge import (
    DOF_MOTOR_MAP,
    MOTOR_DOF_MAP,
    N_DOF,
    ArduinoBridge,
    _angle_to_dof,
    _dof_to_pwm,
)
from v2_digital_self_replication.pico.stepper import StepperSim


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Conversion math
# ═══════════════════════════════════════════════════════════════════════════════

class TestDofToPwm:
    def test_min(self):
        assert _dof_to_pwm(-1.0) == 0

    def test_max(self):
        assert _dof_to_pwm(1.0) == 255

    def test_zero(self):
        assert _dof_to_pwm(0.0) == 127

    def test_half_pos(self):
        assert _dof_to_pwm(0.5) == 191

    def test_half_neg(self):
        assert _dof_to_pwm(-0.5) == 63

    def test_clamp_below(self):
        assert _dof_to_pwm(-2.0) == 0

    def test_clamp_above(self):
        assert _dof_to_pwm(2.0) == 255

    def test_monotone(self):
        vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
        pwms = [_dof_to_pwm(v) for v in vals]
        assert pwms == sorted(pwms)


class TestAngleToDof:
    def test_min_angle(self):
        assert _angle_to_dof(0) == pytest.approx(-1.0)

    def test_max_angle(self):
        assert _angle_to_dof(180) == pytest.approx(1.0)

    def test_mid_angle(self):
        assert _angle_to_dof(90) == pytest.approx(0.0)

    def test_quarter_angle(self):
        assert _angle_to_dof(45) == pytest.approx(-0.5)

    def test_three_quarter_angle(self):
        assert _angle_to_dof(135) == pytest.approx(0.5)

    def test_monotone(self):
        angles = [0, 45, 90, 135, 180]
        dofs   = [_angle_to_dof(a) for a in angles]
        assert dofs == sorted(dofs)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. StepperSim state machine (= Pico firmware logic)
# ═══════════════════════════════════════════════════════════════════════════════

class TestStepperSimConversions:
    def test_pwm_min_to_target(self):
        sim = StepperSim(max_steps=512)
        assert sim._pwm_to_target(0) == -512

    def test_pwm_max_to_target(self):
        sim = StepperSim(max_steps=512)
        assert sim._pwm_to_target(255) == 512

    def test_pwm_mid_to_target_near_zero(self):
        sim = StepperSim(max_steps=512)
        # 128/255 is not exactly 0.5, so target is slightly off-centre — expected
        assert abs(sim._pwm_to_target(128)) <= 2

    def test_pos_min_to_angle(self):
        sim = StepperSim(max_steps=512)
        assert sim._pos_to_angle(-512) == 0

    def test_pos_max_to_angle(self):
        sim = StepperSim(max_steps=512)
        assert sim._pos_to_angle(512) == 180

    def test_pos_zero_to_angle(self):
        sim = StepperSim(max_steps=512)
        assert sim._pos_to_angle(0) == 90


class TestStepperSimMotion:
    def test_initial_state(self):
        sim = StepperSim()
        assert sim.current_pos == 0
        assert sim.target_pos  == 0
        assert sim.at_target()

    def test_tick_moves_toward_target(self):
        sim = StepperSim()
        sim.set_pwm(255)       # max forward
        moved = sim.tick()
        assert moved is True
        assert sim.current_pos == 1

    def test_tick_at_target_returns_false(self):
        sim = StepperSim()
        assert sim.tick() is False  # already at 0

    def test_run_to_target_reaches_target(self):
        sim = StepperSim(max_steps=64)
        sim.set_pwm(255)
        sim.run_to_target()
        assert sim.at_target()
        assert sim.current_pos == sim.target_pos

    def test_run_to_target_backward(self):
        sim = StepperSim(max_steps=64)
        sim.set_pwm(0)
        sim.run_to_target()
        assert sim.at_target()
        assert sim.current_pos == -64

    def test_centre_resets_target(self):
        sim = StepperSim()
        sim.set_pwm(255)
        sim.centre()
        assert sim.target_pos == 0

    def test_phase_advances_forward(self):
        sim = StepperSim()
        sim.set_pwm(200)
        p0 = sim.phase_idx
        sim.tick()
        assert sim.phase_idx == (p0 + 1) % 8

    def test_phase_retreats_backward(self):
        sim = StepperSim()
        sim.set_pwm(0)
        p0 = sim.phase_idx
        sim.tick()
        assert sim.phase_idx == (p0 - 1) % 8

    def test_phase_sequence_is_valid(self):
        sim = StepperSim()
        sim.set_pwm(255)
        phases_seen = set()
        for _ in range(8):
            phases_seen.add(tuple(sim.phase()))
            sim.tick()
        # All 8 half-step phases should appear
        assert len(phases_seen) == 8

    def test_phase_only_one_or_two_coils_energised(self):
        """Each half-step phase energises exactly 1 or 2 coils."""
        sim = StepperSim()
        sim.set_pwm(255)
        for _ in range(8):
            n_on = sum(sim.phase())
            assert n_on in (1, 2), f"Bad phase: {sim.phase()}"
            sim.tick()


class TestStepperRoundtrip:
    """
    Full DOF → PWM → stepper → angle → DOF roundtrip.
    Maximum allowed error is 2% — set by 8-bit PWM quantization.
    """

    @pytest.mark.parametrize("dof", [-1.0, -0.5, 0.0, 0.5, 1.0])
    def test_roundtrip_within_tolerance(self, dof):
        sim = StepperSim(max_steps=512)
        pwm = _dof_to_pwm(dof)
        sim.set_pwm(pwm)
        sim.run_to_target(max_ticks=2048)
        recovered = _angle_to_dof(sim.angle())
        assert abs(recovered - dof) < 0.02, (
            f"Roundtrip error at DOF={dof}: recovered={recovered:.4f}"
        )

    def test_roundtrip_consistency_with_bridge(self):
        """
        StepperSim._pwm_to_target and arduino_bridge._dof_to_pwm use compatible scales.
        Position at DOF extremes must be at stepper extremes.
        """
        sim = StepperSim(max_steps=512)
        sim.set_pwm(_dof_to_pwm(-1.0))
        assert sim.target_pos == -512

        sim.set_pwm(_dof_to_pwm(1.0))
        assert sim.target_pos == 512


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Serial protocol compliance
# ═══════════════════════════════════════════════════════════════════════════════

class RecordingSerial:
    """Captures bytes written to it; readline() always returns empty."""
    def __init__(self):
        self.written: list[bytes] = []

    def write(self, data: bytes):
        self.written.append(data)

    def readline(self) -> bytes:
        return b""

    def read_all(self) -> bytes:
        return b""

    def close(self):
        pass


class TestProtocol:
    def _bridge_with_recording_serial(self) -> tuple[ArduinoBridge, RecordingSerial]:
        bridge = ArduinoBridge()
        serial = RecordingSerial()
        bridge._serial = serial
        return bridge, serial

    def test_dispatch_writes_mapped_dofs_only(self):
        bridge, serial = self._bridge_with_recording_serial()
        bridge._dispatch({"cmd": [1.0] * N_DOF})
        written_lines = b"".join(serial.written).decode().splitlines()
        # Only DOF 0 and DOF 5 are mapped
        assert len(written_lines) == 2

    def test_dispatch_correct_format(self):
        bridge, serial = self._bridge_with_recording_serial()
        bridge._dispatch({"cmd": [1.0] * N_DOF})
        written = b"".join(serial.written).decode()
        assert "M0 255\n" in written   # DOF 0 → motor 0 → pwm 255
        assert "M1 255\n" in written   # DOF 5 → motor 1 → pwm 255

    def test_dispatch_min_dof(self):
        bridge, serial = self._bridge_with_recording_serial()
        cmd = [0.0] * N_DOF
        cmd[0] = -1.0
        cmd[5] = -1.0
        bridge._dispatch({"cmd": cmd})
        written = b"".join(serial.written).decode()
        assert "M0 0\n" in written
        assert "M1 0\n" in written

    def test_dispatch_halt_sends_centre(self):
        bridge, serial = self._bridge_with_recording_serial()
        bridge._dispatch({"halt": True, "cmd": [1.0] * N_DOF})
        written = b"".join(serial.written).decode()
        # halt overrides cmd — DOF 0.0 → pwm 127
        assert "M0 127\n" in written
        assert "M1 127\n" in written

    def test_dispatch_no_serial_dry_run(self):
        bridge = ArduinoBridge()
        bridge._serial = None
        # Must not raise
        bridge._dispatch({"cmd": [0.5] * N_DOF})

    def test_dispatch_midpoint_dof(self):
        bridge, serial = self._bridge_with_recording_serial()
        cmd = [0.0] * N_DOF
        bridge._dispatch({"cmd": cmd})
        written = b"".join(serial.written).decode()
        assert "M0 127\n" in written


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Feedback parsing
# ═══════════════════════════════════════════════════════════════════════════════

class TestFeedbackParsing:
    """
    Test that ArduinoBridge correctly parses 'P<idx> <angle>\\n' lines
    and maps them back to DOF positions.
    """

    def _run_feedback_one_line(self, line: str) -> np.ndarray:
        """
        Feed one feedback line through the bridge parsing logic.
        Returns the updated _position array.
        """
        bridge = ArduinoBridge()
        bridge._position = np.zeros(N_DOF, dtype=np.float32)
        bridge._running   = False   # stop after first read

        responses = [line.encode(), b""]

        class FakeSerial:
            def __init__(self):
                self._idx = 0
            def readline(self_inner):
                val = responses[self_inner._idx]
                self_inner._idx = min(self_inner._idx + 1, len(responses) - 1)
                return val

        bridge._serial = FakeSerial()

        # Replay the parsing logic from _feedback_loop
        raw = bridge._serial.readline().decode(errors="ignore").strip()
        if raw.startswith("P"):
            parts = raw[1:].split()
            if len(parts) == 2:
                motor_idx = int(parts[0])
                angle     = float(parts[1])
                dof_idx   = MOTOR_DOF_MAP.get(motor_idx)
                if dof_idx is not None:
                    with bridge._pos_lock:
                        bridge._position[dof_idx] = _angle_to_dof(angle)

        return bridge._position.copy()

    def test_midpoint_feedback(self):
        pos = self._run_feedback_one_line("P0 90")
        assert pos[0] == pytest.approx(0.0, abs=0.01)

    def test_min_feedback(self):
        pos = self._run_feedback_one_line("P0 0")
        assert pos[0] == pytest.approx(-1.0)

    def test_max_feedback(self):
        pos = self._run_feedback_one_line("P0 180")
        assert pos[0] == pytest.approx(1.0)

    def test_motor1_maps_to_dof5(self):
        pos = self._run_feedback_one_line("P1 90")
        assert pos[5] == pytest.approx(0.0, abs=0.01)
        assert pos[0] == pytest.approx(0.0)   # DOF 0 unchanged

    def test_unknown_motor_ignored(self):
        pos = self._run_feedback_one_line("P9 90")
        # No DOF should change
        assert np.all(pos == 0.0)

    def test_malformed_line_ignored(self):
        pos = self._run_feedback_one_line("GARBAGE")
        assert np.all(pos == 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. FakeSerial end-to-end integration
# ═══════════════════════════════════════════════════════════════════════════════

class FakeSerial:
    """
    Simulates the Pico over a synchronous serial interface.

    write() parses M commands and advances the StepperSim to target.
    readline() always returns the current stepper position (stateless read).

    ArduinoBridge calls write() once per motor, so buffering feedback per
    write() call produces ordering artifacts.  Returning current position
    on every readline() is simpler and correctly models the Pico's 50ms
    periodic report.
    """

    def __init__(self):
        self._sim = StepperSim(max_steps=512)

    def write(self, data: bytes):
        for raw_line in data.decode(errors="ignore").split("\n"):
            line = raw_line.strip()
            if not line.startswith("M"):
                continue
            parts = line[1:].split()
            if len(parts) != 2:
                continue
            try:
                motor_idx = int(parts[0])
                pwm       = max(0, min(255, int(parts[1])))
            except ValueError:
                continue
            if motor_idx == 0:
                self._sim.set_pwm(pwm)
        self._sim.run_to_target(max_ticks=2048)

    def readline(self) -> bytes:
        return f"P0 {self._sim.angle()}\n".encode()

    def read_all(self) -> bytes:
        return b""

    def close(self):
        pass

    @property
    def stepper(self) -> StepperSim:
        return self._sim


class TestFakeSerialIntegration:
    def test_command_moves_stepper_to_max(self):
        fs = FakeSerial()
        fs.write(b"M0 255\n")
        assert fs.stepper.current_pos == 512

    def test_command_moves_stepper_to_min(self):
        fs = FakeSerial()
        fs.write(b"M0 0\n")
        assert fs.stepper.current_pos == -512

    def test_command_moves_stepper_to_centre(self):
        fs = FakeSerial()
        fs.write(b"M0 128\n")
        assert abs(fs.stepper.current_pos) <= 2

    def test_readline_returns_feedback_after_command(self):
        fs = FakeSerial()
        fs.write(b"M0 255\n")
        line = fs.readline().decode().strip()
        assert line == "P0 180"

    def test_readline_min_feedback(self):
        fs = FakeSerial()
        fs.write(b"M0 0\n")
        line = fs.readline().decode().strip()
        assert line == "P0 0"

    def test_bridge_dispatch_then_feedback_roundtrip(self):
        """
        Full pipeline: ArduinoBridge._dispatch → FakeSerial → parse feedback → DOF.
        The recovered DOF must be within 2% of the commanded DOF.
        """
        for dof in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            fs     = FakeSerial()
            bridge = ArduinoBridge()
            bridge._serial   = fs
            bridge._position = np.zeros(N_DOF, dtype=np.float32)

            cmd = [0.0] * N_DOF
            cmd[0] = dof
            bridge._dispatch({"cmd": cmd})

            # Parse the feedback line the stepper generated
            raw = fs.readline().decode(errors="ignore").strip()
            assert raw.startswith("P"), f"No feedback for DOF={dof}"
            parts     = raw[1:].split()
            motor_idx = int(parts[0])
            angle     = float(parts[1])
            dof_idx   = MOTOR_DOF_MAP[motor_idx]
            with bridge._pos_lock:
                bridge._position[dof_idx] = _angle_to_dof(angle)

            recovered = float(bridge._position[0])
            assert abs(recovered - dof) < 0.02, (
                f"End-to-end roundtrip error at DOF={dof}: recovered={recovered:.4f}"
            )

    def test_halt_centres_motor(self):
        fs     = FakeSerial()
        bridge = ArduinoBridge()
        bridge._serial = fs

        bridge._dispatch({"cmd": [1.0] * N_DOF})                    # drive to max
        bridge._dispatch({"halt": True, "cmd": [1.0] * N_DOF})      # halt → centre
        line  = fs.readline().decode().strip()
        angle = float(line[1:].split()[1])
        assert 85 <= angle <= 95, f"Expected centre angle (~90), got {angle}"
