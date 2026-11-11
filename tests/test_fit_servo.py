"""fit_servo.py recovers known servo parameters from a synthetic run folder (no hardware)."""

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy import signal
import yaml

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("fit_servo", ROOT / "experiments/orca/fit_servo.py")
fit_servo = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fit_servo)

TAU, ACCEL, RATIO_DEG = 0.040, 25.0, 22.78 / 25.0 / (180 / math.pi)  # ratio chosen so ACC 150 predicts 25 rad/s^2


@pytest.fixture
def run_dir(tmp_path):
    config = yaml.safe_load((ROOT / "src/hand_control/orca/configs/orcahand_v2/right/config.yaml").read_text())
    t = np.arange(0, 20, 0.05)
    cmd = 30 + 10 * signal.chirp(t, f0=0.1, t1=20, f1=5, method="logarithmic")
    measured = fit_servo.simulate(np.deg2rad(cmd), 20.0, t, np.array([TAU]), np.array([ACCEL]), np.array([0.0]))[0]
    measured = np.rad2deg(measured) + np.random.default_rng(0).normal(0, 0.05, len(t))  # encoder noise
    name = "chirp_index_mcp_test.csv"
    np.savetxt(tmp_path / name, np.column_stack([t, cmd, measured]), delimiter=",",
               header="timestamp,commanded_position,measured_position", comments="")
    motor = abs(config["joint_to_motor_map"]["index_mcp"])
    (tmp_path / "calibration.yaml").write_text(yaml.safe_dump({"joint_to_motor_ratios": {motor: RATIO_DEG}}))
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))
    (tmp_path / "metadata.json").write_text(json.dumps(
        {"joints": ["index_mcp"], "csv_files": [name], "ctrl_freq_hz": 20.0,
         "servo_profile": {"servo_acceleration_raw": 150}}))
    return tmp_path


def test_recovers_dead_time_and_ramp(run_dir):
    assert fit_servo.main([str(run_dir), "--workers", "1"]) == 0
    result = json.loads((run_dir / "servo_fit.json").read_text())
    joint = result["joints"]["index_mcp"]
    assert joint["dead_time_ms"] == pytest.approx(1e3 * TAU, abs=3)
    assert joint["accel_rad_s2"] == pytest.approx(ACCEL, rel=0.03)
    assert joint["accel_predicted_rad_s2"] == pytest.approx(ACCEL, rel=0.01)
    assert joint["rms_deg"] < 0.1 < joint["linear"]["rms_deg"]
    assert result["flagged"] == [] and joint["clipped"] is False
    assert result["sim"]["ORCA_V2_SERVO_ACCEL"]["^right_(index|middle|ring|pinky)_mcp$"] == pytest.approx(ACCEL, rel=0.03)
    assert result["sim"]["dead_time_plus_lag_ms"]["fingers_thumb"] == pytest.approx(1e3 * TAU, abs=3)


def test_clipped_joint_is_flagged(run_dir):
    """A joint stopped short of the command by a neighbour is flagged and left out of the sim values."""
    csv_path = run_dir / "chirp_index_mcp_test.csv"
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    data[:, 2] = np.minimum(data[:, 2], 35.0)  # the command peaks at 40 deg
    np.savetxt(csv_path, data, delimiter=",", header="timestamp,commanded_position,measured_position", comments="")
    fit_servo.main([str(run_dir), "--workers", "1"])
    result = json.loads((run_dir / "servo_fit.json").read_text())
    assert result["joints"]["index_mcp"]["clipped"] is True
    assert result["flagged"] == ["index_mcp"]
