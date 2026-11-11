"""Offline tests of the real-hand evaluation's parts without ROS, camera or hand."""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/rl_inference"))

from cube_eval.metrics import joint_tracking, status_names, tracking_summary  # noqa: E402
from cube_eval.thermal import DEFAULT_RATE, TemperatureFilter, ThermalModel  # noqa: E402
from cube_eval.turns import CUBE_SYMMETRIES, DropMonitor, RestDetector, TurnCounter, lateral_offset  # noqa: E402

UP = np.array([0.0, 0.0, -1.0])


def spin(counter, rate, seconds, hz, rng=None, relabel=False, t0=0.0, base=Rotation.identity()):
    """Feed a cube turning at rate (rad/s) about UP; optionally with a random symmetric relabelling per frame."""
    for i in range(int(seconds * hz) + 1):
        t = i / hz
        rot = Rotation.from_rotvec(UP * rate * t) * base
        if relabel:
            rot = rot * CUBE_SYMMETRIES[int(rng.integers(24))]
        counter.add(t0 + t, rot)


@pytest.mark.parametrize("rate", [2.0, -2.0, 0.7])
def test_turns_at_detector_rate_with_symmetric_relabelling(rate):
    counter = TurnCounter(up=UP)
    spin(counter, rate, 60.0, 5.0, np.random.default_rng(0), relabel=True,
         base=Rotation.from_euler("xyz", [0.3, -0.2, 1.0]))
    assert counter.turns == pytest.approx(rate * 60 / (2 * np.pi), abs=0.01)
    assert counter.rejected == 0
    assert counter.speed(60.0, window=5.0) == pytest.approx(rate, abs=0.02)


def test_wrong_detections_are_rejected_and_do_not_count():
    counter = TurnCounter(up=UP)
    rng = np.random.default_rng(1)
    for i in range(301):
        t = i / 10
        rot = Rotation.from_rotvec(UP * 1.0 * t)
        if i % 17 == 5:  # a flipped pose
            rot = Rotation.from_rotvec(rng.normal(size=3)) * rot
        counter.add(t, rot)
    assert counter.turns == pytest.approx(30 / (2 * np.pi), abs=0.05)
    assert 0 < counter.rejected_share < 0.1


def test_a_long_gap_is_not_counted_but_reported():
    counter = TurnCounter(up=UP, max_gap=0.6)
    spin(counter, 2.0, 5.0, 10.0)
    spin(counter, 2.0, 5.0, 10.0, t0=7.0)  # nothing seen for 2 s
    assert counter.turns == pytest.approx(20 / (2 * np.pi), abs=0.01)
    assert counter.uncounted_s == pytest.approx(2.0)


def test_fast_spin_beyond_the_plain_symmetry_limit():
    counter = TurnCounter(up=UP)  # 60 deg per detection: only the prediction keeps this countable
    for i in range(200):
        t = i * 0.2
        counter.add(t, Rotation.from_rotvec(UP * (0.5 * min(t, 5.2) ** 2 + max(0.0, t - 5.2) * 5.2)))
    expected = 0.5 * 5.2 ** 2 + (199 * 0.2 - 5.2) * 5.2
    assert counter.angle == pytest.approx(expected, abs=0.05)


def test_rebase_ignores_what_happened_during_a_pause():
    counter = TurnCounter(up=UP)
    spin(counter, 1.0, 10.0, 10.0)
    counter.rebase()
    spin(counter, 1.0, 10.0, 10.0, t0=10.0, base=Rotation.from_rotvec(UP * 0.6))  # put back turned by 34 deg
    assert counter.turns == pytest.approx(20 / (2 * np.pi), abs=0.01)


def test_tilting_is_not_counted_as_turning():
    counter = TurnCounter(up=UP)
    for i in range(100):
        counter.add(i / 10, Rotation.from_rotvec([0.02 * i, 0, 0]))
    assert abs(counter.turns) < 0.01
    assert counter.tilt_angle == pytest.approx(1.98, abs=0.01)


def test_lateral_offset_ignores_depth_error():
    home = np.array([0.02, -0.01, 0.40])
    assert lateral_offset(home * 1.2, home) == pytest.approx(0.0, abs=1e-9)  # same pixel, wrong depth
    assert lateral_offset(home + [0.05, 0, 0], home) == pytest.approx(0.05)


def test_rest_detector():
    rest = RestDetector(window=1.5)
    for i in range(20):
        rest.add(i * 0.1, [0.01 + 0.001 * (i % 2), 0.0, 0.4])
    assert rest.resting(2.0) is not None
    assert rest.resting(4.0) is None  # not seen any more
    moving = RestDetector(window=1.5)
    for i in range(20):
        moving.add(i * 0.1, [0.004 * i, 0.0, 0.4])
    assert moving.resting(2.0) is None


def test_drop_monitor_dropped_lost_and_outliers():
    home = np.array([0.0, 0.0, 0.4])
    drop = DropMonitor(home, drop_dist=0.06, confirm=0.5, lost_timeout=3.0)
    drop.start(0.0)
    t = 0.0
    for i in range(50):  # in the hand, with one wild detection
        t = i * 0.1
        drop.add(t, home + ([0.3, 0, 0] if i == 20 else [0.01, 0, 0]))
        assert drop.check(t) == ""
    for i in range(4):
        t += 0.1
        drop.add(t, home + [0.10, 0, 0])
    assert drop.check(t) == ""  # not confirmed yet
    for i in range(6):
        t += 0.1
        drop.add(t, home + [0.10, 0, 0])
    assert drop.check(t) == "dropped"

    lost = DropMonitor(home)
    lost.start(0.0)
    lost.add(0.1, home)
    assert lost.check(2.9) == ""
    assert lost.check(3.2) == "lost"


def test_thermal_gate_default_and_learned(tmp_path):
    joints = ["a", "b"]
    model = ThermalModel(joints, limit=60.0, margin=2.0, history_path=tmp_path / "thermal.json")
    assert model.start_below(60.0) == pytest.approx([58 - 60 * DEFAULT_RATE] * 2)
    assert model.ready([42.0, 30.0], 60.0)
    assert not model.ready([44.0, 30.0], 60.0)
    assert model.ready([44.0, 30.0], 30.0)  # less time left
    assert not model.ready([np.nan, 30.0], 60.0)
    assert model.over_limit([59.0, 60.0]) == 1 and model.over_limit([59.0, 59.9]) is None

    for _ in range(3):
        model.learn([(60.0, [30.0, 30.0], [36.0, 51.0]), (5.0, [30.0, 30.0], [40.0, 40.0])])  # short one ignored
    assert model.rates() == pytest.approx([0.1, 0.35])
    assert model.start_below(60.0) == pytest.approx([52.0, 37.0])
    reloaded = ThermalModel(joints, history_path=tmp_path / "thermal.json")
    assert reloaded.rates() == pytest.approx([0.1, 0.35])


def test_temperature_filter_rejects_glitches_but_follows_real_changes():
    f = TemperatureFilter()
    f.update([30.0, 30.0])
    assert f.update([150.0, 31.0]) == pytest.approx([30.0, 31.0])
    assert f.update([58.0, 31.0]) == pytest.approx([30.0, 31.0])   # a jump, once
    assert f.update([30.0, 31.0]) == pytest.approx([30.0, 31.0])
    assert f.glitches == 2
    f.update([45.0, 31.0])
    assert f.update([45.0, 31.0]) == pytest.approx([45.0, 31.0])   # confirmed by the next reading


def test_joint_tracking_finds_delay_and_error():
    t = np.arange(0, 20, 0.01)
    cmd = np.stack([np.radians(10) * np.sin(2 * np.pi * 0.5 * t), np.zeros_like(t)], axis=1)
    meas = np.stack([np.radians(10) * np.sin(2 * np.pi * 0.5 * (t - 0.05)), np.full_like(t, np.radians(1.0))], axis=1)
    rows = joint_tracking(t, cmd, meas, np.full((len(t), 2), 200.0), ["moving", "still"])
    assert rows[0]["delay_ms"] == 50 and rows[0]["error_rms_deg"] < 0.1
    assert rows[1]["delay_ms"] == "" and rows[1]["error_rms_deg"] == pytest.approx(1.0)
    summary = tracking_summary(rows)
    assert summary["worst_joint"] == "still" and summary["current_total_mean_a"] == pytest.approx(0.4)
    assert status_names(0x24) == ["overheating", "overload"]
