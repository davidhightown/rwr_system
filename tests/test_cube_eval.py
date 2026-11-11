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
    counter = TurnCounter(up=UP, symmetric=True)
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


@pytest.mark.parametrize("rate", [0.55, 2.0, 0.0])
@pytest.mark.parametrize("outlier_deg, tolerance", [(10, 0.12), (50, 0.5)])
def test_noisy_poses_of_a_lettered_cube_do_not_drift(rate, outlier_deg, tolerance):
    """As on the hand: 5 detections per second and 10 deg of noise on every axis of every pose; in the second
    case every 15th pose is far off (much more often than on the hand)."""
    for seed in range(4):
        rng = np.random.default_rng(seed)
        counter = TurnCounter(up=UP)
        for i in range(301):
            t = i * 0.2
            noise = Rotation.from_rotvec(rng.normal(size=3) * np.radians(10 if i % 15 else outlier_deg))
            counter.add(t, noise * Rotation.from_rotvec(UP * rate * t))
        assert counter.turns == pytest.approx(rate * 60 / (2 * np.pi), abs=tolerance)


def test_a_gap_is_not_counted_for_a_symmetric_cube():
    counter = TurnCounter(up=UP, symmetric=True, max_gap=0.6)
    spin(counter, 2.0, 5.0, 10.0)
    spin(counter, 2.0, 5.0, 10.0, t0=7.0)  # nothing seen for 2 s
    assert counter.turns == pytest.approx(20 / (2 * np.pi), abs=0.01)
    assert counter.gap_s == pytest.approx(2.0)


def test_a_lettered_cube_is_counted_across_a_gap():
    counter = TurnCounter(up=UP, max_gap=0.6)
    for t in list(np.arange(0, 5.01, 0.1)) + list(np.arange(6.0, 10.01, 0.1)):  # nothing seen for 1 s: 115 deg
        counter.add(t, Rotation.from_rotvec(UP * 2.0 * t))
    assert counter.turns == pytest.approx(20 / (2 * np.pi), abs=0.01)
    assert counter.gap_s == pytest.approx(1.0)


def test_fast_spin_beyond_the_plain_symmetry_limit():
    counter = TurnCounter(up=UP, symmetric=True)  # 60 deg per detection: only the prediction keeps this countable
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


# -- W&B (no W&B needed: the records are plain dicts as Project.records() returns them)
from cube_eval import wandb_results as wr  # noqa: E402
from cube_eval.metrics import progress_metrics  # noqa: E402


def test_progress_metrics_as_the_other_evaluation():
    t = np.arange(0.0, 60.0, 0.2)
    turns = np.where(t < 20, t * 0.1, np.where(t < 30, 2.0, 2.0 + (t - 30) * 0.2))  # a 10 s stall at 20-30 s
    m = progress_metrics(t, turns, 60.0)
    assert m["turns_30s"] == pytest.approx(2.0)
    assert m["best_10s_rad_per_s"] == pytest.approx(0.2 * 2 * np.pi, abs=0.02)
    assert 10.0 <= m["longest_stall_s"] <= 11.5
    assert progress_metrics([1.0], [0.1], 8.0)["turns_30s"] == ""


def record(name, short, sha="", goal=1.5, own=1.5, start=40.0, turns=5.0, evaluator=None, started="2026-11-10 10:00",
           completed=True, idx=48):
    e = dict(goal_rad_s=goal, own_goal_rad_s=own, at_own_goal=goal == own, cold_start=start <= 42.5, started=started,
             start_temp_max_c=start)
    if evaluator:
        e["evaluator"] = evaluator
    return dict(id=f"{name}-{started}", job_type="real_hand",
                config=dict(model=dict(name=name, short_name=short, idx=idx, description=f"about {short}",
                                       policy_sha256=sha),
                            train=dict(run="rwr-skin/x/abc", url="https://x/abc", run_name=short), sim={}, eval=e),
                summary={"result/turns": turns, "result/duration_s": 60.0, "result/full_60s": completed})


def test_which_models_are_tested(tmp_path):
    for name in ("ibex", "lark", "newt", "copy"):
        (tmp_path / f"{name}_policy.pt").write_bytes(name.encode())
    (tmp_path / "policy.pt").write_bytes(b"generic")  # not a named model: --auto leaves it out
    records = [record("48_ibex", "ibex", sha="other"), record("Newt", "Newt"),
               record("60_x", "x", sha=wr.sha256(tmp_path / "copy_policy.pt")),
               dict(id="leaderboard", job_type="leaderboard", config={}, summary={})]
    known = wr.known_models(records)
    todo, done = wr.split_models(tmp_path, known)
    assert [p.name for p in todo] == ["lark_policy.pt"]
    why = {p.name: w for p, w in done}
    assert why["ibex_policy.pt"] == "48_ibex has 1 run on W&B"          # by name, although another file
    assert why["newt_policy.pt"].startswith("Newt")                        # any case
    assert why["copy_policy.pt"].startswith("same file as 60_x")           # by file, although another name


def meta_of(model="ibex", goal=2.0, outcome="completed", drops=0, start=30.0):
    return dict(model=model, time="2026-11-11 22:15:25", policy_file=f"models/{model}_policy.pt",
                policy_sha256="abc", policy_sha256_full="abcdef", host="mac", goal_rad_s=goal, outcome=outcome,
                active_s=60.0, turns=4.2, turns_before_first_drop=1.5, drops=drops, temp_max_start_c=start,
                heating_max_c_per_min="", servo_errors="", git_commit="1d18a28+changes", hand_config_sha256="h",
                options=dict(duration="60.0", start_wrist_deg="33.0", temp_limit="60.0"))


def test_config_and_summary_in_the_other_evaluations_format(tmp_path):
    known = wr.known_models([record("48_ibex", "ibex", own=1.5)])
    run_dir = tmp_path / "real_eval" / "20261111_221525_ibex"
    c = wr.config_of(meta_of(goal=2.0), run_dir, known)
    assert c["model"]["name"] == "48_ibex" and c["model"]["idx"] == 48 and c["model"]["short_name"] == "ibex"
    assert c["train"]["run"] == "rwr-skin/x/abc"
    assert c["eval"]["evaluator"] == "eval_policies.py" and c["eval"]["reference"]
    assert c["eval"]["at_own_goal"] is False and c["eval"]["cold_start"] is True
    assert c["eval"]["start_wrist_deg"] == 33.0 and c["eval"]["session"] == "real_eval"
    assert c["code"] == dict(commit="1d18a28", uncommitted_changes=True, hand_config_sha256="h")
    new = wr.config_of(meta_of(model="lark"), run_dir, known)
    assert new["model"]["name"] == "lark" and new["model"]["idx"] is None and new["eval"]["at_own_goal"] is True

    s = wr.summary_of(meta_of(drops=2))
    assert s["result/turns"] == 4.2 and s["cube/turns_camera_axis"] == 4.2 and s["result/stop_reason"] == "time"
    assert s["result/dropped"] is True and s["result/full_60s"] is False
    assert "hand/heating_max_c_per_min" not in s and s["hand/servo_errors"] == ""
    assert wr.summary_of(meta_of())["result/full_60s"] is True
    assert wr.run_id(meta_of(), run_dir) == wr.run_id(meta_of(), tmp_path / "moved" / run_dir.name)


def test_leaderboard_per_model_and_evaluator():
    records = [record("48_ibex", "ibex", turns=13.0, started="a"), record("48_ibex", "ibex", turns=11.0, started="b"),
               record("48_ibex", "ibex", goal=2.2, turns=15.0),                      # not at its own goal
               record("48_ibex", "ibex", start=50.0, turns=1.0),                     # not a cold start
               record("48_ibex", "ibex", turns=4.0, evaluator="eval_policies.py", completed=False),
               record("lark", "lark", own=None, goal=2.0, turns=6.0, evaluator="eval_policies.py")]
    records[-1]["config"]["eval"]["at_own_goal"] = True
    rows = wr.leaderboard_rows(records)
    assert [(r["model"], r["evaluator"]) for r in rows] == [
        ("48_ibex", "evaluate_real_policies.py"), ("lark", "eval_policies.py"), ("48_ibex", "eval_policies.py")]
    assert rows[0]["runs"] == 2 and rows[0]["mean_turns"] == 12.0 and rows[0]["each_run"] == "13.0 11.0"
    assert rows[2]["full_60s_runs"] == 0 and rows[2]["std_turns"] is None
    assert list(rows[0])[:11] == ["model", "description", "runs", "mean_turns", "std_turns", "best_turns",
                                  "worst_turns", "full_60s_runs", "mean_duration_s", "each_run", "training_url"]


def test_curves_over_policy_time(tmp_path):
    (tmp_path / "turns.csv").write_text("policy_time_s,turns\n0.5,0.1\n1.2,0.3\n")
    (tmp_path / "temperatures.csv").write_text("time_s,policy_time_s,state,a,b\n0.5,0.0,running,30,35\n"
                                               "5.0,0.0,cooling,50,50\n6.0,2.0,running,31,37\n")
    (tmp_path / "joints.csv").write_text("policy_time_s,current_ma_a,current_ma_b\n0.0,100,-200\n1.0,300,300\n")
    points = wr.curves(tmp_path, 2.0)
    assert [p["time_s"] for p in points] == [0.0, 0.5, 1.0, 1.5, 2.0]
    assert [p["curve/turns"] for p in points] == [0.0, 0.1, 0.1, 0.3, 0.3]
    assert points[0]["curve/max_temp_c"] == 35.0 and points[-1]["curve/max_temp_c"] == 37.0  # cooling left out
    assert points[0]["curve/total_current_a"] == pytest.approx(0.3) and "curve/total_current_a" not in points[1]


def test_auto_arguments():
    import eval_policies as ep
    args = ep.parse_args(["--auto", "--dry-run"])
    assert args.wandb and args.policies == [] and args.models_dir == ROOT / "models"
    with pytest.raises(SystemExit):
        ep.parse_args(["--auto", str(ROOT / "models/crow_policy.pt")])
    with pytest.raises(SystemExit):
        ep.parse_args(["--dry-run", str(ROOT / "models/crow_policy.pt")])
