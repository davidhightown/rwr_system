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
                active_s=60.0, turns=4.2, turns_before_first_drop=1.5 if drops else 4.2,
                duration_before_first_drop_s=20.0 if drops else 60.0, drops=drops, temp_max_start_c=start,
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
    assert new["model"]["name"] == "lark" and new["model"]["idx"] is None
    assert new["eval"]["own_goal_rad_s"] == 2.0 and new["eval"]["at_own_goal"] is True  # its first runs set its goal

    s = wr.summary_of(meta_of(drops=2))  # as the other evaluation: up to the first drop
    assert s["result/turns"] == 1.5 and s["result/duration_s"] == 20.0 and s["cube/turns_camera_axis"] == 1.5
    assert s["result/turns_total"] == 4.2 and s["result/duration_total_s"] == 60.0 and s["result/stop_reason"] == "time"
    assert s["result/dropped"] is True and s["result/full_60s"] is False
    assert "hand/heating_max_c_per_min" not in s and s["hand/servo_errors"] == ""
    assert wr.summary_of(meta_of())["result/full_60s"] is True
    assert wr.run_id(meta_of(), run_dir) == wr.run_id(meta_of(), tmp_path / "moved" / run_dir.name)


def test_leaderboard_per_model_over_both_evaluations():
    records = [record("48_ibex", "ibex", turns=13.0, started="a"), record("48_ibex", "ibex", turns=11.0, started="b"),
               record("48_ibex", "ibex", goal=2.2, turns=15.0),                      # not at its own goal
               record("48_ibex", "ibex", start=50.0, turns=1.0),                     # not a cold start
               record("48_ibex", "ibex", turns=6.0, evaluator="eval_policies.py", started="c", completed=False),
               record("lark", "lark", goal=2.0, own=2.0, turns=8.0, evaluator="eval_policies.py")]
    rows = wr.leaderboard_rows(records)
    assert [r["model"] for r in rows] == ["48_ibex", "lark"]
    assert rows[0]["runs"] == 3 and rows[0]["mean_turns"] == 10.0 and rows[0]["each_run"] == "13.0 11.0 6.0"
    assert rows[0]["full_60s_runs"] == 2 and rows[1]["std_turns"] is None
    assert list(rows[0]) == ["model", "description", "runs", "mean_turns", "std_turns", "best_turns", "worst_turns",
                             "full_60s_runs", "mean_duration_s", "each_run", "training_url"]


def test_goal_speed_from_wandb(tmp_path):
    for name in ("ibex", "lark"):
        (tmp_path / f"{name}_policy.pt").write_bytes(name.encode())
    known = wr.known_models([record("48_ibex", "ibex", goal=1.5, own=1.5)])
    assert wr.goal_speed(known, tmp_path / "ibex_policy.pt") == (1.5, "goal speed of 48_ibex on W&B")
    assert wr.goal_speed(known, tmp_path / "lark_policy.pt")[0] == wr.DEFAULT_SPEED
    assert wr.goal_speed(known, tmp_path / "ibex_policy.pt", speed=-2.5) == (-2.5, "--speed")


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


def test_auto_arguments(tmp_path):
    import eval_policies as ep
    crow = tmp_path / "crow_policy.pt"
    crow.write_bytes(b"crow")
    args = ep.parse_args(["--auto", "--dry-run"])
    assert args.wandb and args.policies == [] and args.models_dir == ROOT / "models_proprio"
    args = ep.parse_args([])  # no policy files: the untested models
    assert args.auto and args.wandb
    args = ep.parse_args(["--models-dir", str(tmp_path), "crow", "crow_policy.pt"])
    assert args.policies == [crow] * 2 and not args.auto
    with pytest.raises(SystemExit):
        ep.parse_args(["--auto", str(crow)])
    with pytest.raises(SystemExit):
        ep.parse_args(["--dry-run", str(crow)])


# -- policies with the cube pose (task orca-v2-inhand-repose-extero-v0)
def pose(position, rotation=Rotation.identity()):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = rotation.as_matrix(), position
    return T


def linear_policy(path, inputs):
    torch = pytest.importorskip("torch")
    torch.jit.script(torch.nn.Linear(inputs, 17)).save(str(path))
    return path


def test_cube_pose_is_held_with_the_flag_0_when_lost():
    from cube_spin_policy import OBS_T_HAND, CubePose
    cube = CubePose(timeout_s=0.25)
    assert cube.observation(0.0) is None
    cube.update(10.0, np.linalg.inv(OBS_T_HAND) @ pose([0.01, 0.0, -0.05]))
    obs = cube.observation(10.1)
    assert obs == pytest.approx([0.01, 0.0, -0.05, 1, 0, 0, 0, 1], abs=1e-5)
    assert cube.observation(10.3) == pytest.approx([*obs[:7], 0], abs=1e-6)
    cube.update(9.0, np.eye(4))  # an older pose does not replace a newer one
    assert cube.observation(10.1) == pytest.approx(obs, abs=1e-6)


def test_cube_quaternion_keeps_its_sign_through_half_a_turn():
    from cube_spin_policy import OBS_T_HAND, CubePose
    cube, last = CubePose(smoothing=False), None
    for deg in range(0, 361, 10):
        cube.update(deg, np.linalg.inv(OBS_T_HAND) @ pose([0, 0, -0.05], Rotation.from_euler("z", deg, degrees=True)))
        w, x, y, z = cube.quat
        assert Rotation.from_quat([x, y, z, w]).magnitude() == pytest.approx(np.radians(min(deg, 360 - deg)), abs=1e-4)
        if last is not None:
            assert np.dot(cube.quat, last) > 0
        last = cube.quat.copy()
    assert cube.quat[0] < 0  # a full turn: -q, as the simulation's continuous quaternion


def test_cube_pose_filters_as_the_simulated_tracker():
    """The filters of faive_lab's CubeTrackerModel: the steady-state Kalman gains for the 20 Hz step (the values faive_lab
    computes for CUBE_TRACKER), slerp 0.25 (about 3 steps of lag: ~17 deg at 2 rad/s), less noise on a still cube."""
    from cube_spin_policy import OBS_T_HAND, CubePose, kalman_gains
    assert kalman_gains(1e-3, 3e-3, 0.05) == pytest.approx((0.46082, 0.42394), abs=1e-5)
    to_hand = np.linalg.inv(OBS_T_HAND)
    cube = CubePose()
    lags = []
    for i in range(200):
        t = i * 0.05
        true = Rotation.from_euler("z", 2.0 * t)
        cube.update(t, to_hand @ pose([0, 0, -0.05], true))
        w, x, y, z = cube.quat
        lags.append(np.degrees((Rotation.from_quat([x, y, z, w]) * true.inv()).magnitude()))
    assert np.median(lags[50:]) == pytest.approx(17.2, abs=1.0)

    rng = np.random.default_rng(0)
    raw, smooth = CubePose(smoothing=False), CubePose()
    out_raw, out_smooth = [], []
    for i in range(400):
        noisy = to_hand @ pose([0.0, 0.0, -0.05 + rng.normal(0, 0.004)])
        raw.update(i * 0.05, noisy)
        smooth.update(i * 0.05, noisy)
        out_raw.append(raw.pos[2])
        out_smooth.append(smooth.pos[2])
    assert np.std(out_smooth[20:]) < 0.8 * np.std(out_raw[20:])


def test_cube_pose_filter_starts_again_after_a_restart_or_a_long_gap():
    from cube_spin_policy import CUBE_FILTER_MAX_GAP_S, OBS_T_HAND, CubePose
    to_hand = np.linalg.inv(OBS_T_HAND)
    cube = CubePose()
    for i in range(20):
        cube.update(i * 0.05, to_hand @ pose([0.0, 0.0, -0.05]))
    cube.restart()  # a new run: the next detection is taken as it is
    cube.update(1.0, to_hand @ pose([0.02, 0.0, -0.05]))
    assert cube.pos == pytest.approx([0.02, 0.0, -0.05], abs=1e-6)
    cube.update(1.0 + CUBE_FILTER_MAX_GAP_S + 0.1, to_hand @ pose([-0.01, 0.0, -0.05]))  # cube put back after a drop
    assert cube.pos == pytest.approx([-0.01, 0.0, -0.05], abs=1e-6)
    cube.update(1.0 + CUBE_FILTER_MAX_GAP_S + 0.15, to_hand @ pose([0.03, 0.0, -0.05]))  # 50 ms later: filtered
    assert -0.01 < cube.pos[0] < 0.03


def test_camera_pose_as_the_launch_file_builds_tf():
    """cube_spin_v2.launch.py publishes tag -> hand_root (inverse of the measured hand_T_tag, moved by the mount) and
    tag -> camera (inverse of cam_T_tag); TF then gives hand_root -> camera."""
    from cube_spin_policy import hand_T_camera
    cam_T_tag = pose([0.03, -0.02, 0.41], Rotation.from_euler("xyz", [170, 5, -30], degrees=True))
    measured = [0.01, 0.05, 0.055, 90.0, 0.0, 180.0]
    t = np.array(measured[:3]) + [-0.01, 0.0, -0.058]
    rot_inv = Rotation.from_euler("XYZ", measured[3:], degrees=True).inv()
    tag_T_hand = pose(-rot_inv.apply(t), rot_inv)
    expected = np.linalg.inv(tag_T_hand) @ np.linalg.inv(cam_T_tag)
    assert hand_T_camera(cam_T_tag, measured) == pytest.approx(expected, abs=1e-12)


def test_policy_with_the_cube_pose_waits_for_it_and_gets_it_after_the_joint_velocities(tmp_path):
    from cube_spin_policy import JOINT_LIMITS_USD_DEG, CubeSpinPolicy
    joints = list(reversed(JOINT_LIMITS_USD_DEG))  # another order than the policy's
    policy = CubeSpinPolicy(linear_policy(tmp_path / "e.pt", 82), joints, use_cube_pose=None)
    assert policy.use_cube_pose and policy.obs_dim == 82
    q, qd = np.zeros(17), np.zeros(17)
    assert policy.step(q, qd) is None and policy.last_joint_cmd_usd is None  # no cube pose yet: nothing happens
    cube = np.arange(8, dtype=np.float32)
    _, obs, _, _ = policy.step(q, qd, cube)
    assert obs.shape == (82,) and np.array_equal(obs[34:42], cube)
    assert obs[42:48] == pytest.approx([0, 0, -0.05, 0, 0, 1.0])
    proprio = CubeSpinPolicy(linear_policy(tmp_path / "p.pt", 74), joints, use_cube_pose=None)
    assert not proprio.use_cube_pose and proprio.step(q, qd, cube)[1].shape == (74,)
    with pytest.raises(ValueError):
        CubeSpinPolicy(tmp_path / "e.pt", joints)  # the ROS node's default: without the cube pose


def test_evaluation_takes_policies_with_the_cube_pose_only_with_a_calibration(tmp_path):
    import eval_policies as ep
    extero, proprio = linear_policy(tmp_path / "e.pt", 82), linear_policy(tmp_path / "p.pt", 74)
    assert ep.not_deployable(proprio, calibrated=False) == ""
    assert "--calibration" in ep.not_deployable(extero, calibrated=False)
    assert ep.not_deployable(extero, calibrated=True) == ""
    assert "none of the layouts" in ep.not_deployable(linear_policy(tmp_path / "x.pt", 80), calibrated=True)
    with pytest.raises(SystemExit):
        ep.parse_args(["--calibration", str(tmp_path / "missing.npy"), str(extero)])
    np.save(tmp_path / "cal.npy", np.eye(4))
    args = ep.parse_args(["--calibration", str(tmp_path / "cal.npy"), str(extero)])
    assert ep.camera_in_hand(args).shape == (4, 4)
    assert args.twin  # the digital twin comes with the calibration
    assert not ep.parse_args(["--calibration", str(tmp_path / "cal.npy"), "--no-twin", str(extero)]).twin
    assert not ep.parse_args([str(proprio)]).twin and ep.parse_args(["--twin", str(proprio)]).twin


def test_twin_panel_goes_into_the_top_right_corner():
    from cube_eval.hud import picture_in_picture
    frame, panel = np.zeros((540, 960, 3), np.uint8), np.full((380, 300, 3), 200, np.uint8)
    picture_in_picture(frame, panel)
    assert (frame[8:388, 652:952] == 200).all() and frame[:8].max() == 0 and frame[:, :652].max() == 0
    picture_in_picture(small := np.zeros((200, 200, 3), np.uint8), panel)  # cut to fit a smaller frame
    assert (small[8:192, 8:192] == 200).all()


def test_mock_camera_sees_the_cube_where_the_task_commands_it():
    import eval_policies as ep
    from cube_eval.vision import MockCamera
    from cube_spin_policy import OBS_T_HAND
    hand_T_camera = ep.camera_in_hand(ep.parse_args(["--mock", "--pose-only"]))
    assert (OBS_T_HAND @ hand_T_camera @ [*MockCamera.HOME, 1.0])[:3] == pytest.approx([0, 0, -0.05], abs=1e-9)
    assert ep.camera_in_hand(ep.parse_args(["--pose-only"])) is None


# -- the special modes (--mode)
def recurrent_policy(path, inputs):
    """As rsl_rl exports an LSTM: the memory in buffers that forward() updates."""
    torch = pytest.importorskip("torch")

    class Memory(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("hidden_state", torch.zeros(1, 1, 4))
            self.register_buffer("cell_state", torch.zeros(1, 1, 4))
            self.actor = torch.nn.Linear(inputs, 17)

        def forward(self, x):
            self.hidden_state.copy_(self.hidden_state + 1.0)
            return self.actor(x) + self.hidden_state.sum()

    torch.jit.script(Memory()).save(str(path))
    return path


def with_training(path, **training):
    import json
    path.with_suffix(".json").write_text(json.dumps(dict(training=training)))
    return path


def test_history_is_per_term_oldest_first_and_filled_by_the_first_value(tmp_path):
    from cube_spin_policy import JOINT_LIMITS_USD_DEG, CubeSpinPolicy
    policy = CubeSpinPolicy(linear_policy(tmp_path / "adr.pt", 210), list(JOINT_LIMITS_USD_DEG), use_cube_pose=None)
    q, zero = (lambda v: np.full(17, v, dtype=np.float32)), np.zeros(17)
    first = policy.step(q(0.1), zero)[1]
    assert first.shape == (210,)
    assert np.allclose(first[:85].reshape(5, 17), policy.normalize(q(0.1)))  # joint_pos: 5 times the first value
    assert first[102:108] == pytest.approx([0, 0, -0.05, 0, 0, 1.0])  # goal_command: no history
    second = policy.step(q(0.2), zero)[1]
    positions = second[:85].reshape(5, 17)
    assert np.allclose(positions[:4], policy.normalize(q(0.1))) and np.allclose(positions[4], policy.normalize(q(0.2)))
    commands = second[125:].reshape(5, 17)  # last_joint_cmd: the newest is the command of the first step
    assert np.allclose(commands[:4], q(0.1)) and not np.allclose(commands[4], q(0.1))
    policy.reset()
    assert np.allclose(policy.step(q(0.3), zero)[1][:85].reshape(5, 17), policy.normalize(q(0.3)))

    student = CubeSpinPolicy(linear_policy(tmp_path / "s.pt", 740), list(JOINT_LIMITS_USD_DEG), use_cube_pose=None)
    obs = student.step(q(0.1), zero)[1]
    assert obs[340:400].reshape(10, 6) == pytest.approx(np.tile([0, 0, -0.05, 0, 0, 1.0], (10, 1)))


def test_the_memory_of_a_recurrent_policy_is_cleared_at_every_reset(tmp_path):
    from cube_spin_policy import JOINT_LIMITS_USD_DEG, CubeSpinPolicy
    policy = CubeSpinPolicy(recurrent_policy(tmp_path / "m.pt", 74), list(JOINT_LIMITS_USD_DEG), use_cube_pose=None)
    assert policy.recurrent
    q = np.zeros(17)
    first = policy.step(q, q)[2]
    assert not np.allclose(policy.step(q, q)[2], first)
    policy.reset()
    assert np.allclose(policy.step(q, q)[2], first)  # also after the size probes when it was loaded


def test_goal_orientations_follow_the_tracked_cube():
    from cube_spin_policy import GoalOrientations, rotation_of, wxyz
    goals = GoalOrientations(tolerance=0.4, seed=0)
    goal = goals.quat.copy()
    far = wxyz(rotation_of(goal) * Rotation.from_rotvec([0.0, 0.0, 1.0]))
    obs = goals.observation(np.array([0, 0, -0.05, *far, 1.0]))
    assert goals.reached == 0 and goals.error == pytest.approx(1.0) and np.allclose(obs[:4], goal)
    expected = rotation_of(far) * rotation_of(goal).inv()  # tracked * conj(goal), as cube_pose_tracked_goal
    assert obs[4] >= 0 and (rotation_of(obs[4:]) * expected.inv()).magnitude() == pytest.approx(0.0, abs=1e-5)
    near = wxyz(rotation_of(goal) * Rotation.from_rotvec([0.0, 0.3, 0.0]))
    goals.observation(np.array([0, 0, -0.05, *near, 0.0]))  # lost: does not count
    assert goals.reached == 0
    obs = goals.observation(np.array([0, 0, -0.05, *near, 1.0]))
    assert goals.reached == 1 and not np.allclose(goals.quat, goal) and np.allclose(obs[:4], goals.quat)


def test_goal_and_axis_layouts(tmp_path):
    from cube_spin_policy import JOINT_LIMITS_USD_DEG, CubeSpinPolicy, axis_in_camera
    import eval_policies as ep
    joints, q = list(JOINT_LIMITS_USD_DEG), np.zeros(17)
    goals = CubeSpinPolicy(linear_policy(tmp_path / "g.pt", 84), joints, use_cube_pose=None)
    cube = np.array([0, 0, -0.05, 1, 0, 0, 0, 1], dtype=np.float32)
    assert goals.goals is not None and goals.step(q, q) is None  # waits for the cube pose
    obs = goals.step(q, q, cube)[1]
    assert np.array_equal(obs[34:42], cube) and np.allclose(obs[42:46], goals.goals.quat)
    axis = CubeSpinPolicy(linear_policy(tmp_path / "a.pt", 82), joints, use_cube_pose=None, goal_ang_vel=1.5, axis="x")
    assert axis.goal_command == pytest.approx([0, 0, -0.05, 1.5, 0, 0])
    axis.flip_direction()
    assert axis.goal_command[3] == pytest.approx(-1.5)
    mock = ep.camera_in_hand(ep.parse_args(["--mock", "--pose-only"]))  # looks down at the palm
    assert axis_in_camera(mock, "z") == pytest.approx([0, 0, -1]) and axis_in_camera(mock, "x") == pytest.approx([1, 0, 0])


def test_policies_that_need_a_mode_say_which_arguments_run_them(tmp_path):
    import eval_policies as ep
    student = linear_policy(tmp_path / "hist_policy.pt", 740)
    lstm = recurrent_policy(tmp_path / "lstm_policy.pt", 74)
    adr = linear_policy(tmp_path / "adr_policy.pt", 210)
    goals = linear_policy(tmp_path / "goals_policy.pt", 84)
    axes = with_training(linear_policy(tmp_path / "axes_policy.pt", 82),
                         commands=dict(continuous_rotation=dict(axis_probs=[0.1, 0.1, 0.8])))
    alpha = with_training(linear_policy(tmp_path / "alpha_policy.pt", 74), action_alpha=0.5)
    teacher = with_training(linear_policy(tmp_path / "teacher_policy.pt", 218), actor_obs=["critic"])
    assert ep.not_deployable(student, True) == ("a distilled student with 10 steps of history of every observation: "
                                                "try --mode student")
    assert ep.not_deployable(student, True, mode="student") == ""
    assert ep.not_deployable(lstm, True).endswith("an LSTM: try --mode student")
    assert ep.not_deployable(adr, True).endswith("try --mode history")
    assert ep.not_deployable(goals, False).endswith("try --mode goals --calibration CAM_T_TAG.npy --hand-to-tag X Y Z "
                                                    "ROLL PITCH YAW")
    assert ep.not_deployable(goals, True, mode="goals") == ""
    assert ep.not_deployable(axes, True).endswith("try --mode all-axes")
    assert ep.not_deployable(alpha, False).endswith("try --action-alpha 0.5")
    assert ep.not_deployable(alpha, False, action_alpha=0.5) == ""
    assert ep.not_deployable(alpha, False, mode="student", action_alpha=0.5).endswith("try without --mode student")
    assert ep.not_deployable(alpha, False, mode="student").endswith("try without --mode student --action-alpha 0.5")
    assert ep.not_deployable(teacher, True).startswith("a teacher: it takes the 218 privileged observations")
    assert ep.run_setting(alpha) == ("", 0.5) and ep.run_setting(lstm) == ("student", 0.2)


def test_mode_arguments(tmp_path):
    import eval_policies as ep
    policy = linear_policy(tmp_path / "p_policy.pt", 74)
    assert ep.parse_args([str(policy)]).duration == 60.0
    assert ep.parse_args(["--mode", "goals", str(policy)]).duration == 120.0
    for wrong in (["--axis", "x"], ["--mode", "all-axes", "--axis", "y"], ["--goal-timeout", "30"]):
        with pytest.raises(SystemExit):
            ep.parse_args([*wrong, str(policy)])
    np.save(tmp_path / "cal.npy", np.eye(4))
    assert ep.parse_args(["--mode", "all-axes", "--axis", "x", "--calibration", str(tmp_path / "cal.npy"),
                          str(policy)]).axis == "x"


def test_a_run_counts_as_a_test_only_with_the_setting_the_model_needs(tmp_path):
    for name in ("lstm", "paca"):
        (tmp_path / f"{name}_policy.pt").write_bytes(name.encode())
    old = [record("lstm", "lstm"), record("paca", "paca")]  # before --mode and --action-alpha: ("", 0.2)
    known = wr.known_models(old)
    assert wr.tested_because(known, "lstm", "", wr.setting()) != ""
    assert wr.tested_because(known, "lstm", "", wr.setting("student")) == ""
    settings = {"lstm_policy.pt": wr.setting("student"), "paca_policy.pt": wr.setting("", 0.5)}
    todo, _ = wr.split_models(tmp_path, known, setting_of=lambda p: settings[p.name])
    assert [p.name for p in todo] == ["lstm_policy.pt", "paca_policy.pt"]
    new = record("lstm", "lstm")
    new["config"]["eval"].update(mode="student", action_alpha=0.2)
    todo, _ = wr.split_models(tmp_path, wr.known_models(old + [new]), setting_of=lambda p: settings[p.name])
    assert [p.name for p in todo] == ["paca_policy.pt"]


def test_mode_in_the_wandb_config_and_the_leaderboard(tmp_path):
    meta = dict(meta_of(model="lemur"), observations=82, cube_pose=True, mode="all-axes", axis="x", action_alpha=0.2)
    e = wr.config_of(meta, tmp_path / "s" / "r", {})["eval"]
    assert (e["mode"], e["axis"], e["cube_pose"], e["action_alpha"]) == ("all-axes", "x", True, 0.2)
    student = wr.config_of(dict(meta_of(model="hist"), observations=740, cube_pose=False), tmp_path / "s" / "r", {})
    assert student["eval"]["cube_pose"] is False and student["eval"]["mode"] == ""
    about_x = record("lemur", "lemur", turns=3.0)
    about_x["config"]["eval"].update(mode="all-axes", axis="x")
    about_z = record("lemur", "lemur", turns=5.0)
    about_z["config"]["eval"].update(mode="all-axes", axis="z")
    assert [r["each_run"] for r in wr.leaderboard_rows([about_x, about_z])] == ["5.0"]
