"""Offline tests (no hardware, no W&B) of the turn counting, the drop detection, the analysis, the live view, the video,
the saving of results, the temperature filter, the servo status of the hand controller, which runs count and can be
compared (results_rules.py), and what upload_wandb.py sends:

    pixi run python src/rwr_system/experiments/rl_inference/real_eval/test_offline.py [image.png]
"""

import os
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from real_eval.cube_tracking import CUBE_SYMMETRIES, DropMonitor, TurnCounter, resting_position  # noqa: E402

CAMERA_DOWN = Rotation.from_euler("x", 180, degrees=True)  # camera z points down at the table
rng = np.random.default_rng(0)


def detections(speed=1.5, seconds=60.0, rate=8.0, tilt_deg=0.0, noise=0.02, wrong=0.0):
    """Detections of a cube turning about an axis tilted from vertical, seen from a camera looking down. Every pose is
    a random one of the 24 symmetric ones; a share `wrong` of them are random wrong poses (e.g. PnP flips)."""
    axis = Rotation.from_euler("x", tilt_deg, degrees=True).apply([0, 0, 1])
    position = np.array([0.0, 0.0, 0.45])
    for k in range(int(seconds * rate)):
        t = k / rate
        world = Rotation.from_rotvec(axis * speed * t)
        rot = CAMERA_DOWN.inv() * world * CUBE_SYMMETRIES[int(rng.integers(24))]
        rot = rot * Rotation.from_rotvec(rng.normal(0, noise, 3))
        if rng.random() < wrong:
            rot = Rotation.random(random_state=int(rng.integers(1 << 30)))
        yield t, position + rng.normal(0, 0.001, 3), rot


def check(name, ok, detail):
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {detail}")
    return ok


def main():
    results = []
    truth = 1.5 * 60 / (2 * np.pi)
    for tilt in (0, 30, 45):
        c = TurnCounter()
        for t, p, r in detections(tilt_deg=tilt):
            c.add(t, p, r)
        res = c.result()
        results.append(check(f"turns, spin axis tilted {tilt} deg", abs(res.turns - truth) < 0.05,
                             f"{res.turns:.2f} of {truth:.2f} (camera axis {res.turns_camera_axis:.2f}, "
                             f"estimated tilt {res.spin_axis_tilt_deg:.0f} deg)"))
    c = TurnCounter()
    for t, p, r in detections(wrong=0.05):
        c.add(t, p, r)
    res = c.result()
    results.append(check("turns with 5 % wrong detections", abs(res.turns - truth) < 0.15 * truth,
                         f"{res.turns:.2f} of {truth:.2f}, rejected {res.rejected_share:.0%}"))
    c = TurnCounter()
    for t, p, r in detections(speed=0.0):
        c.add(t, p, r)
    results.append(check("still cube", abs(c.result().turns) < 0.02, f"{c.result().turns:+.3f} turns"))

    # drop detection: a single wrong position is not a drop, a cube that stays away is
    c = TurnCounter()
    drops = DropMonitor(np.array([0.0, 0.0, 0.45]))
    single = False
    for k in range(80):
        t = k / 8
        p = np.array([0.0, 0.0, 0.45]) + (np.array([0.10, 0, 0]) if k == 40 else 0)
        c.add(t, p, Rotation.identity())
        single |= drops.update(t, c)
    results.append(check("one wrong position", not single, "no drop"))
    fell_at = None
    for k in range(80, 120):
        t = k / 8
        c.add(t, np.array([0.12, 0.0, 0.50]), Rotation.identity())
        if fell_at is None and drops.update(t, c):
            fell_at = t - 10.0
    results.append(check("cube that falls", fell_at is not None and fell_at <= 1.5, f"drop after {fell_at} s"))

    still = [np.array([0.0, 0.0, 0.45]) + rng.normal(0, 0.001, 3) for _ in range(15)]
    moving = [np.array([0.002 * k, 0.0, 0.45]) for k in range(15)]
    results.append(check("cube lying still", resting_position(still) is not None and resting_position(moving) is None,
                         "detected still, not while moving"))

    # progress over time: 1.5 rad/s for 20 s, still for 15 s, then 1.5 rad/s again until 60 s
    from real_eval.analysis import joint_summary, joint_tracking, plot_run, progress_metrics, quality_warnings
    t = np.arange(0.0, 60.0, 0.125)
    angle = 1.5 * np.minimum(t, 20.0) + 1.5 * np.clip(t - 35.0, 0.0, None)
    m = progress_metrics(t, angle / (2 * np.pi), 0.0, 60.0)
    results.append(check("progress over time", abs(m["turns_30s"] - 30 / (2 * np.pi)) < 0.02
                         and abs(m["best_10s_rad_per_s"] - 1.5) < 0.02 and 15.0 <= m["longest_stall_s"] <= 16.5,
                         f"{m['turns_30s']} turns after 30 s, best 10 s {m['best_10s_rad_per_s']} rad/s, longest stall "
                         f"{m['longest_stall_s']} s (15 s still)"))

    # joints: command at 20 Hz, joint 0 follows 150 ms late, joint 1 moves a third of its command, joint 2 still
    names = ["wrist", "pinky_mcp", "thumb_cmc"]
    jt = np.arange(0.0, 20.0, 0.01)
    def command(tt):
        tt = np.floor(tt * 20) / 20  # zero-order hold of the 20 Hz policy
        return np.stack([0.3 * np.sin(2 * tt), 0.4 * np.sin(3 * tt + 1), np.full_like(tt, 0.2)], axis=-1)
    cmd = command(jt)
    meas = np.stack([command(jt - 0.15)[:, 0], cmd[:, 1] / 3, cmd[:, 2] + 0.01], axis=-1)
    meas += rng.normal(0, 0.002, meas.shape)
    log = list(zip(jt, cmd, meas))
    tracking = joint_tracking(log, names)
    s = joint_summary(tracking)
    results.append(check("joint delay and stuck joint", tracking[0].lag_ms is not None and abs(tracking[0].lag_ms - 150) <= 20
                         and tracking[2].lag_ms is None and s["stuck_joints"] == "pinky_mcp",
                         f"delays {[j.lag_ms for j in tracking]} ms (150, -, -), barely moving: {s['stuck_joints']}"))
    w = quality_warnings(dict(duration_s=60, tracked_fraction=0.3, longest_gap_s=4.2, rejected_poses=0.1), 0.5, 3.0)
    results.append(check("quality warnings", len(w) == 2, "; ".join(w)))
    import tempfile
    out = Path(tempfile.mkdtemp())
    plot_run(out, "test", t, angle / (2 * np.pi), 0.0, 60.0, 1.5, [[k, 40 + k / 10, 35, 45] for k in range(60)],
             names, log, "time")
    results.append(check("plots", (out / "turns_vs_time.png").exists() and (out / "joints.png").exists(), str(out)))

    # live view on a synthetic image
    from real_eval.live_view import ViewState, render
    K = np.array([[775.0, 0, 480], [0, 775.0, 270], [0, 0, 1]])
    image = np.full((540, 960, 3), 200, np.uint8)
    state = ViewState(title="06_wombat   goal +1.5 rad/s", phase="RUNNING 12.3 / 60 s", turns=1.42, pose_age=0.05,
                      position=np.array([0.02, -0.03, 0.45]), rotation=Rotation.from_euler("xyz", [20, 10, 35], degrees=True),
                      trail=[np.array([0.02 + 0.01 * np.sin(a), -0.03 + 0.01 * np.cos(a), 0.45]) for a in np.linspace(0, 3, 30)],
                      rest_position=np.array([0.0, -0.03, 0.45]), temps=np.array([41.0, 52.0, 47.0]),
                      joint_names=["wrist", "pinky_mcp", "thumb_cmc"], detections_per_s=8.0)
    frame = render(image, state, K, np.zeros(5), 0.045)
    results.append(check("live view", frame.shape == image.shape and (frame != image).any(), "rendered"))
    if len(sys.argv) > 1:
        import cv2
        cv2.imwrite(sys.argv[1], frame)
    results += check_recording_and_saving()
    results += check_results_and_wandb()
    print(f"\n{sum(results)} of {len(results)} checks passed")
    sys.exit(0 if all(results) else 1)


def check_recording_and_saving():
    """The parts added for the hardware runs: video thread, currents, temperature filter, model order, results files,
    servo status bits and the cleanup of empty sessions."""
    import csv
    import json
    import tempfile
    import threading
    import time
    import types

    import cv2

    import evaluate_real_policies as ev
    from real_eval.analysis import joint_summary, joint_tracking
    from real_eval.live_view import VideoRecorder
    from real_eval.processes import Processes
    from real_eval.ros_interface import RosInterface
    results, tmp = [], Path(tempfile.mkdtemp())

    # video: written from its own thread in real time; closing while frames arrive must not crash
    rec, stop = VideoRecorder(tmp / "v.mp4"), threading.Event()
    frame = np.zeros((120, 160, 3), np.uint8)

    def feed():
        while not stop.is_set():
            rec.write(frame)
            time.sleep(0.01)
    feeder = threading.Thread(target=feed)
    feeder.start()
    time.sleep(1.5)
    rec.close()
    stop.set()
    feeder.join()
    n = int(cv2.VideoCapture(str(tmp / "v.mp4")).get(cv2.CAP_PROP_FRAME_COUNT))
    results.append(check("video while closing", 25 <= n <= 35, f"{n} frames in 1.5 s at 20 fps"))

    # motor currents in the joint analysis
    names = [f"j{i}" for i in range(17)]
    t = np.arange(0, 10, 0.01)
    cmd = np.sin(t)[:, None] * np.ones(17) * 0.5
    cur = np.full((len(t), 17), 100.0)
    cur[:, 3] = -400.0
    tracking = joint_tracking([(t[i], cmd[i], np.roll(cmd, 15, axis=0)[i], cur[i]) for i in range(len(t))], names)
    s = joint_summary(tracking)
    results.append(check("motor currents", tracking[3].mean_abs_current_ma == 400 and s["total_current_a"] == 2.0
                         and s["highest_current_joint"] == "j3",
                         f"{s['total_current_a']} A, most: {s['highest_current_joint']}"))

    # bad temperature readings (the wrist servo sent 150 C and 58 C at 30 C) are ignored; a repeated new value counts
    ros = RosInterface.__new__(RosInterface)
    ros.lock, ros.temps, ros._temp_pending, ros.temp_glitches = threading.Lock(), None, None, 0
    seen = []
    for wrist in [31, 31, 150, 31, 31, 58, 30, 32, 50, 50, 51]:
        ros._temps_cb(types.SimpleNamespace(data=[wrist, 36.0]))
        seen.append(float(ros.temps[0]))
    results.append(check("temperature glitches", seen == [31, 31, 31, 31, 31, 31, 30, 32, 32, 50, 51]
                         and ros.temp_glitches == 3, f"{ros.temp_glitches} ignored, wrist seen {seen}"))

    # --models: in the command-line order, 9 = 09, names, no repeats
    (tmp / "models.json").write_text(json.dumps([dict(idx=i, name=f"m{i}", dir="", goal=1.5) for i in (48, 20, 9, 44)]))
    args = types.SimpleNamespace(models_dir=str(tmp), models=["44", "20", "9", "m48", "44"], oldest_first=False)
    order = [m["idx"] for m in ev.load_models(args)]
    results.append(check("model order", order == [44, 20, 9, 48], f"{order}"))

    # results: a new column extends the header; only saved runs go into all_results.csv
    e = ev.Evaluation.__new__(ev.Evaluation)
    e.args = types.SimpleNamespace(models_dir=str(tmp))
    session = tmp / "eval_x"
    (session / "r1").mkdir(parents=True)
    (session / "r2").mkdir()
    e._save(session / "results.csv", dict(model="a", turns=1, valid="yes", run_dir=str(session / "r1")))
    e._save(session / "results.csv", dict(model="b", turns=2, valid="no", run_dir=str(session / "r2"), new="x"))
    rows = list(csv.DictReader(open(session / "results.csv")))
    saved = list(csv.DictReader(open(tmp / "all_results.csv")))
    results.append(check("results files", len(rows) == 2 and rows[0]["new"] == "" and len(saved) == 1
                         and saved[0]["session"] == "eval_x", "both rows in the session, the saved one in all_results"))

    # W&B: a saved run starts the uploader in the background, without this Python's PYTHONPATH; an unsaved one does not
    started, real_popen, real_path = [], ev.subprocess.Popen, os.environ.get("PYTHONPATH")
    ev.subprocess.Popen = lambda cmd, **kw: started.append((cmd, kw))
    os.environ["PYTHONPATH"] = "/opt/ros/somewhere"
    try:
        e.wandb_python = "/other/python"
        e._save(session / "results.csv", dict(model="c", turns=3, valid="yes", run_dir=str(session / "r1")))
        e._save(session / "results.csv", dict(model="d", turns=4, valid="no", run_dir=str(session / "r2")))
    finally:
        ev.subprocess.Popen = real_popen
        if real_path is None:
            os.environ.pop("PYTHONPATH")
        else:
            os.environ["PYTHONPATH"] = real_path
    cmd, kw = started[0] if started else ([], {})
    results.append(check("W&B upload", len(started) == 1 and cmd[:2] == ["/other/python", str(ev.WANDB_UPLOADER)]
                         and cmd[-4:] == ["--models-dir", str(tmp), "--session", "eval_x"]
                         and "PYTHONPATH" not in kw["env"] and kw["start_new_session"]
                         and ev.find_wandb_python("/no/such/python") is None,
                         "one background upload of the session for the saved run; no Python with wandb: off"))

    # servo status: faults of the run and of the parking afterwards, lowest voltage
    e.joint_ids = ["wrist", "index_abd", "pinky_mcp"]
    e.ros = types.SimpleNamespace(lock=threading.Lock(), status_errors=np.array([0, 0x02, 0x20]))
    e.motor_status = [(1.0, np.array([11.8, 11.9, 12.0]), None), (2.0, np.array([10.5, 11.0, 11.2]), None)]
    e._run_errors = np.array([0, 0, 0x20])
    s = e._motor_status_summary()
    results.append(check("servo status", s == dict(min_voltage_v=10.5, servo_errors="pinky_mcp: overload",
                                                   servo_errors_parking="index_abd: angle_sensor"), f"{s}"))

    # the hand controller takes voltage and status bits from orca_core's last Feetech packet
    import importlib.util
    hc_path = Path(__file__).resolve().parents[3] / "src/hand_control/orca/hand_controller_v2.py"
    spec = importlib.util.spec_from_file_location("hand_controller_v2", hc_path)
    hc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hc)
    from orca_core.hardware.feetech.group_sync_read import GroupSyncRead
    reader = GroupSyncRead.__new__(GroupSyncRead)
    reader.ph = types.SimpleNamespace(scs_makeword=lambda a, b: a | (b << 8))
    reader.start_address, reader.data_length = 56, 15
    reader.data_dict = {mid: [0x20 if mid == 3 else 0] + [0] * 6 + [118 + mid % 3] + [0] * 8 for mid in (1, 2, 3)}
    hand = hc.OrcaHandV2.__new__(hc.OrcaHandV2)
    hand.num_motors, hand._joint_order = 3, np.array([2, 0, 1])
    hand._status_errors, hand._status_voltage = np.zeros(3, dtype=np.int64), np.full(3, np.nan)
    hand.status_available, hand.status_problem = None, ""
    hand._hand = types.SimpleNamespace(_motor_client=types.SimpleNamespace(_sync_reads={(56, 15): reader}),
                                       config=types.SimpleNamespace(motor_ids=[1, 2, 3]))
    hand.collect_status()
    volt, errors = hand.take_status()
    ok = errors.tolist() == [0x20, 0, 0] and np.allclose(volt, [11.8, 11.9, 12.0]) and hand.take_status()[1].sum() == 0
    results.append(check("controller servo status", ok, f"voltages {volt.tolist()}, status {errors.tolist()}"))

    # node logs: a restarted node appends; empty session folders are removed, others kept
    procs = Processes(tmp / "logs")
    for word in ("first", "second"):
        procs.start("x", ["echo", word])
        time.sleep(0.3)
        procs.stop("x")
    log = (tmp / "logs/x.log").read_text()
    kept = []
    for name, new_folder, failed, run in [("empty", True, False, False), ("failed", True, True, False),
                                          ("has_run", True, False, True), ("continued", False, False, False)]:
        (tmp / name / "node_logs").mkdir(parents=True)
        if run:
            (tmp / name / "51_bison_1600").mkdir()
        ev.remove_if_empty(tmp / name, new_folder, failed)
        kept.append((tmp / name).exists())
    ok = "first" in log and "second" in log and kept == [False, True, True, True]
    results.append(check("logs and empty sessions", ok, "restart appends; only the empty new session removed"))
    return results



def check_results_and_wandb():
    """results_rules.py (saved runs, conditions, stable ids) and upload_wandb.py's curves and leaderboard (no W&B)."""
    import csv
    import json
    import shutil
    import tempfile

    import results_rules as rr
    import upload_wandb as up
    results, tmp = [], Path(tempfile.mkdtemp())
    (tmp / "models.json").write_text(json.dumps([dict(idx=48, name="ibex", goal=1.5, description="d")]))
    session = tmp / "eval_20261111_100000"
    rows = [dict(time="2026-11-11 10:00:05", model="48_ibex", goal_ang_vel=1.5, start_temp_max=41, turns=1.59,
                 duration_s=60, outcome="completed", valid="yes", run_dir=str(session / "48_ibex_100005")),
            dict(time="2026-11-11 10:05:00", model="48_ibex", goal_ang_vel=2.2, start_temp_max=45, turns=2.0,
                 duration_s=60, outcome="completed", valid="yes", run_dir=str(session / "48_ibex_100500")),
            dict(time="2026-11-11 10:10:00", model="48_ibex", goal_ang_vel=1.5, start_temp_max=40, turns=9.0,
                 duration_s=60, outcome="overheated", valid="no", run_dir=str(session / "48_ibex_101000"))]
    for r in rows:
        Path(r["run_dir"]).mkdir(parents=True)
    with open(session / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    # 100 accepted steps of 0.1 rad about the up axis (camera -z) and one rejected step: 10 rad = 1.59 turns
    with open(session / "48_ibex_100005" / "trace.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "x", "y", "z", "qx", "qy", "qz", "qw", "accepted", "step_rx", "step_ry", "step_rz"])
        for i in range(101):
            w.writerow([1000.0 + 0.6 * i, 0, 0, 0.3, 0, 0, 0, 1, 1, 0, 0, 0.0 if i == 0 else -0.1])
        w.writerow([1060.7, 0, 0, 0.3, 0, 0, 0, 1, 0, 0, 0, -0.9])

    runs = rr.saved_runs(tmp)
    first, other = runs
    why = "goal 2.2 instead of 1.5 rad/s; started at 45 C"
    results.append(check("saved runs and conditions", len(runs) == 2 and first.comparable and not other.at_own_goal
                         and not other.cold_start and other.not_comparable_because == why,
                         f"{[r.not_comparable_because for r in runs]}"))
    moved = tmp / "archive" / session.name
    moved.parent.mkdir()
    shutil.move(str(session), str(moved))
    after = rr.saved_runs(tmp)
    results.append(check("run ids", [r.run_id for r in after] == [r.run_id for r in runs]
                         and after[0].folder == moved / "48_ibex_100005", "the same after moving the session"))
    curve = up.curves(after[0].folder, 1.5)
    results.append(check("turn curve", abs(curve[-1]["curve/turns"] - 10 / (2 * np.pi)) < 1e-3
                         and abs(curve[-1]["time_s"] - 60.7) < 1e-6
                         and up.display_name(after[0]) == "48_ibex 2026-11-11 10:00",
                         f"ends at {curve[-1]['curve/turns']:.3f} turns, {curve[-1]['time_s']:.1f} s"))
    board = up.leaderboard(after, rr.load_models(tmp))
    results.append(check("leaderboard", len(board) == 1 and board[0]["runs"] == 1 and board[0]["mean_turns"] == 1.59,
                         f"{board}"))
    return results

if __name__ == "__main__":
    main()
