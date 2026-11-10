"""Offline tests of the turn counting, the drop detection and the live view, on synthetic detections (no ROS):

    pixi run python src/rwr_system/experiments/rl_inference/real_eval/test_offline.py [image.png]
"""

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
    print(f"\n{sum(results)} of {len(results)} checks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
