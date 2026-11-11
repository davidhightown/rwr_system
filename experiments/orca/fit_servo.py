"""Fit the ORCA v2 servo model to a chirp run and write the simulation values.

    python experiments/orca/fit_servo.py logs/chirp_v2/<run>_good

What the chirps can identify, and what this script therefore fits per joint, in the time domain:

    command (20 Hz) -> dead time tau -> acceleration ramp a (the servo's ACC register) -> first-order lag T -> joint

- a is the servo ramp. It also follows from the ACC register and the calibration (22.8 rad/s^2 at the motor /
  motor-per-joint ratio); the script prints both and flags joints that disagree by more than 10 %.
- tau + T is well defined; the split between them is not at 20 Hz sampling. The fast (100 Hz) and step files
  from chirp.py separate them better.
- A linear model (dead time + lag, no ramp) is fit too, as a reference: it is what a pure PD fit can represent.

Joint stiffness, damping, friction and the effort limit are NOT identifiable from free-motion chirps: once the
ramp and the dead time are in the model, very different K/D give the same response. Set them from physics
(see README) instead of fitting them to these data.

Output in the run folder: servo_fit.json (per joint and the sim values) and servo_fit.png.
Requires numpy and pyyaml; matplotlib for the plot.
"""

import argparse
import itertools
import json
import math
from multiprocessing import Pool
import os
from pathlib import Path
import sys

import numpy as np
import yaml

DT = 1e-3                 # model step
SIM_SUBSTEP = 1 / 120     # physics step of the Isaac Lab task
MOTOR_ACC_RAD_S2_PER_RAW = 0.15184364492350666
SKIP_S = 1.0              # transient at the start of each segment, not scored
FINGERS = ["index", "middle", "ring", "pinky"]
GROUPS = {  # sim joint regex -> real joints
    "^right_(index|middle|ring|pinky)_abd$": [f"{f}_abd" for f in FINGERS],
    "^right_(index|middle|ring|pinky)_mcp$": [f"{f}_mcp" for f in FINGERS],
    "^right_(index|middle|ring|pinky)_pip$": [f"{f}_pip" for f in FINGERS],
    "right_thumb_cmc": ["thumb_cmc"], "right_thumb_abd": ["thumb_abd"],
    "right_thumb_mcp": ["thumb_mcp"], "right_thumb_dip": ["thumb_dip"], "right_wrist": ["wrist"],
}


# ---------------------------------------------------------------- model
def simulate(cmd, ctrl_freq, sample_t, tau, a, T, y0=None):
    """Batched model. cmd: (n,) targets written at k / ctrl_freq. tau, a, T: (B,). Returns (B, len(sample_t))."""
    n = len(cmd)
    N = int(math.ceil((max(sample_t.max(), n / ctrl_freq) + 1e-9) / DT)) + 1
    t = np.arange(N) * DT
    B = len(tau)
    p = np.full(B, cmd[0] if y0 is None else y0)
    y, vel = p.copy(), np.zeros(B)
    finite = np.isfinite(a)
    adt = np.where(finite, a, 1.0) * DT
    alpha = DT / (T + DT)
    sample_idx = np.clip(np.round(sample_t / DT).astype(int), 0, N - 1)
    order = np.argsort(sample_idx)
    out = np.empty((B, len(sample_t)))
    s = 0
    for i in range(N):
        while s < len(order) and sample_idx[order[s]] == i:
            out[:, order[s]] = y
            s += 1
        if s == len(order):
            break
        goal = cmd[np.clip(((t[i] - tau) * ctrl_freq).astype(int), 0, n - 1)]
        err = goal - p
        dist = np.abs(err)
        # largest speed from which braking at a still stops at the goal (the servo's trapezoid)
        brake = (np.sqrt(1 + 8 * dist / (adt * DT)) - 1) / 2
        v_des = np.sign(err) * np.minimum(brake * adt, dist / DT)
        vel = np.where(finite, vel + np.clip(v_des - vel, -adt, adt), err / DT)
        p = p + vel * DT
        y = y + alpha * (p - y)
    return out


def score(Y, meas, mask):
    """RMS error after the best static gain and offset (calibration), per row of Y."""
    Ys, m = Y[:, mask], meas[mask]
    xm = Ys.mean(1, keepdims=True)
    gain = ((Ys - xm) * (m - m.mean())).sum(1) / np.maximum(((Ys - xm) ** 2).sum(1), 1e-12)
    resid = (Ys - xm) * gain[:, None] + m.mean() - m
    return np.sqrt((resid ** 2).mean(1))


def segment_errors(segments, P):
    """Mean squared error over all segments for each parameter row (tau, a, T)."""
    total = np.zeros(len(P))
    count = 0
    for seg in segments:
        for c in range(0, len(P), 3000):
            chunk = P[c:c + 3000]
            Y = simulate(seg["cmd"], seg["ctrl_freq"], seg["t"], *chunk.T, y0=seg["y0"])
            total[c:c + 3000] += score(Y, seg["meas"], seg["mask"]) ** 2 * seg["mask"].sum()
        count += seg["mask"].sum()
    return np.sqrt(total / count)


def staged_fit(segments, stages, fixed=None, starts=8):
    """Coarse-to-fine grid over (tau, a, T). The error surface is sharp in a, so the finer stages refine around
    the ``starts`` best coarse points, not only the best one. Every evaluated point counts, so refining never
    makes the result worse."""
    def grid(axes):
        P = np.array(list(itertools.product(*axes)), dtype=float)
        for column, value in (fixed or {}).items():
            P[:, column] = value
        return np.unique(P, axis=0)

    P = grid(stages[0](None))
    seen_P, seen_e = [P], [segment_errors(segments, P)]
    for start in P[np.argsort(seen_e[0])[:starts]]:
        best = start
        for build in stages[1:]:
            P = grid(build(best))
            e = segment_errors(segments, P)
            seen_P.append(P)
            seen_e.append(e)
            best = P[np.argmin(e)]  # each finer grid contains its center, so this never gets worse
    allP, alle = np.concatenate(seen_P), np.concatenate(seen_e)
    near = allP[alle <= 1.1 * alle.min()]
    return allP[np.argmin(alle)], float(alle.min()), near


def around(center, half, n, lo=0.0):
    return np.unique(np.clip(np.linspace(center - half, center + half, n), lo, None))


CAP_STAGES = [
    lambda b: [np.arange(0, .0901, .003), np.geomspace(5, 120, 44), np.arange(0, .0601, .005)],
    lambda b: [around(b[0], .004, 9), b[1] * np.linspace(.94, 1.06, 13), around(b[2], .006, 9)],
    lambda b: [around(b[0], .001, 9), b[1] * np.linspace(.98, 1.02, 9), around(b[2], .0015, 7)],
]
LINEAR_STAGES = [
    lambda b: [np.arange(0, .1201, .005), [np.inf], np.arange(0, .1501, .005)],
    lambda b: [around(b[0], .006, 13), [np.inf], around(b[2], .006, 13)],
]


# ---------------------------------------------------------------- data
def chirp_time_at(metadata, frequency):
    """Time at which the logarithmic chirp of the run reaches ``frequency``."""
    f0, f1 = metadata.get("f0_hz", 0.1), metadata.get("f1_hz", 5.0)
    duration = metadata.get("duration_s", 20.0)
    return duration * math.log(min(max(frequency, f0), f1) / f0) / math.log(f1 / f0)


def load_segments(run_dir, metadata, joint, max_chirp_hz=None):
    """Chirp (fast file if present, else the 20 Hz file) and steps, each as a model segment (radians).

    Chirp samples above ``max_chirp_hz`` are not scored: there the ramp saturates completely, the servo runs
    bang-bang, and which way it switches depends chaotically on the last 1 % of the parameters.
    """
    ctrl = float(metadata["ctrl_freq_hz"])
    chirp_end = math.inf if max_chirp_hz is None else chirp_time_at(metadata, max_chirp_hz)
    files = []
    fast = {name.split("_", 1)[1]: name for name in metadata.get("fast_csv_files", [])}
    for name in metadata["csv_files"]:
        if name.startswith(f"chirp_{joint}_"):
            files.append(fast.get(name.split("_", 1)[1], name))
    files += [name for name in metadata.get("step_csv_files", []) if name.startswith(f"steps_{joint}_")]
    segments = []
    for name in files:
        path = run_dir / name
        if not path.exists() or path.stat().st_size == 0:
            continue
        data = np.genfromtxt(path, delimiter=",", names=True)
        if data.size < 10:
            continue
        t = data["timestamp"]
        # One command per control period; in a fast file, the command is the row's commanded_position at the
        # first read after each write.
        k = np.floor(t * ctrl + 1e-6).astype(int)
        cmd = np.empty(k.max() + 1)
        cmd[k] = data["commanded_position"]
        missing = np.setdiff1d(np.arange(len(cmd)), k)
        if len(missing):
            cmd[missing] = np.interp(missing, k, data["commanded_position"])
        # The model samples where the read happened; the row time is after the read, ~ a constant that tau absorbs.
        # 20 Hz rows: the recorder reads right after writing command k, so sample at k / ctrl.
        sample_t = k / ctrl if not name.startswith(("fast_", "steps_")) else t
        segments.append(dict(name=name, cmd=np.deg2rad(cmd), ctrl_freq=ctrl, t=sample_t,
                             meas=np.deg2rad(data["measured_position"]),
                             mask=(t >= SKIP_S) & ((t < chirp_end) | name.startswith("steps_")),
                             y0=float(np.deg2rad(data["measured_position"][0]))))
    return segments


def predicted_accel(run_dir, metadata, config, joint):
    """Ramp at the joint from the ACC register and the calibration ratio; None if unknown."""
    raw = (metadata.get("servo_profile") or {}).get("servo_acceleration_raw", 150)  # runs before 2026-11-10: 150
    calibration_path = run_dir / "calibration.yaml"
    if not raw or not calibration_path.exists():
        return None
    ratios = (yaml.safe_load(calibration_path.read_text()) or {}).get("joint_to_motor_ratios") or {}
    ratio_deg = ratios.get(abs(config["joint_to_motor_map"][joint]))  # motor rad per joint deg
    return raw * MOTOR_ACC_RAD_S2_PER_RAW / (ratio_deg * 180 / math.pi) if ratio_deg else None


def clipped(segments, threshold=0.85):
    """True if the chirp's peaks fall well short of the command below 1 Hz, where neither the ramp nor the dead
    time can explain it: the joint is stopped by something (a neighbouring finger, a coupled tendon, the
    current limit). Such a joint measures contact, not the free actuator, and is left out of the sim values."""
    seg = next((s for s in segments if not s["name"].startswith("steps_")), None)
    if seg is None:
        return False
    slow = seg["mask"] & (seg["t"] < 0.5 * seg["t"].max())
    if slow.sum() < 20:
        return False
    def span(x):
        return np.percentile(x[slow], 98) - np.percentile(x[slow], 2)
    command = np.interp(seg["t"][slow], np.arange(len(seg["cmd"])) / seg["ctrl_freq"], seg["cmd"])
    return bool(span(seg["meas"]) < threshold * (np.percentile(command, 98) - np.percentile(command, 2)))


def fit_joint(job):
    run_dir, metadata, config, joint, max_chirp_hz = job
    segments = load_segments(run_dir, metadata, joint, max_chirp_hz)
    if not segments:
        return joint, None
    cap, cap_rms, _ = staged_fit(segments, CAP_STAGES, fixed={2: 0.0})
    full, full_rms, near = staged_fit(segments, CAP_STAGES)
    linear, linear_rms, _ = staged_fit(segments, LINEAR_STAGES)
    every_sample = load_segments(run_dir, metadata, joint)
    return joint, {
        "clipped": clipped(segments),
        "dead_time_ms": 1e3 * cap[0], "accel_rad_s2": cap[1], "rms_deg": math.degrees(cap_rms),
        "rms_deg_full_band": math.degrees(float(segment_errors(every_sample, cap[None])[0])),
        "with_lag": {"dead_time_ms": 1e3 * full[0], "accel_rad_s2": full[1], "lag_ms": 1e3 * full[2],
                     "rms_deg": math.degrees(full_rms),
                     "within_10pct": {"dead_time_plus_lag_ms": [1e3 * float((near[:, 0] + near[:, 2]).min()),
                                                                1e3 * float((near[:, 0] + near[:, 2]).max())],
                                      "accel_rad_s2": [float(near[:, 1].min()), float(near[:, 1].max())]}},
        "linear": {"dead_time_ms": 1e3 * linear[0], "lag_ms": 1e3 * linear[2], "rms_deg": math.degrees(linear_rms)},
        "accel_predicted_rad_s2": predicted_accel(run_dir, metadata, config, joint),
        "files": [s["name"] for s in segments],
    }


# ---------------------------------------------------------------- output
def sim_values(joints, flagged):
    """Per sim joint group: median ramp over clean joints; actuator dead time + lag for fingers/thumb and wrist.

    The sim realises dead time + lag as delay_substeps * 1/120 s plus the joint's D/K. A joint whose ramp
    disagrees with ACC/calibration or that is clipped is used only if its group has no clean joint.
    """
    def clean(names):
        good = [j for j in names if joints.get(j) and j not in flagged]
        return good or [j for j in names if joints.get(j)]

    accel = {}
    for group, members in GROUPS.items():
        good = [j for j in members if joints.get(j) and j not in flagged]
        values = [joints[j]["accel_rad_s2"] for j in good]
        if not values:  # no clean joint: the ramp the register and the calibration predict, else the fit
            values = [joints[j]["accel_predicted_rad_s2"] or joints[j]["accel_rad_s2"]
                      for j in members if joints.get(j)]
        if values:
            accel[group] = round(float(np.median(values)), 1)

    def total_ms(names):
        values = [joints[j]["with_lag"]["dead_time_ms"] + joints[j]["with_lag"]["lag_ms"] for j in clean(names)]
        return round(float(np.median(values)), 1) if values else None

    return {
        "ORCA_V2_SERVO_ACCEL": accel,
        "dead_time_plus_lag_ms": {"fingers_thumb": total_ms([j for j in joints if j != "wrist"]),
                                  "wrist": total_ms(["wrist"])},
        "note": "Actuator only, from the command write to the motor encoder. In the sim, delay_substeps / 120 s + "
                "D/K should equal it (e.g. D/K = 20 ms and the rest as delay). The deployment pipeline (ROS, "
                "policy compute, observation age, about 2 substeps) comes on top for the training delay range.",
    }


def plot(run_dir, metadata, joints, path, max_chirp_hz):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    names = [j for j in metadata["joints"] if joints.get(j)]
    cols = 4
    rows = math.ceil(len(names) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(18, 2.8 * rows), squeeze=False)
    for ax, joint in zip(axes.flat, names):
        r = joints[joint]
        seg = load_segments(run_dir, metadata, joint)[0]
        end = chirp_time_at(metadata, max_chirp_hz)  # the last 4 s of the scored band, where the ramp binds
        window = (seg["t"] >= end - 4.0) & (seg["t"] < end)
        params = np.array([[r["dead_time_ms"] / 1e3, r["accel_rad_s2"], 0.0],
                           [r["linear"]["dead_time_ms"] / 1e3, np.inf, r["linear"]["lag_ms"] / 1e3]])
        Y = simulate(seg["cmd"], seg["ctrl_freq"], seg["t"], *params.T, y0=seg["y0"])
        c = np.rad2deg(seg["meas"][seg["mask"]].mean())
        ax.plot(seg["t"][window], np.rad2deg(seg["meas"][window]) - c, color="tab:blue", lw=1.2, label="real")
        for y, color, label in ((Y[0], "tab:red", "ramp + dead time"), (Y[1], "0.6", "linear (lag + dead time)")):
            ax.plot(seg["t"][window], np.rad2deg(y[window]) - np.rad2deg(y[seg["mask"]].mean()), color=color,
                    lw=1, label=label)
        ax.set_title(f"{joint}: {r['rms_deg']:.2f} vs {r['linear']['rms_deg']:.2f} deg rms", fontsize=9)
    for ax in list(axes.flat)[len(names):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=7)
    fig.suptitle(f"{run_dir.name}: chirp up to {max_chirp_hz:g} Hz (real, servo model, best linear model); "
                 "rms over the scored band")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--joints", nargs="+", help="Default: every joint of the run.")
    parser.add_argument("--max-chirp-hz", type=float, default=2.5,
                        help="Score the chirp up to this frequency (default 2.5). Above it the ramp saturates "
                             "completely and the fit becomes chaotic in the parameters.")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args(argv)
    metadata = json.loads((args.run_dir / "metadata.json").read_text())
    config_path = args.run_dir / "config.yaml"
    config = yaml.safe_load(config_path.read_text()) if config_path.exists() else metadata["config"]
    joints = args.joints or metadata["joints"]
    jobs = [(args.run_dir, metadata, config, joint, args.max_chirp_hz) for joint in joints]
    if args.workers <= 1 or len(jobs) == 1:
        results = dict(map(fit_joint, jobs))
    else:
        with Pool(min(args.workers, len(jobs))) as pool:
            results = dict(pool.imap_unordered(fit_joint, jobs))
    results = {joint: results[joint] for joint in joints}

    print(f"\n{'joint':<12}{'dead ms':>8}{'ramp':>7}{'pred':>7}{'rms':>7}{'linear':>8}   (ramp in rad/s^2, rms in deg)")
    flagged = []
    for joint, r in results.items():
        if r is None:
            print(f"{joint:<12}  no data")
            continue
        pred = r["accel_predicted_rad_s2"]
        flags = []
        if r["clipped"]:
            flags.append("clipped: stopped short of the command (neighbour finger, tendon, current limit)")
        if pred and joint != "wrist" and abs(r["accel_rad_s2"] / pred - 1) > 0.10:
            flags.append("ramp differs from ACC/calibration")
        if flags:
            flagged.append(joint)
        print(f"{joint:<12}{r['dead_time_ms']:>8.1f}{r['accel_rad_s2']:>7.1f}"
              f"{(f'{pred:.1f}' if pred else '-'):>7}{r['rms_deg']:>7.2f}{r['linear']['rms_deg']:>8.2f}"
              + (f"  {'; '.join(flags)}" if flags else ""))
    sim = sim_values(results, flagged)
    output = {"run": args.run_dir.name, "max_chirp_hz": args.max_chirp_hz, "joints": results, "sim": sim,
              "flagged": flagged}
    (args.run_dir / "servo_fit.json").write_text(json.dumps(output, indent=2) + "\n")
    totals = sim["dead_time_plus_lag_ms"]
    print(f"\nSim values (flagged joints left out): dead time + lag {totals['fingers_thumb']} ms "
          f"(wrist {totals['wrist']} ms); ORCA_V2_SERVO_ACCEL = {json.dumps(sim['ORCA_V2_SERVO_ACCEL'])}")
    png = plot(args.run_dir, metadata, results, args.run_dir / "servo_fit.png", args.max_chirp_hz)
    print(f"Wrote {args.run_dir / 'servo_fit.json'}" + (f" and {png}" if png else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
