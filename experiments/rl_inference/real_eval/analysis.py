"""Analysis of one run after it ended, without ROS: the cube's progress over time, how well the joints followed the
policy's commands, warnings about runs whose numbers are not trustworthy, and the plots."""

from dataclasses import dataclass

import numpy as np

STALL_BAND = 0.125  # turns: progress smaller than this counts as no progress
LAGS_S = np.arange(0.0, 0.51, 0.01)  # candidate command-to-motion delays


# -- cube progress over time
def progress_metrics(times: np.ndarray, turns: np.ndarray, t0: float, t_end: float) -> dict:
    """turns_30s: turns after 30 s (empty if the run was shorter); best_10s_rad_per_s: the fastest 10 s;
    longest_stall_s: the longest time in which the cube made less than STALL_BAND turns of progress."""
    t = np.concatenate([[0.0], np.asarray(times) - t0, [t_end - t0]])
    c = np.concatenate([[0.0], np.asarray(turns), [turns[-1] if len(turns) else 0.0]])
    duration = t_end - t0
    turns_30s = round(float(np.interp(30.0, t, c)), 2) if duration >= 30.0 else ""
    best = ""
    if duration >= 10.0:
        starts = np.arange(0.0, duration - 10.0 + 1e-9, 0.25)
        best = round(float(np.max(np.interp(starts + 10.0, t, c) - np.interp(starts, t, c))) * 2 * np.pi / 10.0, 2)
    stall = 0.0
    for i in range(len(t)):
        lo = hi = c[i]
        j = i
        while j + 1 < len(t) and max(hi, c[j + 1]) - min(lo, c[j + 1]) < STALL_BAND:
            j += 1
            lo, hi = min(lo, c[j]), max(hi, c[j])
        stall = max(stall, t[j] - t[i])
    return dict(turns_30s=turns_30s, best_10s_rad_per_s=best, longest_stall_s=round(stall, 1))


# -- joints
@dataclass
class JointTracking:
    name: str
    command_range_deg: float  # 5th to 95th percentile of the command
    measured_range_deg: float
    rms_error_deg: float  # measured - commanded at the same time
    lag_ms: float | None  # delay that best explains the measured motion from the command, None for a still joint
    rms_error_at_lag_deg: float  # what remains after the delay: offset, too little or too much motion
    follows: float  # measured range / command range: well below 1 = the joint does not reach its commands


def joint_tracking(log, joint_names, rate_hz=100.0) -> list[JointTracking]:
    """log: [(time, command, measured)] (rad, the hand controller's joint order)."""
    if len(log) < 50:
        return []
    t = np.array([row[0] for row in log])
    cmd = np.degrees(np.array([row[1] for row in log]))
    meas = np.degrees(np.array([row[2] for row in log]))
    if cmd.shape != meas.shape or cmd.shape[1] != len(joint_names):
        return []
    grid = np.arange(t[0], t[-1], 1.0 / rate_hz)  # the messages arrive with jitter
    cmd = np.stack([np.interp(grid, t, cmd[:, j]) for j in range(cmd.shape[1])], axis=1)
    meas = np.stack([np.interp(grid, t, meas[:, j]) for j in range(meas.shape[1])], axis=1)
    out = []
    for j, name in enumerate(joint_names):
        c, m = cmd[:, j], meas[:, j]
        c_range = float(np.percentile(c, 95) - np.percentile(c, 5))
        m_range = float(np.percentile(m, 95) - np.percentile(m, 5))
        rms = float(np.sqrt(np.mean((m - c) ** 2)))
        lag, rms_lag = None, rms
        if c_range >= 2.0:  # the joint moves: find the delay
            errors = []
            for L in LAGS_S:
                k = int(round(L * rate_hz))
                d = m[k:] - c[:len(c) - k]
                d = d - np.mean(d)  # the delay, not a constant offset
                errors.append(np.sqrt(np.mean(d ** 2)))
            best = int(np.argmin(errors))
            k = int(round(LAGS_S[best] * rate_hz))
            lag = round(float(LAGS_S[best]) * 1000.0)
            rms_lag = float(np.sqrt(np.mean((m[k:] - c[:len(c) - k]) ** 2)))
        out.append(JointTracking(name, round(c_range, 1), round(m_range, 1), round(rms, 1), lag, round(rms_lag, 1),
                                 round(m_range / c_range, 2) if c_range >= 2.0 else 1.0))
    return out


def joint_summary(tracking: list[JointTracking]) -> dict:
    """The results.csv fields: median delay, the joint that follows worst, joints that barely move."""
    if not tracking:
        return dict(median_lag_ms="", worst_joint="", worst_joint_error_deg="", stuck_joints="")
    lags = [j.lag_ms for j in tracking if j.lag_ms is not None]
    worst = max(tracking, key=lambda j: j.rms_error_at_lag_deg)
    stuck = [j.name for j in tracking if j.command_range_deg >= 5.0 and j.follows < 0.5]
    return dict(median_lag_ms=round(float(np.median(lags))) if lags else "", worst_joint=worst.name,
                worst_joint_error_deg=worst.rms_error_at_lag_deg, stuck_joints=" ".join(stuck))


# -- trust
def quality_warnings(result: dict, min_tracked: float, max_gap: float, max_rejected: float = 0.3) -> list[str]:
    """Reasons to distrust the turn count of a run that was not stopped early by a failure."""
    warnings = []
    if result.get("duration_s", 0) < 1.0:
        return warnings
    if result["tracked_fraction"] < min_tracked:
        warnings.append(f"low tracking ({result['tracked_fraction']:.0%})")
    if result["longest_gap_s"] > max_gap:
        warnings.append(f"no detection for {result['longest_gap_s']:.1f} s")
    if result["rejected_poses"] > max_rejected:
        warnings.append(f"{result['rejected_poses']:.0%} of the poses rejected")
    return warnings


# -- plots
def plot_run(run_dir, title, times, turns, t0, t_end, goal_rad_per_s, temperatures, joint_names, joint_log,
             stop_reason):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, ax_t) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True, height_ratios=[3, 1.3])
    t = np.concatenate([[0.0], np.asarray(times) - t0])
    c = np.concatenate([[0.0], np.asarray(turns)])
    ax.step(t, c, where="post", color="tab:blue", label="cube (about the spin axis)")
    ax.plot(t, c, ".", color="tab:blue", ms=2)
    span = np.array([0.0, t_end - t0])
    ax.plot(span, abs(goal_rad_per_s) * span / (2 * np.pi), "--", color="gray", label=f"goal {abs(goal_rad_per_s):.1f} rad/s")
    ax.axvline(t_end - t0, color="tab:red", lw=1)
    ax.text(t_end - t0, ax.get_ylim()[1], f" {stop_reason}", color="tab:red", va="top", fontsize=8)
    ax.set_ylabel("turns")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    if temperatures:
        temps = np.array(temperatures)
        hottest = np.argsort(temps[:, 1:].max(axis=0))[::-1][:3]
        for j in hottest:
            ax_t.plot(temps[:, 0], temps[:, 1 + j], label=joint_names[j])
        ax_t.legend(loc="upper left", fontsize=7)
    ax_t.set_ylabel("motor (C)")
    ax_t.set_xlabel("time (s)")
    ax_t.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "turns_vs_time.png", dpi=110)
    plt.close(fig)

    if len(joint_log) < 50:
        return
    jt = np.array([row[0] for row in joint_log]) - t0
    cmd = np.degrees(np.array([row[1] for row in joint_log]))
    meas = np.degrees(np.array([row[2] for row in joint_log]))
    n = len(joint_names)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 2.2 * rows), sharex=True)
    for j, a in enumerate(np.ravel(axes)):
        if j >= n:
            a.axis("off")
            continue
        a.plot(jt, cmd[:, j], color="tab:orange", lw=0.8, label="command")
        a.plot(jt, meas[:, j], color="tab:blue", lw=0.8, label="measured")
        a.set_title(joint_names[j], fontsize=9)
        a.tick_params(labelsize=7)
        a.grid(alpha=0.3)
    np.ravel(axes)[0].legend(fontsize=7)
    fig.supxlabel("time (s)")
    fig.supylabel("deg")
    fig.tight_layout()
    fig.savefig(run_dir / "joints.png", dpi=100)
    plt.close(fig)
