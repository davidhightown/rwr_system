"""Per-joint figures of a run from the joint log: how late and how far off each joint follows its command, and the
motor currents. No ROS."""

import numpy as np

STATUS_BITS = ((0x01, "input_voltage"), (0x02, "angle_sensor"), (0x04, "overheating"), (0x08, "overcurrent"),
               (0x20, "overload"))  # the Feetech status byte


def status_names(byte):
    byte = int(byte)
    names = [name for bit, name in STATUS_BITS if byte & bit]
    rest = byte & ~sum(bit for bit, _ in STATUS_BITS)
    return names + ([f"0x{rest:02X}"] if rest else [])


def joint_tracking(times, commands, measured, currents, joint_names, max_delay=0.3, dt=0.01):
    """One dict per joint: delay_ms (the shift of the measurement that fits the command best), error_rms_deg (after
    that shift), error_max_deg, command_range_deg, measured_range_deg, current_mean_ma, current_peak_ma.

    times: seconds (n); commands, measured: radians (n, joints); currents: mA (n, joints) or None. The log is
    resampled to ``dt``. A joint whose command barely moves (range under 2 deg) gets no delay."""
    times = np.asarray(times, dtype=float)
    commands, measured = np.asarray(commands, dtype=float), np.asarray(measured, dtype=float)
    if len(times) < 50:
        return []
    grid = np.arange(times[0], times[-1], dt)
    shifts = np.arange(0, int(max_delay / dt) + 1)
    out = []
    for j, name in enumerate(joint_names):
        cmd = np.degrees(np.interp(grid, times, commands[:, j]))
        meas = np.degrees(np.interp(grid, times, measured[:, j]))
        row = dict(joint=name, delay_ms="", command_range_deg=round(float(np.ptp(cmd)), 1),
                   measured_range_deg=round(float(np.ptp(meas)), 1))
        best = 0
        if np.ptp(cmd) >= 2.0 and len(grid) > 2 * len(shifts):
            errors = [np.sqrt(np.mean((meas[s:] - cmd[:len(cmd) - s]) ** 2)) for s in shifts]
            best = int(np.argmin(errors))
            row["delay_ms"] = round(best * dt * 1000)
        error = meas[best:] - cmd[:len(cmd) - best]
        row["error_rms_deg"] = round(float(np.sqrt(np.mean(error ** 2))), 2)
        row["error_max_deg"] = round(float(np.max(np.abs(error))), 2)
        if currents is not None:
            cur = np.abs(np.asarray(currents, dtype=float)[:, j])
            row["current_mean_ma"] = round(float(np.nanmean(cur)))
            row["current_peak_ma"] = round(float(np.nanmax(cur)))
        out.append(row)
    return out


STALL_BAND = 0.125  # turns: less progress than this counts as none (as in the other evaluation's analysis.py)


def progress_metrics(times, turns, duration):
    """The cube's progress over policy time, defined as in the other evaluation (feat/real-hand-evaluation):
    turns_30s (turns after 30 s, "" for a shorter run), best_10s_rad_per_s (the fastest 10 s), longest_stall_s (the
    longest time with less than STALL_BAND turns of progress). times: policy seconds of the counted detections,
    turns: the turns at those times (signed: positive = the commanded direction)."""
    times, turns = np.asarray(times, dtype=float), np.asarray(turns, dtype=float)
    t = np.concatenate([[0.0], times, [duration]])
    c = np.concatenate([[0.0], turns, [turns[-1] if len(turns) else 0.0]])
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
    return dict(turns_30s=turns_30s, best_10s_rad_per_s=best, longest_stall_s=round(float(stall), 1))


def tracking_summary(rows):
    """The CSV fields of a run from joint_tracking()'s rows."""
    if not rows:
        return {}
    delays = [r["delay_ms"] for r in rows if r["delay_ms"] != ""]
    worst = max(rows, key=lambda r: r["error_rms_deg"])
    out = dict(joint_delay_median_ms=round(float(np.median(delays))) if delays else "",
               joint_error_rms_deg=round(float(np.sqrt(np.mean([r["error_rms_deg"] ** 2 for r in rows]))), 2),
               worst_joint=worst["joint"], worst_joint_error_rms_deg=worst["error_rms_deg"])
    if "current_mean_ma" in rows[0]:
        highest = max(rows, key=lambda r: r["current_mean_ma"])
        out.update(current_total_mean_a=round(sum(r["current_mean_ma"] for r in rows) / 1000, 2),
                   current_peak_ma=max(r["current_peak_ma"] for r in rows),
                   highest_current_joint=highest["joint"])
    return out
