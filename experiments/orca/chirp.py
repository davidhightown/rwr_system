"""Collect ORCA v2 joint chirp data through orca_core, without ROS.

By default, only validate and print the experiment plan. Pass --execute to
connect to a previously calibrated hand and collect real measurements.

Commands go out at --ctrl-freq (20 Hz, as in deployment). The joint is read
right after every command (one row of the chirp CSV, the format the faive_lab
fit reads) and in between at --read-hz (the fast CSV, with the motor's velocity
and current). Before recording, the servo acceleration ramp from config.yaml is
written to every motor and read back (see src/hand_control/orca/servo_profile.py).
"""

import argparse
import csv
from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy import signal
import yaml

# Shared with the deployment controller; plain Python, so it loads without installing faive_system.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src/hand_control/orca"))
import servo_profile  # noqa: E402


DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "src/hand_control/orca/configs/orcahand_v2/right/config.yaml"
)
CSV_FIELDS = ["timestamp", "commanded_position", "measured_position"]
FAST_CSV_FIELDS = CSV_FIELDS + ["motor_velocity", "motor_current"]


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="This hand's config.yaml; calibration.yaml must be beside it.")
    parser.add_argument("--joints", nargs="+", default=["index_mcp"],
                        help="Joint names to sweep sequentially (ORCA v2 uses thumb_cmc).")
    parser.add_argument("--ctrl-freq", type=float, default=20.0)
    parser.add_argument("--read-hz", type=float, default=100.0,
                        help="Read rate of the fast CSV, a whole multiple of --ctrl-freq. 100 Hz is the read rate "
                             "of the deployment controller, so the bus carries the same load. Equal to "
                             "--ctrl-freq: no reads between commands.")
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--f0", type=float, default=0.1)
    parser.add_argument("--f1", type=float, default=5.0)
    parser.add_argument("--amplitude-deg", type=float, default=10.0)
    parser.add_argument("--center-deg", type=float, default=None,
                        help="Sweep center; by default use each joint's ROM midpoint.")
    parser.add_argument("--steps-deg", type=float, nargs="*", default=[],
                        help="After each chirp, steps of these sizes about the center (+s, back, -s, back), "
                             "streamed at --ctrl-freq. They show the dead time and the acceleration ramp directly.")
    parser.add_argument("--step-hold", type=float, default=0.6, help="Seconds to hold each step target.")
    parser.add_argument("--settle-time", type=float, default=1.0)
    parser.add_argument("--no-ramp", action="store_true",
                        help="Jump straight to each joint's first chirp sample instead of ramping for about 1 s. "
                             "The previous joint also snaps back to neutral. Moves at full motor speed.")
    parser.add_argument("--output-dir", type=Path, default=Path("logs/chirp_v2"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="Connect and move real hardware.")
    mode.add_argument("--dry-run", action="store_true", help="Print the plan only (the default).")
    return parser


def make_plan(args, config):
    """Reject invalid or clipped experiments before constructing hardware."""
    for name in ("ctrl_freq", "duration", "f0", "f1", "amplitude_deg", "read_hz", "step_hold"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(args.settle_time) or args.settle_time < 0:
        raise ValueError("settle_time must be finite and non-negative")
    if not args.f0 < args.f1 < args.ctrl_freq / 2:
        raise ValueError("Require 0 < f0 < f1 < ctrl_freq / 2 (Nyquist frequency)")
    if args.duration * args.ctrl_freq < 2:
        raise ValueError("The experiment must contain at least two samples")
    reads = args.read_hz / args.ctrl_freq
    if reads < 1 or abs(reads - round(reads)) > 1e-9:
        raise ValueError("read_hz must be a whole multiple of ctrl_freq")
    if any(not math.isfinite(step) or step <= 0 for step in args.steps_deg):
        raise ValueError("steps_deg must be finite and positive")
    if len(set(args.joints)) != len(args.joints):
        raise ValueError("Do not repeat joints in --joints")

    joint_ids = config["joint_ids"]
    if not joint_ids or len(set(joint_ids)) != len(joint_ids):
        raise ValueError("config joint_ids must be non-empty and unique")
    unknown = set(args.joints) - set(joint_ids)
    if unknown:
        raise ValueError(f"Unknown joints: {sorted(unknown)}. Available: {joint_ids}")
    neutral = config["neutral_position"]
    roms = config["joint_roms"]
    for joint in joint_ids:
        lower, upper = roms[joint]
        position = neutral[joint]
        if not all(math.isfinite(x) for x in (lower, upper, position)):
            raise ValueError(f"Non-finite ROM or neutral position for {joint}")
        if lower >= upper or not lower <= position <= upper:
            raise ValueError(f"Invalid ROM or neutral position for {joint}")

    reach = max([args.amplitude_deg, *args.steps_deg])
    centers = {}
    for joint in args.joints:
        lower, upper = roms[joint]
        center = (lower + upper) / 2 if args.center_deg is None else args.center_deg
        if not math.isfinite(center) or center - reach < lower or center + reach > upper:
            raise ValueError(f"{joint}: center +/- amplitude or step exceeds ROM [{lower}, {upper}]")
        centers[joint] = center
    return {
        "config_path": str(args.config.resolve()),
        "config": config,
        "joints": args.joints,
        "centers_deg": centers,
        "ctrl_freq_hz": args.ctrl_freq,
        "read_hz": args.read_hz,
        "duration_s": args.duration,
        "f0_hz": args.f0,
        "f1_hz": args.f1,
        "amplitude_deg": args.amplitude_deg,
        "steps_deg": list(args.steps_deg),
        "step_hold_s": args.step_hold,
        "settle_time_s": args.settle_time,
        "ramp_to_start": not args.no_ramp,
        "sample_count_per_joint": len(sample_times(args)),
        "servo_profile": servo_profile.describe(config),
        "max_current": config.get("max_current", "default"),
    }


def sample_times(args):
    return np.arange(0, args.duration, 1.0 / args.ctrl_freq)


def step_targets(args, center):
    """Piecewise-constant targets at ctrl_freq: center, then +s, center, -s, center for each step size."""
    hold = max(1, round(args.step_hold * args.ctrl_freq))
    levels = [center]
    for step in args.steps_deg:
        levels += [center + step, center, center - step, center]
    return np.repeat(levels, hold).astype(float)


def joint_motor_index(config, joint):
    return list(config["motor_ids"]).index(abs(config["joint_to_motor_map"][joint]))


def read_joint(hand, joint, motor_index):
    """One bus transaction: joint position (deg), the joint motor's velocity (rad/s) and current (mA)."""
    state = hand.get_motor_state()
    measured = hand._motor_to_joint_pos(state.position).get(joint)
    if measured is None or not math.isfinite(measured):
        raise RuntimeError(f"No finite position measurement for {joint}")
    return float(measured), float(state.velocity[motor_index]), float(state.current[motor_index])


def stream(hand, args, joint, motor_index, command, targets, writer, fast_writer, *, clock, sleep):
    """Send one target per control period; read after each command (writer) and between commands at read_hz.

    Every row is flushed at once, so partial data survives a failure. A late command stops the run: catching
    up in a burst would invalidate the excitation. A late read in between is skipped instead.
    """
    period = 1.0 / args.ctrl_freq
    reads = round(args.read_hz / args.ctrl_freq)
    start = clock()
    for k, position in enumerate(targets):
        deadline = start + k * period
        sleep(max(0.0, deadline - clock()))
        if clock() - deadline >= period:
            raise RuntimeError("Missed a full control period; stopped instead of sending a burst. "
                               "Check bus latency and the partial CSV timestamps.")
        command[joint] = float(position)
        hand.set_joint_positions(command)
        for j in range(reads):
            if j:
                read_at = deadline + j * period / reads
                if clock() > read_at + 0.5 * period / reads:
                    continue
                sleep(max(0.0, read_at - clock()))
            measured, velocity, current = read_joint(hand, joint, motor_index)
            now = clock() - start
            if j == 0 and writer is not None:
                writer.writerow([now, float(position), measured])
            if fast_writer is not None:
                fast_writer.writerow([now, float(position), measured, velocity, current])
    # Hold the last target for a full period before changing pose or disabling torque.
    sleep(max(0.0, start + len(targets) * period - clock()))


class _Csv:
    """CSV file that flushes every row."""

    def __init__(self, path, fields):
        self.stream = path.open("x", newline="")
        self.writer = csv.writer(self.stream)
        self.writerow(fields)

    def writerow(self, row):
        self.writer.writerow(row)
        self.stream.flush()

    def close(self):
        self.stream.close()


def collect_joint(hand, args, plan, joint, paths, *, clock=time.monotonic, sleep=time.sleep):
    """Chirp one joint, then run its steps. paths: 'chirp', and optionally 'fast' and 'steps'."""
    t = sample_times(args)
    trajectory = plan["centers_deg"][joint] + args.amplitude_deg * signal.chirp(
        t, f0=args.f0, t1=args.duration, f1=args.f1, method="logarithmic"
    )
    motor_index = joint_motor_index(plan["config"], joint)
    command = dict(plan["config"]["neutral_position"])
    command[joint] = float(trajectory[0])
    if args.no_ramp:
        hand.set_joint_positions(command)
    else:
        # Ramp for roughly one second rather than jumping to the first chirp sample.
        hand.set_joint_positions(command, num_steps=max(2, math.ceil(args.ctrl_freq)),
                                 step_size=1.0 / args.ctrl_freq)
    sleep(args.settle_time)

    files = []
    try:
        files.append(_Csv(paths["chirp"], CSV_FIELDS))
        chirp_csv = files[-1]
        fast_csv = None
        if paths.get("fast"):
            files.append(_Csv(paths["fast"], FAST_CSV_FIELDS))
            fast_csv = files[-1]
        stream(hand, args, joint, motor_index, command, trajectory, chirp_csv, fast_csv, clock=clock, sleep=sleep)
        if paths.get("steps"):
            files.append(_Csv(paths["steps"], FAST_CSV_FIELDS))
            stream(hand, args, joint, motor_index, command, step_targets(args, plan["centers_deg"][joint]),
                   None, files[-1], clock=clock, sleep=sleep)
    finally:
        for f in files:
            f.close()


def read_servo_gains(hand):
    """The servos' own position-loop gains by motor ID, for the record; None if the read fails."""
    try:
        return {str(motor): (None if gains is None else {key: value for key, value in vars(gains).items()})
                for motor, gains in hand.get_servo_gains().items()}
    except Exception:
        return None


def read_temperatures(hand):
    """Motor temperatures in degrees C by motor ID; None if the read fails, which must not stop a run."""
    try:
        return {str(motor): float(temp) for motor, temp in hand.get_motor_temp(as_dict=True).items()}
    except Exception:
        return None


def joint_paths(args, run_dir, stamp, joint):
    suffix = (f"{joint}_{stamp}_ctrl{args.ctrl_freq:g}Hz_"
              f"{args.duration}s_{args.f0}-{args.f1}Hz_{args.amplitude_deg:g}deg.csv")
    paths = {"chirp": run_dir / f"chirp_{suffix}"}
    if round(args.read_hz / args.ctrl_freq) > 1:
        paths["fast"] = run_dir / f"fast_{suffix}"
    if args.steps_deg:
        paths["steps"] = run_dir / f"steps_{joint}_{stamp}_ctrl{args.ctrl_freq:g}Hz.csv"
    return paths


def run_hardware(args, plan, *, factory=None, clock=time.monotonic, sleep=time.sleep, on_run_dir=None):
    """on_run_dir, if given, is called with the run directory as soon as it exists."""
    if factory is None:
        # Preview/help work without orca_core, ROS or any serial access.
        from orca_core import load_hand
        factory = load_hand

    hand = factory(config_path=str(args.config.resolve()))
    metadata = dict(plan, status="starting", csv_files=[], fast_csv_files=[], step_csv_files=[])
    run_dir = None
    try:
        # init_joints() can auto-calibrate. Refuse to enter it without a complete
        # existing calibration: calibration is a separate, deliberate operation.
        if not hand.calibrated or not hand.is_calibrated():
            raise RuntimeError("A complete calibration.yaml for this hand is required beside config.yaml. "
                               "Calibrate separately before running chirp.")
        ok, message = hand.connect(interactive=False)
        if not ok:
            raise RuntimeError(f"Could not connect: {message}")
        print(message)

        started_at = datetime.now(timezone.utc)
        stamp = started_at.strftime("%Y%m%d_%H%M%S_%f")
        run_dir = args.output_dir / stamp
        run_dir.mkdir(parents=True, exist_ok=False)
        if on_run_dir is not None:
            on_run_dir(run_dir)
        metadata.update(status="running", orca_core_version=version("orca_core"),
                        run_id=stamp, started_at_utc=started_at.isoformat(),
                        motor_temperature_start_c=read_temperatures(hand))
        (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        hand.init_joints(force_calibrate=False, move_to_neutral=True)
        # init_joints writes orca_core's own ramp: replace it with the configured one and record what the motors hold.
        metadata["servo_profile"] = servo_profile.apply(hand, plan["config"])
        metadata["servo_gains"] = read_servo_gains(hand)
        print(f"Servo acceleration {metadata['servo_profile'][servo_profile.CONFIG_KEY]} (raw) on every motor")
        (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        for joint in args.joints:
            paths = joint_paths(args, run_dir, stamp, joint)
            metadata["csv_files"].append(paths["chirp"].name)
            for key, listing in (("fast", "fast_csv_files"), ("steps", "step_csv_files")):
                if key in paths:
                    metadata[listing].append(paths[key].name)
            print(f"Collecting {joint}: {paths['chirp']}")
            collect_joint(hand, args, plan, joint, paths, clock=clock, sleep=sleep)
        metadata["status"] = "complete"
    except BaseException as exc:
        metadata.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                        error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        # Always attempt both operations, including if connect/init fails.
        active_exception = sys.exc_info()[0] is not None
        cleanup_errors = []
        operations = []
        if hand.is_connected():
            metadata["motor_temperature_end_c"] = read_temperatures(hand)
            operations.append(("disable torque", hand.disable_torque))
        operations.append(("disconnect", hand.disconnect))
        for label, operation in operations:
            try:
                result = operation()
                if label == "disable torque" and result:
                    raise RuntimeError(f"Torque disable not acknowledged by motors {result}")
                if label == "disconnect" and result is not None and not result[0]:
                    raise RuntimeError(result[1])
            except Exception as exc:
                cleanup_errors.append(str(exc))
                print(f"Hardware cleanup ({label}) failed: {exc}", file=sys.stderr)
        if cleanup_errors:
            metadata.update(status="failed", cleanup_errors=cleanup_errors)
        if run_dir is not None:
            (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
            print(f"Experiment {metadata['status']}; data: {run_dir}")
        if cleanup_errors and not active_exception:
            raise RuntimeError("Hardware cleanup failed; check the hand connection")
    return run_dir


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        with args.config.open() as stream:
            config = yaml.safe_load(stream)
        plan = make_plan(args, config)
    except (OSError, ValueError, TypeError, KeyError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    print(json.dumps({key: value for key, value in plan.items() if key != "config"}, indent=2))
    if not args.execute:
        print("Preview only. Add --execute to connect and move the calibrated hand.")
        return 0
    try:
        run_hardware(args, plan)
    except KeyboardInterrupt:
        print("Stopped all sweeps; partial CSV data has been kept.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Chirp failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
