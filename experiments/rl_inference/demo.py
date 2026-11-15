#!/usr/bin/env python3
"""Demo mode of the ORCA v2 hand: a policy spins the cube in front of an audience, with a live dashboard.

    python experiments/rl_inference/demo.py --mode proprio                  # no camera: the policy of the 74 inputs
    python experiments/rl_inference/demo.py --mode extero                   # camera plugged in: the cube pose policy
    python experiments/rl_inference/demo.py --mode optional                 # optional.task in the config (axes)
    python experiments/rl_inference/demo.py --mode optional --task goals    # random goal orientations
    python experiments/rl_inference/demo.py --mode extero --dry-run         # only the pre-flight check
    python experiments/rl_inference/demo.py --mode extero --mock            # rehearsal: fake hand, drawn cube

Everything a demo uses (policy file, servo acceleration, action alpha, speed, temperatures, camera calibration) is in
one YAML file, experiments/cfgs/demo.yaml (--config another one). Built on eval_policies.py: the same start pose, hand
controller, camera process, cube detector and result files, with these differences:

- every demo is `repeats` runs (3) of `duration_s` (60 s of policy time) with the same policy; optional/axes runs once
  about each axis of its list (x, y, z), optional/goals `repeats` times with new random goals each time,
- drop detection is off: the policy keeps running when the cube falls (P parks the hand so it can be put back),
- a run starts only when every motor is at most `start_temp_c` warm (lower than the evaluation's 42.5 C), and after
  the operator pressed SPACE (cube in the palm),
- a full-screen dashboard (cube_eval/dashboard.py) instead of the evaluation's camera window; in proprio without a
  camera it shows the hand's digital twin, the joint angles and the motors,
- nothing goes to W&B; results in common.out (default logs/demo): results.csv and a folder per run.

Keys in the dashboard: SPACE = start / continue, P = pause (hand to the start pose, torque off), E = emergency stop
(torque off where the hand is), N = next run, Q = quit, F = full screen on/off. Ctrl+C in the terminal stops
everything and leaves the hand with the torque off.

For the best performance: the pre-flight check runs before the hand moves (policy files load and fit the mode,
camera calibration present), the policy runs on one CPU thread (common.torch_threads) and is warmed up, RViz and W&B
are off, the motors start cool, and on macOS `caffeinate` keeps the laptop from sleeping or throttling the process
(App Nap) while the demo runs. The dashboard shows the policy loop's rate and jitter: it should stay at 20 Hz.
"""

import argparse
import json
import contextlib
import io
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

import eval_policies as ev  # noqa: E402
from eval_policies import STALE_JOINTS_S, Evaluation, Quit, Run, Skip  # noqa: E402

DEFAULT_CONFIG = REPO / "experiments/cfgs/demo.yaml"
MODE_LABELS = dict(proprio="PROPRIOCEPTIVE", extero="EXTEROCEPTIVE (camera)", axes="OPTIONAL - ALL AXES",
                   goals="OPTIONAL - GOAL ORIENTATIONS")
KEYS = "SPACE start   P pause   E stop   N next run   Q quit   F fullscreen"
WARMUP_STEPS = 50
PROGRESS_EVERY_S = 10.0  # a progress line this often (policy time) while a run runs
COOLING_REPORT_S = 15.0  # a line about the hottest motors this often while cooling
COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
STYLES = dict(info="", ok="32", warn="33", error="31;1", head="36;1", dim="2", key="35")


def log(message, style="info"):
    """One line in the terminal: the time, then the message (coloured in a terminal, unless NO_COLOR is set)."""
    code = STYLES.get(style, "")
    text = f"\033[{code}m{message}\033[0m" if COLOR and code else message
    print(f"{datetime.now():%H:%M:%S}  {text}", flush=True)


def rule(title, style="head"):
    log(f"==== {title} {'=' * max(4, 100 - len(title))}", style)


class Progress:
    """The run's progress line in the terminal, every ``every`` seconds of policy time."""

    def __init__(self, demo, run, duration, every):
        self.demo, self.run, self.duration, self.every = demo, run, duration, every
        self.next = every
        self.loop_hz = []           # the policy loop's rate at every report, for the end of the run
        self.detections = 0
        self.last = (0.0, 0)        # (policy time, detections) at the last report

    def maybe_report(self):
        if self.every > 0 and self.run.elapsed() >= self.next:
            self.next += self.every
            self.report()

    def report(self, final=False):
        d, run, hand = self.demo, self.run, self.demo.hand
        t = min(run.elapsed(), self.duration)
        parts = [f"{t:5.1f}/{self.duration:.0f} s"]
        f = run.figures()
        if run.goals is not None:
            err = f.get("goal_error_deg")
            parts.append(f"goals {f.get('goals', 0)}, to the goal {'-' if err is None else f'{err:.0f} deg'}")
        elif d.args.camera:
            parts.append(f"turns {f.get('turns', 0.0):+6.2f}, speed {f.get('speed', 0.0):+.2f}/{run.speed:+.1f} rad/s")
        if d.args.camera:
            n = len(run.detection_elapsed)
            dt = t - self.last[0]
            parts.append(f"cube {(n - self.last[1]) / dt:4.1f} poses/s" if dt > 0 else "cube -")
            self.last = (t, n)
        loop = hand.loop_stats()
        if loop is not None:
            self.loop_hz.append(loop["hz"])
            parts.append(f"loop {loop['hz']:4.1f} Hz (jitter {loop['jitter_ms']:.1f} ms, step {loop['infer_ms']:.1f} ms)")
        elif not final:
            parts.append("loop -")
        status = d.extra_status()["demo"]
        if status["track_err_deg"] is not None:
            parts.append(f"tracking {status['track_err_deg']:.1f} deg")
        temps = hand.temperatures()
        if temps is not None:
            hot = int(np.nanargmax(temps))
            parts.append(f"hottest {d.joint_ids[hot]} {temps[hot]:.0f} C")
        style = "ok" if loop is None or loop["hz"] >= 18.0 else "warn"
        if final and len(self.loop_hz) > 1:
            parts.append(f"loop over the run: mean {np.mean(self.loop_hz):.1f} Hz, min {np.min(self.loop_hz):.1f} Hz")
        log("  " + " | ".join(parts), style)
        if loop is not None and loop["hz"] < 18.0:
            log(f"  WARNING: the policy loop runs at {loop['hz']:.1f} Hz instead of 20 (laptop busy? on battery?)",
                "warn")


class EmergencyStop(Exception):
    """E was pressed: the torque goes off where the hand is."""


class Pause(Exception):
    """P was pressed: the hand goes to the start pose, so the cube can be put back."""


# ---------------------------------------------------------------------------------------------------- configuration
def repo_path(value):
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else REPO / path


def load_plan(config_path, mode, task=None):
    """The demo to run, from the YAML: dict(kind = proprio|extero|axes|goals, common, section (that kind's settings,
    with the camera's), runs = [(label, axis)])."""
    config = yaml.safe_load(Path(config_path).read_text())
    common = dict(config.get("common") or {})
    if mode == "optional":
        task = task or (config.get("optional") or {}).get("task", "axes")
        if task not in ("axes", "goals"):
            raise SystemExit(f"optional.task must be axes or goals, not {task!r}")
        section = dict((config["optional"] or {})[task])
        kind = task
    else:
        if task:
            raise SystemExit("--task goes with --mode optional")
        section, kind = dict(config[mode]), mode
    if kind != "proprio":
        section = {**(config.get("camera") or {}), **section}
    repeats = int(common.get("repeats", 3))
    if kind == "axes":
        axes = list(section.get("axes") or ["x", "y", "z"])
        if any(a not in ("x", "y", "z") for a in axes):
            raise SystemExit(f"optional.axes.axes: x, y or z, not {axes}")
        runs = [(f"about {a}", a) for a in axes]
    else:
        runs = [(f"run {i + 1}", "z") for i in range(repeats)]
    return dict(kind=kind, common=common, section=section, runs=runs, config_path=str(config_path))


def eval_args(plan, cli):
    """eval_policies.py's arguments for this demo (its checks run on them), plus the demo's own."""
    c, s, kind = plan["common"], plan["section"], plan["kind"]
    model = repo_path(s["model"])
    argv = [str(model), "--duration", str(c.get("duration_s", 60)), "--out", str(repo_path(c.get("out", "logs/demo"))),
            "--temp-limit", str(c.get("temp_limit_c", 65.0)), "--temp-margin", str(c.get("temp_margin_c", 2.0)),
            "--action-alpha", str(s.get("action_alpha", 0.2)), "--cube-edge", str(c.get("cube_edge", 0.045)),
            "--twin" if c.get("twin", True) else "--no-twin", "--rviz" if c.get("rviz", False) else "--no-rviz",
            "--note", f"demo {kind}"]
    if s.get("servo_acc") is not None:
        argv += ["--servo-acc", str(int(s["servo_acc"]))]
    if c.get("start_wrist_deg") is not None:
        argv += ["--start-wrist-deg", str(c["start_wrist_deg"])]
    if kind != "goals":
        argv += ["--speed", str(s.get("speed") if s.get("speed") is not None else 1.0)]  # per axis: see speed_for
    if kind == "axes":
        argv += ["--mode", "all-axes"]
    if kind == "goals":
        argv += ["--mode", "goals", "--goal-tolerance", str(s.get("goal_tolerance", 0.4))]
    if kind != "proprio":
        if s.get("calibration") and not cli.mock:
            argv += ["--calibration", str(repo_path(s["calibration"]))]
        if s.get("hand_to_tag") is not None:
            argv += ["--hand-to-tag", *map(str, s["hand_to_tag"])]
        argv += ["--exposure-us", str(s.get("exposure_us", 5000)), "--iso", str(s.get("iso", 400))]
    if cli.mock:
        argv += ["--mock"]
    args = ev.parse_args(argv)
    args.camera = kind != "proprio"
    args.dashboard, args.fullscreen = True, bool(c.get("fullscreen", True)) and not cli.windowed
    return args


def speed_for(plan, policy_path, axis):
    """The commanded speed of a run (rad/s): the config's, else the top of the training range about the axis."""
    s = plan["section"]
    if s.get("speed") is not None:
        return float(s["speed"])
    ranges = ((ev.training_info(policy_path).get("commands") or {}).get("continuous_rotation") or {}).get(
        "angular_vel_range") or [None] * 3
    top = ranges["xyz".index(axis)]
    return float(top) if top else ev.wandb_results.DEFAULT_SPEED


def preflight(plan, args):
    """Everything that can be checked before the hand moves. Returns the problems (empty: go)."""
    kind, policy = plan["kind"], args.policies[0]
    problems = []
    print(f"Pre-flight check: {MODE_LABELS[kind]}, {plan['config_path']}")
    print(f"  policy        {policy}")
    why = ev.not_deployable(policy, args.calibration is not None or args.mock, args.mode, args.action_alpha)
    if why:
        problems.append(f"the policy cannot run in this mode: {why}")
    needs = ev.policy_needs(policy)
    if kind == "proprio" and needs["cube_pose"]:
        problems.append("the proprio demo has no camera, but the policy takes the cube pose")
    if kind == "extero" and not needs["cube_pose"]:
        print("  note          the extero demo runs a policy without the cube pose: the camera only counts turns")
    if kind != "proprio" and args.calibration is None and not args.mock:
        problems.append("no camera calibration (camera.calibration in the config): run apriltag_calibration.py")
    speeds = {axis: speed_for(plan, policy, axis) for _, axis in plan["runs"]} if kind != "goals" else {}
    print(f"  runs          {len(plan['runs'])} x {args.duration:.0f} s: " + ", ".join(
        label + (f" ({speeds[axis]:+.1f} rad/s)" if speeds else "") for label, axis in plan["runs"]))
    acc = "config.yaml's" if args.servo_acc is None else args.servo_acc
    print(f"  servo acc     {acc}    action alpha {args.action_alpha:g}    start below "
          f"{plan['common'].get('start_temp_c', 38.0):.1f} C    limit {args.temp_limit:.0f} C")
    print(f"  camera        {'none (window only)' if not args.camera else 'mock' if args.mock else 'OAK'}"
          + (f", calibration {args.calibration}" if args.calibration else ""))
    for p in problems:
        print(f"  PROBLEM       {p}")
    if not problems:
        print("  ok")
    return problems


def keep_awake():
    """macOS: no sleep and no App Nap for this process while the demo runs (caffeinate ends with it)."""
    if sys.platform == "darwin" and shutil.which("caffeinate"):
        return subprocess.Popen(["caffeinate", "-dimsu", "-w", str(os.getpid())], stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
    return None


# ---------------------------------------------------------------------------------------------------- the demo
class Demo(Evaluation):
    def __init__(self, args, plan):
        super().__init__(args)
        self.plan, self.kind = plan, plan["kind"]
        c = plan["common"]
        self.start_temp = float(c.get("start_temp_c", 38.0))
        self.countdown_s = float(c.get("countdown_s", 3))
        self.record_video = bool(c.get("record_video", True)) and args.camera
        self.results = [None] * len(plan["runs"])
        self.rows = [None] * len(plan["runs"])
        self.run_number, self.state, self.countdown = 0, "starting", None
        self.summarized = False
        self.progress_every = float(getattr(args, "progress_every", PROGRESS_EVERY_S))
        self.policy_name, self.task = Path(args.policies[0]).stem.removesuffix("_policy"), ""
        config = yaml.safe_load(ev.HAND_CONFIG.read_text())
        roms = config["joint_roms"]
        self.lo = [float(roms[j][0]) for j in self.joint_ids]
        self.hi = [float(roms[j][1]) for j in self.joint_ids]
        self.session_dir = self.out / f"{datetime.now():%Y%m%d_%H%M%S}_demo_{self.kind}"
        acc = "config" if args.servo_acc is None else str(args.servo_acc)
        self.params = [["servo acc", acc], ["action alpha", f"{args.action_alpha:g}"]]
        if self.kind == "goals":
            self.params.append(["goal tolerance", f"{np.degrees(args.goal_tolerance or 0.4):.0f} deg"])
        self.params += [["start below", f"{self.start_temp:.0f} C"], ["limit", f"{args.temp_limit:.0f} C"]]

    # -- the dashboard's figures, with every status push (eval_policies.Evaluation.tick)
    def extra_status(self):
        h = self.hand
        q = cmd = current = None
        loop = track = None
        voltage = None
        if h is not None:
            with h.lock:
                q = None if h.q is None else np.degrees(h.q)
                cmd = None if h.last_command is None or not h.policy_on else np.degrees(h.last_command)
                current = None if h.current is None else h.current.copy()
            loop = h.loop_stats()
            if q is not None and cmd is not None and len(q) == len(cmd):
                track = float(np.nanmean(np.abs(cmd - q)))
            if h.voltage is not None and np.any(np.isfinite(h.voltage)):
                voltage = float(np.nanmin(h.voltage))
        tolist = lambda x: None if x is None else [float(v) for v in x]  # noqa: E731  (NaN: no reading)
        return dict(demo=dict(
            title="ORCA Hand - in-hand cube rotation", mode=self.kind, mode_label=MODE_LABELS[self.kind],
            model=self.policy_name, task=self.task, axis=self.args.axis if self.kind == "axes" else None,
            goal_tol_deg=float(np.degrees(self.args.goal_tolerance or 0.4)) if self.kind == "goals" else None,
            run=self.run_number, runs=len(self.plan["runs"]), run_labels=[label for label, _ in self.plan["runs"]],
            results=self.results, state=self.state, countdown=self.countdown, camera=bool(self.args.camera),
            start_temp_c=self.start_temp, joint_lo_deg=self.lo, joint_hi_deg=self.hi, q_deg=tolist(q),
            cmd_deg=tolist(cmd), current_ma=tolist(current), voltage_v=voltage, loop=loop, track_err_deg=track,
            params=self.params, keys=KEYS))

    def set_state(self, state, banner=None, color="yellow", countdown=None, note=None):
        """The dashboard's state and banner; the terminal gets a line when the state changes (note: its text)."""
        if state != self.state:
            style = dict(running="ok", cooling="warn", stopped="error", done="head", countdown="ok").get(state, "info")
            log(f"  [{state.upper()}] {note or banner or ''}".rstrip(), style)
        self.state, self.countdown = state, countdown
        phase_color = dict(running="green", cooling="orange", stopped="red", countdown="green").get(state, "yellow")
        self.show(state, phase_color, banner, color)

    def demo_keys(self, running=False):
        """The keys pressed since the last call: q, n, e (and p while running) raise; returns True for SPACE."""
        space = False
        names = {"q": "Q (quit)", "n": "N (next run)", "e": "E (emergency stop)", "p": "P (pause)"}  # SPACE: where
        # it does something (wait_start, hold_stopped)
        while self.keys:
            k = self.keys.popleft().lower()
            if k in names and (k != "p" or running):
                log(f"  key {names[k]}", "key")
            if k == "q":
                raise Quit
            if k == "n":
                raise Skip
            if k == "e":
                raise EmergencyStop
            if k == "p" and running:
                raise Pause
            if k in (" ", "\r", "\n"):
                space = True
        return space

    def check_controller(self):
        if self.hand.joints_age() > STALE_JOINTS_S:
            raise RuntimeError("the hand controller stopped sending joint positions")

    def wait_start(self, remaining, first):
        """Until the motors are cool enough (at the start of a run also below start_temp_c) and SPACE was pressed,
        then the countdown. E (torque off, then wait again), N and Q work here; P does nothing."""
        self.status["want_cube"] = True  # (the --mock camera puts its cube back)
        try:
            return self._wait_start(remaining, first)
        finally:
            self.status["want_cube"] = False

    def _wait_start(self, remaining, first):
        armed = False
        last_report = 0.0
        while True:
            self.tick()
            self.check_controller()
            try:
                space = self.demo_keys()
            except EmergencyStop:
                self.hold_stopped("emergency stop (E)")
                armed = False
                continue
            temps = self.hand.temperatures()
            cool = temps is not None and self.thermal.ready(temps, remaining)
            if cool and first and np.nanmax(temps) > self.start_temp:
                cool = False
            if not cool:
                armed = False
                if temps is None:
                    self.set_state("cooling", "Waiting for the motor temperatures", "orange")
                else:
                    hot = int(np.nanargmax(temps))
                    below = self.start_temp if first else float(np.min(self.thermal.start_below(remaining)))
                    what = "start" if first else "continue"
                    self.set_state("cooling", f"COOLING: {self.joint_ids[hot]} {temps[hot]:.0f} C, {what} below "
                                   f"{below:.0f} C", "orange", note=f"motors too warm to {what}; torque off, cooling")
                    if time.time() - last_report >= COOLING_REPORT_S:
                        last_report = time.time()
                        warm = ", ".join(f"{self.joint_ids[i]} {temps[i]:.0f}" for i in np.argsort(-temps)[:3])
                        log(f"    cooling: hottest {warm} C (mean {np.nanmean(temps):.1f} C), {what} below "
                            f"{below:.1f} C", "warn")
                continue
            if not armed:
                self.set_state("waiting", "Put the cube in the palm, then press SPACE", "yellow",
                               note=f"motors ok (hottest {np.nanmax(temps):.0f} C): put the cube in the palm, then "
                               "press SPACE in the dashboard")
                armed = space
                if not armed:
                    continue
                log("  key SPACE: countdown", "key")
                t0 = time.time()
            left = self.countdown_s - (time.time() - t0)
            if left <= 0:
                return
            self.set_state("countdown", f"{'START' if first else 'CONTINUE'} IN {np.ceil(left):.0f} - hands away",
                           "green", countdown=float(left), note=f"{self.countdown_s:.0f} s, hands away")

    def hold_stopped(self, why):
        """After E: torque off where the hand is, until SPACE (continue) or N / Q."""
        self.hand.stop_policy()
        try:
            self.hand.set_torque(False)
        except RuntimeError as e:
            log(f"  could not switch the torque off ({e}): SWITCH THE HAND'S POWER OFF", "error")
        log(f"  {why}: torque off", "error")
        self.set_state("stopped", "TORQUE OFF  -  SPACE: continue the run   N: next run   Q: quit", "red",
                       note="SPACE continues the run, N goes to the next one, Q quits")
        while not self.demo_keys():
            self.tick()
        log("  key SPACE: continue", "key")

    def demo_run(self, index, label, axis):
        from cube_spin_policy import CubeSpinPolicy
        a, hand = self.args, self.hand
        a.axis = axis
        speed = 0.0 if self.kind == "goals" else speed_for(self.plan, a.policies[0], axis)
        policy = CubeSpinPolicy(a.policies[0], self.joint_ids, goal_ang_vel=speed, use_cube_pose=None,
                                action_alpha=a.action_alpha, axis=axis, goal_tolerance=a.goal_tolerance)
        t_warm = time.perf_counter()
        warm_up(policy)
        warm_ms = (time.perf_counter() - t_warm) * 1000
        self.task = (f"goal orientations within {np.degrees(policy.goals.tolerance):.0f} deg" if policy.goals is not None
                     else f"spin {speed:+.1f} rad/s about {axis}")
        run_dir = self.session_dir / f"{index + 1}_{label.replace(' ', '_')}_{self.policy_name}"
        run_dir.mkdir(parents=True)
        run = Run(self.policy_name, a.policies[0], run_dir, a, self.joint_ids, policy, self.hand_T_camera)
        self.run_number = index + 1
        self.status.update(model=f"{self.policy_name}   {self.task}", duration=a.duration, goal=speed, home=None,
                           goals=None, trail_id=self.status["trail_id"] + 1, record=None, record_on=True,
                           running=False, elapsed=0.0, turns=0.0, speed=0.0, drops=0, goal_error_deg=None)
        print(flush=True)
        rule(f"RUN {index + 1}/{len(self.plan['runs'])} ({label})  {self.policy_name}: {self.task}, {a.duration:.0f} s")
        log(f"  policy: {policy.obs_dim} observations, {'with' if policy.use_cube_pose else 'without'} the cube pose"
            f"{', recurrent' if policy.recurrent else ''}, action alpha {a.action_alpha:g}, warm-up {WARMUP_STEPS} "
            f"steps in {warm_ms:.0f} ms", "dim")
        log(f"  files: {run_dir}", "dim")
        hand.goal_log = run.goal_times
        row = None
        try:
            log("  moving to the start pose, then torque off")
            self.park()
            hand.reset_health()
            self.wait_start(a.duration, first=True)
            home = self.pose[1] if self.pose is not None and self.pose_age() < 1.0 else np.zeros(3)
            run.begin(home, hand.temperatures(), speed)
            self.run = run
            temps = hand.temperatures()
            log(f"  start: motors mean {np.nanmean(temps):.1f} C, hottest {np.nanmax(temps):.1f} C"
                + (", cube seen in the palm" if self.pose is not None and self.pose_age() < 1.0 else ""), "ok")
            progress = Progress(self, run, a.duration, self.progress_every)
            # no home in the view: its circle is the drop distance, and drop detection is off
            self.status.update(home=None, record=str(run_dir / "video.mp4") if self.record_video else None)
            while run.elapsed() < a.duration:
                segment_temps = hand.temperatures()
                hand.start_policy(policy, run.elapsed, deadline=a.duration)
                run.resume()
                self.status.update(running=True, record_on=True)
                self.set_state("running")
                interruption = None
                try:
                    while run.elapsed() < a.duration:
                        self.tick()
                        self.check_controller()
                        run.log_health(hand, "running")
                        progress.maybe_report()
                        temps = hand.temperatures()
                        if temps is not None and self.thermal.over_limit(temps) is not None:
                            interruption = "heat"
                            break
                        self.demo_keys(running=True)  # no drop check: the policy runs on whatever happens
                except Pause:
                    interruption = "pause"
                except EmergencyStop:
                    interruption = "stop"
                hand.stop_policy()
                run.pause()
                self.status.update(running=False)
                run.segments.append((run.elapsed() - run.segment_start_elapsed, segment_temps, hand.temperatures()))
                if interruption is None:
                    break
                remaining = a.duration - run.elapsed()
                self.status.update(record_on=False)
                if interruption == "heat":
                    hot = self.thermal.over_limit(hand.temperatures())
                    log(f"  {run.elapsed():5.1f} s: {self.joint_ids[hot]} reached {hand.temperatures()[hot]:.0f} C "
                        f"(limit {a.temp_limit:.0f} C): cooling with the torque off", "warn")
                    hand.set_torque(False)
                    run.heat_pauses += 1
                elif interruption == "pause":
                    log(f"  {run.elapsed():5.1f} s: paused: hand to the start pose, torque off", "warn")
                    self.park()
                else:
                    self.hold_stopped(f"{run.elapsed():5.1f} s: emergency stop")
                log(f"  {remaining:.1f} s of the run left")
                self.wait_start(remaining, first=False)
                self.status.update(record_on=True)
            run.outcome = run.outcome or "completed"
            progress.report(final=True)
        except Skip:
            run.outcome = "skipped"
            log("  run skipped", "warn")
        except (Quit, KeyboardInterrupt):
            run.outcome = "aborted"
            raise
        except RuntimeError as e:
            run.outcome = f"error: {e}"
            raise
        finally:
            hand.stop_policy()
            self.status.update(running=False, banner=None)
            run.pause()
            self.run = None
            if run.started:
                log(f"  run over ({run.outcome}): parking the hand and writing the files")
                out = io.StringIO()
                with contextlib.redirect_stdout(out):  # the evaluation's lines (in proprio its turns mean nothing)
                    row = self.finish(run)
                for line in out.getvalue().splitlines():
                    if line.strip() and (self.args.camera or "could not" in line):
                        log(line.rstrip(), "dim")
            else:
                try:
                    run_dir.rmdir()
                except OSError:
                    pass
        if row is not None:
            self.rows[index] = row
            self.results[index] = headline(self.kind, row)
            r = self.results[index]
            log(f"  RESULT run {index + 1} ({label}): {r['headline']} ({r['detail']}) in {row['active_s']} s, motors "
                f"{row['temp_max_start_c']:.0f} -> {row['temp_max_end_c']:.0f} C, {row['heat_pauses']} heat pauses",
                "ok" if row["outcome"] == "completed" else "warn")

    def session(self):
        self.session_dir.mkdir(parents=True, exist_ok=True)
        for index, (label, axis) in enumerate(self.plan["runs"]):
            self.demo_run(index, label, axis)
        self.summary()
        self.set_state("done", "Demo finished  -  Q: quit", "white", note="demo finished: Q in the dashboard quits")
        while True:
            self.tick()
            self.demo_keys()

    def summary(self):
        if self.summarized:
            return
        self.summarized = True
        print(flush=True)
        rule(f"RESULTS  {MODE_LABELS[self.kind]}  {self.policy_name}")
        log(f"  {'run':10s} {'result':>14s}  {'detail':30s} {'outcome':12s} {'policy s':>8s}  {'motors C':>9s}", "dim")
        for (label, _), r, row in zip(self.plan["runs"], self.results, self.rows):
            if row is None:
                log(f"  {label:10s} {'-':>14s}  not run")
                continue
            log(f"  {label:10s} {r['headline']:>14s}  {r['detail']:30s} {row['outcome']:12s} {row['active_s']:>8}  "
                f"{row['temp_max_start_c']:>3.0f} -> {row['temp_max_end_c']:<3.0f}",
                "ok" if row["outcome"] == "completed" else "warn")
        (self.session_dir / "summary.json").write_text(json.dumps(dict(
            kind=self.kind, model=self.policy_name, config=self.plan, results=self.results,
            rows=[None if r is None else {k: str(v) for k, v in r.items()} for r in self.rows]), indent=1, default=str))
        log(f"  files in {self.session_dir}", "dim")


def headline(kind, row):
    """The dashboard's line for a finished run."""
    if kind == "goals":
        per = f"{row['s_per_goal']} s per goal" if row.get("s_per_goal") not in ("", None) else "no goal"
        return dict(headline=f"{row['goals']} goals", detail=per)
    if kind == "proprio":
        return dict(headline=f"{row['active_s']:.0f} s", detail=f"{row['outcome']}, peak {row['temp_peak_c']:.0f} C")
    return dict(headline=f"{row['turns']:+.1f} turns", detail=f"{row['mean_speed_rad_s']:+.2f} rad/s mean")


def logged(step):
    """step() with what eval_policies.py prints turned into log lines (its own "Starting ..." lines left out: the
    demo says what starts). The lines come when the step is done."""
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            return step()
    finally:
        for line in out.getvalue().splitlines():
            if line.strip() and not line.startswith("Starting"):
                log("  " + line.strip(), "dim")


def warm_up(policy):
    """A few steps of the network on the observation of the start pose, then a reset: the first real steps are as
    fast as the others (TorchScript optimises on the first calls)."""
    import torch
    with torch.no_grad():
        x = torch.zeros(1, policy.obs_dim)
        for _ in range(WARMUP_STEPS):
            policy.policy(x)
    policy.reset()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True, choices=("proprio", "extero", "optional"))
    p.add_argument("--task", choices=("axes", "goals"), help="With --mode optional: instead of optional.task.")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="The demo's YAML (default: %(default)s).")
    p.add_argument("--dry-run", action="store_true", help="Only the pre-flight check; nothing starts.")
    p.add_argument("--mock", action="store_true", help="Rehearsal without hardware: a fake hand and a drawn cube.")
    p.add_argument("--windowed", action="store_true", help="The dashboard in a window, not full screen.")
    p.add_argument("--progress-every", type=float, default=PROGRESS_EVERY_S, metavar="S", help="A progress line in "
                   "the terminal every S seconds of a run (default %(default)g; 0: none).")
    cli = p.parse_args(argv)
    plan = load_plan(cli.config, cli.mode, cli.task)
    args = eval_args(plan, cli)
    args.progress_every = cli.progress_every
    rule(f"ORCA HAND DEMO  {MODE_LABELS[plan['kind']]}")
    if preflight(plan, args):
        log("Pre-flight check failed: fix the config, nothing was started.", "error")
        return 1
    if cli.dry_run:
        log("Dry run: nothing was started.", "ok")
        return 0

    import torch
    threads = int(plan["common"].get("torch_threads", 1))
    torch.set_num_threads(threads)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, ev._terminate)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, ev._terminate)
    awake = keep_awake()
    log(f"Performance: torch on {threads} thread(s), RViz {'on' if args.rviz else 'off'}, W&B off, "
        f"{'caffeinate on (no sleep, no App Nap)' if awake is not None else 'no caffeinate (not macOS)'}", "dim")
    demo = Demo(args, plan)
    log(f"Results: {demo.session_dir}", "dim")
    code = 0
    t_start = time.time()
    try:
        log("Starting the window" + (" and the camera with the cube detector" if args.camera else " (no camera)")
            + " ...")
        logged(demo.start_vision)
        if args.twin:
            log("Starting the digital twin ...")
            logged(demo.start_twin)
        log(f"Starting the hand controller (servo acc {'from config.yaml' if args.servo_acc is None else args.servo_acc})"
            ": THE HAND MOVES to its neutral pose ...", "warn")
        logged(demo.start_hand)
        temps = demo.hand.temperatures()
        volts = demo.hand.voltage
        log(f"Hand ready: {len(demo.joint_ids)} joints, motors mean {np.nanmean(temps):.1f} C, hottest "
            f"{demo.joint_ids[int(np.nanargmax(temps))]} {np.nanmax(temps):.1f} C"
            + (f", supply {np.nanmin(volts):.1f} V" if volts is not None and np.any(np.isfinite(volts)) else ""),
            "ok")
        log(f"Keys in the dashboard: {KEYS}", "dim")
        demo.session()
    except Quit:
        log("Stopped (Q).", "warn")
        if any(demo.results):
            demo.summary()
    except KeyboardInterrupt:
        log("Stopped (Ctrl+C).", "warn")
    except RuntimeError as e:
        log(f"Error: {e}", "error")
        code = 1
    finally:
        log("Shutting down: torque off, closing the camera and the hand controller ...")
        demo.shutdown()
        if awake is not None:
            awake.terminate()
        log(f"Done after {(time.time() - t_start) / 60:.1f} min.", "ok" if code == 0 else "error")
    return code


if __name__ == "__main__":
    sys.exit(main())
