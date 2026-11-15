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
        import os
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

    def set_state(self, state, banner=None, color="yellow", countdown=None):
        self.state, self.countdown = state, countdown
        phase_color = dict(running="green", cooling="orange", stopped="red", countdown="green").get(state, "yellow")
        self.show(state, phase_color, banner, color)

    def demo_keys(self, running=False):
        """The keys pressed since the last call: q, n, e (and p while running) raise; returns True for SPACE."""
        space = False
        while self.keys:
            k = self.keys.popleft().lower()
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
                    self.set_state("cooling", f"COOLING: {self.joint_ids[hot]} {temps[hot]:.0f} C, "
                                   f"{'start' if first else 'continue'} below {below:.0f} C", "orange")
                continue
            if not armed:
                self.set_state("waiting", "Put the cube in the palm, then press SPACE", "yellow")
                armed = space
                if not armed:
                    continue
                t0 = time.time()
            left = self.countdown_s - (time.time() - t0)
            if left <= 0:
                return
            self.set_state("countdown", f"{'START' if first else 'CONTINUE'} IN {np.ceil(left):.0f} - hands away",
                           "green", countdown=float(left))

    def hold_stopped(self, why):
        """After E: torque off where the hand is, until SPACE (continue) or N / Q."""
        self.hand.stop_policy()
        try:
            self.hand.set_torque(False)
        except RuntimeError as e:
            print(f"  could not switch the torque off ({e}): switch the hand's power off")
        print(f"  {why}: torque off")
        self.set_state("stopped", "TORQUE OFF  -  SPACE: continue the run   N: next run   Q: quit", "red")
        while not self.demo_keys():
            self.tick()

    def demo_run(self, index, label, axis):
        from cube_spin_policy import CubeSpinPolicy
        a, hand = self.args, self.hand
        a.axis = axis
        speed = 0.0 if self.kind == "goals" else speed_for(self.plan, a.policies[0], axis)
        policy = CubeSpinPolicy(a.policies[0], self.joint_ids, goal_ang_vel=speed, use_cube_pose=None,
                                action_alpha=a.action_alpha, axis=axis, goal_tolerance=a.goal_tolerance)
        warm_up(policy)
        self.task = (f"goal orientations within {np.degrees(policy.goals.tolerance):.0f} deg" if policy.goals is not None
                     else f"spin {speed:+.1f} rad/s about {axis}")
        run_dir = self.session_dir / f"{index + 1}_{label.replace(' ', '_')}_{self.policy_name}"
        run_dir.mkdir(parents=True)
        run = Run(self.policy_name, a.policies[0], run_dir, a, self.joint_ids, policy, self.hand_T_camera)
        self.run_number = index + 1
        self.status.update(model=f"{self.policy_name}   {self.task}", duration=a.duration, goal=speed, home=None,
                           goals=None, trail_id=self.status["trail_id"] + 1, record=None, record_on=True,
                           running=False, elapsed=0.0, turns=0.0, speed=0.0, drops=0, goal_error_deg=None)
        print(f"\n=== run {index + 1}/{len(self.plan['runs'])} ({label}): {self.policy_name}, {self.task}, "
              f"{a.duration:.0f} s")
        hand.goal_log = run.goal_times
        row = None
        try:
            self.park()
            hand.reset_health()
            self.wait_start(a.duration, first=True)
            home = self.pose[1] if self.pose is not None and self.pose_age() < 1.0 else np.zeros(3)
            run.begin(home, hand.temperatures(), speed)
            self.run = run
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
                    print(f"  {run.elapsed():5.1f} s: {self.joint_ids[hot]} reached {hand.temperatures()[hot]:.0f} C, "
                          "cooling with the torque off")
                    hand.set_torque(False)
                    run.heat_pauses += 1
                elif interruption == "pause":
                    print(f"  {run.elapsed():5.1f} s: paused (P)")
                    self.park()
                else:
                    self.hold_stopped(f"{run.elapsed():5.1f} s: emergency stop (E)")
                self.wait_start(remaining, first=False)
                self.status.update(record_on=True)
            run.outcome = run.outcome or "completed"
        except Skip:
            run.outcome = "skipped"
            print("  next run (N)")
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
                row = self.finish(run)
            else:
                try:
                    run_dir.rmdir()
                except OSError:
                    pass
        if row is not None:
            self.rows[index] = row
            self.results[index] = headline(self.kind, row)

    def session(self):
        self.session_dir.mkdir(parents=True, exist_ok=True)
        for index, (label, axis) in enumerate(self.plan["runs"]):
            self.demo_run(index, label, axis)
        self.summary()
        self.set_state("done", "Demo finished  -  Q: quit", "white")
        while True:
            self.tick()
            self.demo_keys()

    def summary(self):
        if self.summarized:
            return
        self.summarized = True
        lines = [f"{label:10s} {r['headline'] if r else '-':>14s}  {r['detail'] if r else ''}"
                 for (label, _), r in zip(self.plan["runs"], self.results)]
        print(f"\nDemo {self.kind} with {self.policy_name}:\n  " + "\n  ".join(lines))
        (self.session_dir / "summary.json").write_text(json.dumps(dict(
            kind=self.kind, model=self.policy_name, config=self.plan, results=self.results,
            rows=[None if r is None else {k: str(v) for k, v in r.items()} for r in self.rows]), indent=1, default=str))
        print(f"  files in {self.session_dir}")


def headline(kind, row):
    """The dashboard's line for a finished run."""
    if kind == "goals":
        per = f"{row['s_per_goal']} s per goal" if row.get("s_per_goal") not in ("", None) else "no goal"
        return dict(headline=f"{row['goals']} goals", detail=per)
    if kind == "proprio":
        return dict(headline=f"{row['active_s']:.0f} s", detail=f"{row['outcome']}, peak {row['temp_peak_c']:.0f} C")
    return dict(headline=f"{row['turns']:+.1f} turns", detail=f"{row['mean_speed_rad_s']:+.2f} rad/s mean")


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
    cli = p.parse_args(argv)
    plan = load_plan(cli.config, cli.mode, cli.task)
    args = eval_args(plan, cli)
    if preflight(plan, args):
        return 1
    if cli.dry_run:
        return 0

    import torch
    torch.set_num_threads(int(plan["common"].get("torch_threads", 1)))
    Path(args.out).mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, ev._terminate)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, ev._terminate)
    awake = keep_awake()
    demo = Demo(args, plan)
    code = 0
    try:
        demo.start_vision()
        if args.twin:
            demo.start_twin()
        demo.start_hand()
        demo.session()
    except Quit:
        print("\nStopped (Q).")
        if any(demo.results):
            demo.summary()
    except KeyboardInterrupt:
        print("\nStopped (Ctrl+C).")
    except RuntimeError as e:
        print(f"\nError: {e}")
        code = 1
    finally:
        demo.shutdown()
        if awake is not None:
            awake.terminate()
    return code


if __name__ == "__main__":
    sys.exit(main())
