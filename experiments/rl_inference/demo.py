#!/usr/bin/env python3
"""Demo mode of the ORCA v2 hand: a policy spins the cube in front of an audience, with a live dashboard.

    python experiments/rl_inference/demo.py --mode proprio                  # the policy of the 74 inputs (no cube pose)
    python experiments/rl_inference/demo.py --mode extero                   # camera plugged in: the cube pose policy
    python experiments/rl_inference/demo.py --mode optional                 # optional.task in the config (axes)
    python experiments/rl_inference/demo.py --mode optional --task goals    # random goal orientations
    python experiments/rl_inference/demo.py --mode extero --dry-run         # only the pre-flight check
    python experiments/rl_inference/demo.py --mode extero --mock            # rehearsal: fake hand, drawn cube

Everything a demo uses (policy file, servo acceleration, action alpha, speed, temperatures, camera) is in one YAML
file, experiments/cfgs/demo.yaml (--config another one).

A program of its own (it does not use eval_policies.py): it starts the hand controller, the dashboard's process (with
the camera and the cube detector when there is a camera) and the digital twin's renderer, and runs the policy in a
thread of its own at 20 Hz. It shares only the building blocks in cube_eval/ (camera and dashboard process, twin
renderer, thermal model, turn counter) and cube_spin_policy.py with the evaluation.

- every demo is `repeats` runs (3) of `duration_s` (60 s of policy time) with the same policy; optional/axes runs once
  about each axis of its list (x, y, z), optional/goals `repeats` times with new random goals each time,
- proprio: `proprio.use_camera` false (no camera allowed): the dashboard shows the perceived state (the hand in its
  measured pose) and the joints; true: the camera counts the turns as in extero, without the perceived state,
- no drop detection: the policy keeps running when the cube falls (P parks the hand so the cube can be put back),
- a run starts only when every motor is at most `start_temp_c` warm and the operator pressed SPACE (cube in the palm),
- results in common.out (default logs/demo): results.csv, and a folder per session and run (joints.csv, steps.csv
  with every policy step, health.csv, trace.csv with the cube detections, video.mp4 with a camera, meta.json).

Keys in the dashboard: SPACE = start / continue, P = pause (hand to the start pose, torque off), E = emergency stop
(torque off where the hand is), N = next run, Q = quit, F = full screen on/off. Ctrl+C in the terminal stops
everything and leaves the hand with the torque off.

The terminal says when the policy loop misses its 20 Hz (a step later than 150 ms after the one before) and whether the
hand controller's joint positions were late at that moment, so a hitch on the hand can be told apart.
"""

import argparse
import csv
import json
import multiprocessing
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from cube_eval import twin_render, vision  # noqa: E402
from cube_eval.thermal import TemperatureFilter, ThermalModel  # noqa: E402
from cube_eval.turns import TurnCounter  # noqa: E402

DEFAULT_CONFIG = REPO / "experiments/cfgs/demo.yaml"
sys.path.insert(0, str(REPO / "src/hand_control/orca"))
import hands  # noqa: E402

# The hand (--hand): its config.yaml in the source tree and the hand controller's hand_cfg_name
HAND_CONFIG = hands.config_path(hands.DEFAULT_HAND)
HAND_CFG_NAME = hands.cfg_name(hands.DEFAULT_HAND)
HAND_NODE = REPO / "src/hand_control/hand_control_node.py"
MOCK_HAND_NODE = HERE / "cube_eval/mock_hand_node.py"
MODE_LABELS = dict(proprio="PROPRIOCEPTIVE", extero="EXTEROCEPTIVE (camera)", axes="OPTIONAL - ALL AXES",
                   goals="OPTIONAL - GOAL ORIENTATIONS")
KEYS = "SPACE start   P pause   E stop   N next run   Q quit   F fullscreen"
TWIN_TITLE = "perceived state"
POLICY_HZ = 20.0
STALL_S = 0.15            # a policy step later than this after the one before: reported in the terminal
STALE_JOINTS_S = 2.0      # no joint positions for this long: the hand controller is gone
DEFAULT_SPEED = 2.0       # rad/s, when neither the config nor the policy's training info has one
WARMUP_STEPS = 50
PROGRESS_EVERY_S = 10.0   # a progress line this often (policy time) while a run runs
COOLING_REPORT_S = 15.0   # a line about the hottest motors this often while cooling
COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
STYLES = dict(info="", ok="32", warn="33", error="31;1", head="36;1", dim="2", key="35")


def log(message, style="info"):
    """One line in the terminal: the time, then the message (coloured in a terminal, unless NO_COLOR is set)."""
    code = STYLES.get(style, "")
    text = f"\033[{code}m{message}\033[0m" if COLOR and code else message
    print(f"{datetime.now():%H:%M:%S}  {text}", flush=True)


def rule(title, style="head"):
    log(f"==== {title} {'=' * max(4, 100 - len(title))}", style)


class Quit(Exception):
    """Q was pressed: stop after cleaning up."""


class Skip(Exception):
    """N was pressed: the next run."""


class EmergencyStop(Exception):
    """E was pressed: the torque goes off where the hand is."""


class Pause(Exception):
    """P was pressed: the hand goes to the start pose, so the cube can be put back."""


def _terminate(*_):
    raise KeyboardInterrupt


# ---------------------------------------------------------------------------------------------------- configuration
def repo_path(value):
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else REPO / path


def load_plan(config_path, mode, task=None):
    """The demo to run, from the YAML: dict(kind = proprio|extero|axes|goals, common, section (that kind's settings,
    with the camera's), camera (bool), runs = [(label, axis)])."""
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
    camera = bool(section.get("use_camera", False)) if kind == "proprio" else True
    if camera:
        section = {**(config.get("camera") or {}), **section}
    repeats = int(common.get("repeats", 3))
    if kind == "axes":
        axes = list(section.get("axes") or ["x", "y", "z"])
        if any(a not in ("x", "y", "z") for a in axes):
            raise SystemExit(f"optional.axes.axes: x, y or z, not {axes}")
        runs = [(f"about {a}", a) for a in axes]
    else:
        runs = [(f"run {i + 1}", "z") for i in range(repeats)]
    return dict(kind=kind, common=common, section=section, camera=camera, runs=runs, config_path=str(config_path))


def settings(plan, cli):
    """Everything the demo needs from the plan and the command line, as attributes."""
    from cube_spin_policy import CUBE_POSE_TIMEOUT_S
    c, s, kind = plan["common"], plan["section"], plan["kind"]
    a = argparse.Namespace()
    a.kind, a.camera, a.mock = kind, plan["camera"], cli.mock
    a.policy = repo_path(s["model"])
    a.duration = float(c.get("duration_s", 60))
    a.out = repo_path(c.get("out", "logs/demo"))
    a.temp_limit, a.temp_margin = float(c.get("temp_limit_c", 65.0)), float(c.get("temp_margin_c", 2.0))
    a.start_temp, a.countdown_s = float(c.get("start_temp_c", 38.0)), float(c.get("countdown_s", 3))
    a.start_wrist_deg = c.get("start_wrist_deg")
    a.cube_edge = float(c.get("cube_edge", 0.045))
    a.action_alpha = float(s.get("action_alpha", 0.2))
    a.servo_acc = None if s.get("servo_acc") is None else int(s["servo_acc"])
    current = s.get("max_current", c.get("max_current"))
    a.max_current = None if current is None or str(current).strip().lower() == "default" else int(current)
    if a.max_current is not None and not 0 < a.max_current <= 5000:
        raise SystemExit(f"max_current: a number of mA from 1 to 5000 (or null: config.yaml's), not {current!r}")
    a.speed = per_axis(s.get("speed"), "speed", float)
    a.direction = per_axis(s.get("direction"), "direction", int)
    if a.direction is not None and any(v not in (1, -1, None) for v in a.direction.values()):
        raise SystemExit(f"direction: +1 or -1 (or one per axis, e.g. {{x: 1, y: -1, z: 1}}), not {s['direction']!r}")
    a.goal_tolerance = float(s.get("goal_tolerance", 0.4)) if kind == "goals" else None
    # goals: uniform (random orientations) or face90 (each goal 90 deg from the last); null: the model's .json, else uniform
    a.goal_mode = s.get("goal_mode") if kind == "goals" else None
    # goals: seconds the hand holds still (the last joint targets) when the cube reached a goal, before the next goal
    a.goal_hold_s = max(0.0, float(s.get("goal_hold_s", 1.0) or 0.0)) if kind == "goals" else 0.0
    # the perceived state: where there is no camera view to look at, and where the cube pose matters (extero, goals)
    a.twin = bool(c.get("twin", True)) and not (kind == "proprio" and a.camera)
    a.fullscreen = bool(c.get("fullscreen", True)) and not cli.windowed
    a.record_video = bool(c.get("record_video", True)) and a.camera
    a.torch_threads = int(c.get("torch_threads", 1))
    a.progress_every = cli.progress_every
    # the camera's pose in the hand: for the cube pose (extero, goals) and the axis of the turns (axes); proprio
    # counts the turns about the camera's viewing axis and needs none
    a.calibration = repo_path(s["calibration"]) if kind != "proprio" and s.get("calibration") and not cli.mock else None
    a.hand_to_tag = [float(v) for v in s["hand_to_tag"]] if s.get("hand_to_tag") is not None else None
    a.exposure_us, a.iso = int(s.get("exposure_us", 5000)), int(s.get("iso", 400))
    a.camera_id, a.device = str(s.get("camera_id", "")), str(s.get("device", "auto"))
    a.cube_pose_timeout = CUBE_POSE_TIMEOUT_S
    return a


def training_info(policy_path):
    """The training info download_policies.py saved in the policy's .json ({} if there is none)."""
    try:
        return json.loads(Path(policy_path).with_suffix(".json").read_text()).get("training") or {}
    except (OSError, ValueError):
        return {}


def per_axis(value, name, kind):
    """A config value for every axis: None, one value for all axes, or a dict per axis ({x: .., y: .., z: ..}).
    Returns None or a dict axis -> value (an axis missing from the dict: None)."""
    if value is None:
        return None
    try:
        if isinstance(value, dict):
            if set(value) - {"x", "y", "z"}:
                raise ValueError
            return {axis: None if value.get(axis) is None else kind(value[axis]) for axis in "xyz"}
        return {axis: kind(value) for axis in "xyz"}
    except (TypeError, ValueError):
        raise SystemExit(f"{name}: one number, or one per axis ({{x: .., y: .., z: ..}}), not {value!r}") from None


def speed_for(a, axis):
    """The commanded speed of a run (rad/s): the config's (speed, one or per axis), else the top of the training range
    about the axis; direction (+1 / -1, one or per axis), where given, sets the sign."""
    speed = None if a.speed is None else a.speed[axis]
    if speed is None:
        ranges = ((training_info(a.policy).get("commands") or {}).get("continuous_rotation") or {}).get(
            "angular_vel_range") or [None] * 3
        top = ranges["xyz".index(axis)]
        speed = float(top) if top else DEFAULT_SPEED
    direction = None if a.direction is None else a.direction[axis]
    return abs(speed) * direction if direction is not None else speed


def policy_problems(a):
    """Why the policy file cannot run in this demo (empty: it can)."""
    import torch
    from cube_spin_policy import LAYOUTS, NUM_JOINTS, input_size, is_recurrent
    try:
        net = torch.jit.load(str(a.policy), map_location="cpu").eval()
        size = input_size(net)
        if size not in LAYOUTS:
            return [f"it takes {size or 'other'} observations, none of the layouts of cube_spin_policy.py"]
        with torch.no_grad():
            out = net(torch.zeros(1, size))
    except Exception as e:
        return [f"it cannot be loaded ({type(e).__name__}: {e})"]
    if out.shape[-1] != NUM_JOINTS:
        return [f"it outputs {out.shape[-1]} actions, not {NUM_JOINTS}"]
    terms, problems = LAYOUTS[size].terms, []
    if is_recurrent(net) or LAYOUTS[size].history:
        problems.append("a policy with history or memory (a student): not a demo policy")
    if a.kind == "proprio" and "cube_pose" in terms:
        problems.append("the proprio demo gives the policy no cube pose, but it takes one: use --mode extero")
    if (a.kind == "goals") != ("goal_orientation" in terms):
        problems.append("optional.goals needs a policy of the goal task" if a.kind == "goals"
                        else "a policy of the goal task: use --mode optional --task goals")
    info = training_info(a.policy)
    alpha = info.get("action_alpha") or 0.2
    if abs(alpha - a.action_alpha) > 1e-9:
        problems.append(f"trained with action alpha {alpha:g}, the config says {a.action_alpha:g}")
    if a.kind == "axes":
        probs = ((info.get("commands") or {}).get("continuous_rotation") or {}).get("axis_probs") or [0, 0, 1]
        if not any(probs[:2]):
            problems.append("optional.axes needs a policy trained about x, y and z (all-axes task)")
    return problems


def preflight(a, plan):
    """Everything that can be checked before the hand moves. Returns the problems (empty: go)."""
    print(f"Pre-flight check: {MODE_LABELS[a.kind]}, {plan['config_path']}")
    print(f"  policy        {a.policy}")
    problems = [] if a.policy.exists() else [f"no policy file {a.policy}"]
    if not problems:
        problems += [f"the policy: {p}" for p in policy_problems(a)]
    if a.kind != "proprio" and a.calibration is None and not a.mock:
        problems.append("no camera calibration (camera.calibration in the config): run apriltag_calibration.py")
    if a.calibration is not None and not a.calibration.exists():
        problems.append(f"no camera calibration file {a.calibration}: run apriltag_calibration.py")
    if a.calibration is not None and a.hand_to_tag is None:
        problems.append("camera.hand_to_tag is missing")
    speeds = {axis: speed_for(a, axis) for _, axis in plan["runs"]} if a.kind != "goals" and a.policy.exists() else {}
    print(f"  runs          {len(plan['runs'])} x {a.duration:.0f} s: " + ", ".join(
        label + (f" ({speeds[axis]:+.1f} rad/s)" if speeds else "") for label, axis in plan["runs"]))
    acc = "config.yaml's" if a.servo_acc is None else a.servo_acc
    current = "config.yaml's" if a.max_current is None else f"{a.max_current} mA"
    print(f"  servo acc     {acc}    max current {current}    action alpha {a.action_alpha:g}    start below "
          f"{a.start_temp:.1f} C    limit {a.temp_limit:.0f} C")
    print(f"  camera        {'none' if not a.camera else 'mock' if a.mock else 'OAK'}"
          + (f", calibration {a.calibration}" if a.calibration else "")
          + f"    perceived state {'on' if a.twin else 'off'}")
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


def transform(position, rotation):
    """4x4 from a position and a scipy Rotation."""
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = rotation.as_matrix(), position
    return T


def camera_in_hand(a):
    """hand_T_camera (4x4), or None (no calibration). --mock: a camera that looks down at the palm and sees the drawn
    cube resting where the task commands it, (0, 0, -0.05)."""
    from cube_spin_policy import OBS_T_HAND, hand_T_camera
    if a.calibration is not None:
        return hand_T_camera(np.load(a.calibration), a.hand_to_tag)
    if a.mock and a.kind != "proprio":
        obs_T_camera = np.diag([1.0, -1.0, -1.0, 1.0])
        obs_T_camera[:3, 3] = np.array([0.0, 0.0, -0.05]) - obs_T_camera[:3, :3] @ vision.MockCamera.HOME
        return np.linalg.inv(OBS_T_HAND) @ obs_T_camera
    return None


# ---------------------------------------------------------------------------------------------------- the hand
class Hand:
    """The hand controller's topics and torque service (one ROS executor thread), and the policy loop: a thread of its
    own that steps the policy every 50 ms on its own clock, while it runs."""

    def __init__(self, joint_ids, cube_pose_timeout):
        from cube_spin_policy import CubePose
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from std_msgs.msg import Float32MultiArray
        from std_srvs.srv import SetBool

        self._msg, self._SetBool = Float32MultiArray, SetBool
        self.joint_ids = list(joint_ids)
        self.lock = threading.Lock()
        self.q = self.qd = self.current = self.voltage = None
        self.q_time = 0.0
        self.temps, self.temps_time = None, 0.0
        self.temp_filter = TemperatureFilter()
        self.policy_on = False
        self.last_command = None
        self.clock = None          # the run's policy time, for the logs
        self.joint_log = []        # (policy time, command, measured, current) while the policy runs
        self.step_log = []         # (policy time, wall interval since the step before (ms), step computation (ms))
        self.stalls = []           # (policy time, interval (ms), age of the joint positions (ms)) of late steps
        self.goal_log = []         # policy time of every goal orientation reached (optional/goals)
        self.skipped = 0           # steps without a finite joint state (nothing sent)
        self.cube = CubePose(cube_pose_timeout)
        self.step_times = deque(maxlen=2 * int(POLICY_HZ))
        self.infer_ms = deque(maxlen=2 * int(POLICY_HZ))
        self.torque_on = None
        self._thread, self._stop = None, threading.Event()

        self.node = Node("orca_demo")
        sub = lambda name, cb: self.node.create_subscription(Float32MultiArray, f"/right/hand/{name}", cb, 20)  # noqa: E731
        sub("joint_position_from_motor", self._position)
        sub("joint_velocity_from_motor", self._velocity)
        sub("motor_current", self._current)
        sub("motor_temperature", self._temperature)
        sub("motor_voltage", self._voltage)
        self.command_pub = self.node.create_publisher(Float32MultiArray, "/right/hand/policy_output", 10)
        self.torque_client = self.node.create_client(SetBool, "/right/hand/set_torque_enabled")
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        threading.Thread(target=self.executor.spin, daemon=True, name="ros").start()

    # -- ROS callbacks (the executor thread)
    def _position(self, msg):
        q = np.array(msg.data, dtype=np.float64)
        with self.lock:
            self.q, self.q_time = q, time.time()
            if self.policy_on and self.last_command is not None and self.clock is not None:
                self.joint_log.append((self.clock(), self.last_command, q, self.current))

    def _velocity(self, msg):
        qd = np.array(msg.data, dtype=np.float64)
        with self.lock:
            self.qd = qd

    def _current(self, msg):
        self.current = np.array(msg.data, dtype=np.float64)

    def _temperature(self, msg):
        with self.lock:
            self.temps, self.temps_time = self.temp_filter.update(msg.data).copy(), time.time()

    def _voltage(self, msg):
        self.voltage = np.array(msg.data, dtype=np.float64)

    # -- the policy loop (its own thread)
    def start_policy(self, policy, clock, deadline, goal_hold_s=0.0, on_hold=None):
        """Torque on at the present pose, then the policy takes over, starting as after a reset in the task, until
        stop_policy() or the policy time clock() reaches deadline. goal_hold_s (goal task): when the cube reached a
        goal, the policy pauses for this long (wall time) and the hand holds its last joint targets; then the next
        goal. on_hold(True / False): called when a hold starts and ends (Run.set_hold: the run's timer stops)."""
        self.goal_hold_s = float(goal_hold_s or 0.0)
        self.on_hold = on_hold or (lambda on: None)
        self.holding = False  # True while the hand holds still at a reached goal (the dashboard's banner)
        self.stop_policy()
        self.set_torque(True)
        with self.lock:
            policy.reset()
            self.cube.restart()  # the tracker's filters start again, as at a reset in the task
            self.clock, self.last_command, self.policy_on = clock, None, True
            self.step_times.clear()
            self.infer_ms.clear()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, args=(policy, clock, deadline, self._stop), daemon=True,
                                        name="policy")
        self._thread.start()

    def stop_policy(self):
        """The policy loop ends; when this returns, it sends nothing more."""
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        with self.lock:
            self.policy_on = False
        self.holding = False

    def _loop(self, policy, clock, deadline, stop):
        period = 1.0 / POLICY_HZ
        next_step = time.perf_counter()
        last = None
        hold_until = None  # wall time (perf_counter) until which the hand holds still at a reached goal; the run's
        # policy time stands still meanwhile (on_hold), so the hold does not shorten the run
        try:
            self._steps(policy, clock, deadline, stop, period, next_step, last, hold_until)
        finally:
            if self.holding:  # stopped during a hold (P, E, Q): the run's timer goes on counting from here
                self.holding = False
                self.on_hold(False)

    def _steps(self, policy, clock, deadline, stop, period, next_step, last, hold_until):
        while not stop.is_set():
            if clock() >= deadline:
                break  # the run's time is up, even if the main loop is late
            if hold_until is not None:
                if time.perf_counter() < hold_until:
                    # holding at the goal: no policy step, the last joint targets again (the hand stays where it is)
                    t = time.perf_counter()
                    with self.lock:
                        command = self.last_command
                    if command is not None:
                        self._publish(command)
                    last = t  # a held step is no late step
                    next_step += period
                    delay = next_step - time.perf_counter()
                    if delay < -period:
                        next_step = time.perf_counter()
                    elif delay > 0:
                        stop.wait(delay)
                    continue
                hold_until = None
                self.holding = False
                self.on_hold(False)  # the run's timer goes on
                policy.goals.advance()  # the next goal, and the policy goes on
            with self.lock:
                q, qd, q_time = self.q, self.qd, self.q_time
                cube = self.cube.observation(time.time()) if policy.use_cube_pose else None
            t = time.perf_counter()
            goals = policy.goals
            reached = goals.reached if goals is not None else 0
            out = None if q is None or qd is None else policy.step(q, qd, cube)
            compute_ms = (time.perf_counter() - t) * 1000.0
            if out is None:
                self.skipped += 1
            else:
                command = np.asarray(out[0], dtype=np.float64)
                self._publish(command)
                now = clock()
                interval = 0.0 if last is None else (t - last) * 1000.0
                with self.lock:
                    self.last_command = command
                    self.step_times.append(t)
                    self.infer_ms.append(compute_ms)
                    self.step_log.append((now, interval, compute_ms))
                    if last is not None and t - last > STALL_S:
                        self.stalls.append((now, interval, (time.time() - q_time) * 1000.0))
                    if goals is not None and goals.reached > reached:
                        self.goal_log.append(now)
                if goals is not None and goals.reached > reached and goals.waiting:
                    hold_until = time.perf_counter() + self.goal_hold_s
                    self.holding = True
                    self.on_hold(True)  # the run's timer stops until the hold ends
                last = t
            next_step += period
            delay = next_step - time.perf_counter()
            if delay < -period:  # far behind: go on from now, no burst of steps to catch up
                next_step = time.perf_counter()
            elif delay > 0:
                stop.wait(delay)

    def _publish(self, joint_pos_rad):
        self.command_pub.publish(self._msg(data=np.asarray(joint_pos_rad, dtype=np.float32).tolist()))

    # -- from the main thread
    def joints_age(self):
        return time.time() - self.q_time if self.q_time else float("inf")

    def measured(self):
        with self.lock:
            return None if self.q is None else self.q.copy()

    def loop_stats(self):
        """The policy loop of the last 2 s: dict(hz, jitter_ms, infer_ms), or None while it does not run."""
        with self.lock:
            times, infer = np.array(self.step_times), np.array(self.infer_ms)
        if len(times) < 3 or time.perf_counter() - times[-1] > 0.5:
            return None
        dt = np.diff(times)
        return dict(hz=float(1.0 / np.mean(dt)), jitter_ms=float(np.std(dt) * 1000.0), infer_ms=float(np.mean(infer)))

    def take_stalls(self):
        with self.lock:
            stalls, self.stalls = self.stalls, []
        return stalls

    def temperatures(self, max_age=5.0):
        """Filtered motor temperatures in joint order, or None without a recent reading."""
        with self.lock:
            if self.temps is None or time.time() - self.temps_time > max_age:
                return None
            return self.temps.copy()

    def cube_detection(self, stamp, hand_T_cube):
        with self.lock:
            self.cube.update(stamp, hand_T_cube)

    def cube_state(self, now):
        """(position, quaternion w x y z, age in s) of the cube as the policy sees it, or None before the first."""
        with self.lock:
            if self.cube.pos is None:
                return None
            return self.cube.pos.copy(), self.cube.quat.copy(), self.cube.age(now)

    def take_logs(self):
        """The joint and step logs since the last call."""
        with self.lock:
            joints, steps = self.joint_log, self.step_log
            self.joint_log, self.step_log = [], []
        return joints, steps

    def set_torque(self, enabled, timeout=5.0):
        if enabled:  # make the measured pose the goal first, so the hand does not jump to an old goal
            q = self.measured()
            if q is not None and np.all(np.isfinite(q)):
                for _ in range(3):
                    self._publish(q)
                    time.sleep(0.05)
        if not self.torque_client.wait_for_service(timeout_sec=timeout):
            raise RuntimeError("no torque service: is the hand controller running?")
        future = self.torque_client.call_async(self._SetBool.Request(data=bool(enabled)))
        t0 = time.time()
        while not future.done():
            if time.time() - t0 > timeout:
                raise RuntimeError("the hand controller did not answer the torque request")
            time.sleep(0.02)
        if not future.result().success:
            raise RuntimeError(f"could not switch the torque: {future.result().message}")
        self.torque_on = bool(enabled)

    def move_to(self, pose_deg, seconds=2.0, tick=None):
        """Torque on and a ramp from the measured pose to pose_deg (joint order, degrees)."""
        self.set_torque(True)
        target = np.deg2rad(np.asarray(pose_deg, dtype=float))
        q = self.measured()
        start = q if q is not None and np.all(np.isfinite(q)) else target
        steps = max(1, int(seconds * POLICY_HZ))
        for i in range(1, steps + 1):
            self._publish(start + (target - start) * i / steps)
            time.sleep(1.0 / POLICY_HZ)
            if tick is not None:
                tick(sleep=0.0)
        time.sleep(0.5)

    def close(self):
        self.stop_policy()
        self.executor.shutdown(timeout_sec=1.0)
        self.node.destroy_node()


# ---------------------------------------------------------------------------------------------------- one run
class Run:
    """What one run measures and writes."""

    def __init__(self, label, axis, speed, run_dir, policy, up):
        self.label, self.axis, self.speed, self.run_dir = label, axis, speed, run_dir
        self.goals = policy.goals
        self.sign = 1.0 if speed >= 0 else -1.0
        self.counter = TurnCounter(up=up)
        self.outcome = ""
        self.started = self.active = False
        self.t0 = None
        self.accumulated, self.segment_start, self.segment_start_elapsed = 0.0, None, 0.0
        # goal task: the timer stops while the hand holds still at a reached goal (goal_hold_s): seconds held in the
        # running stretch, and the wall time the hold going on began (None: no hold)
        self.held, self.hold_since = 0.0, None
        self.segments = []         # (policy seconds, temperatures at start, at end) per uninterrupted stretch
        self.heat_pauses = self.pauses = self.stops = 0
        self.trace, self.detection_elapsed = [], []
        self.health, self.last_health = [], 0.0
        self.joints, self.steps, self.stalls = [], [], []
        self.start_temps = None

    def begin(self, temps):
        self.started, self.t0, self.start_temps = True, time.time(), temps

    def _held(self, t):
        """Seconds held at goals in the running stretch up to wall time t."""
        since = self.hold_since
        return self.held + (max(0.0, t - since) if since is not None else 0.0)

    def elapsed_at(self, t):
        """Policy time at wall time t (the running stretch; holds at goals do not count)."""
        return self.accumulated + t - self.segment_start - self._held(t)

    def elapsed(self):
        """Policy time: seconds the policy has run (holds at goals do not count)."""
        if self.active:
            return self.elapsed_at(time.time())
        return self.accumulated

    def set_hold(self, on):
        """From the policy loop: the hand starts (True) or ends (False) holding still at a reached goal."""
        now = time.time()
        if on and self.hold_since is None:
            self.hold_since = now
        elif not on and self.hold_since is not None:
            self.held += now - self.hold_since
            self.hold_since = None

    def resume(self):
        self.segment_start, self.segment_start_elapsed = time.time(), self.accumulated
        self.held, self.hold_since = 0.0, None
        self.active = True
        self.counter.rebase()

    def pause(self):
        if self.active:
            self.accumulated = self.elapsed_at(time.time())
            self.held, self.hold_since = 0.0, None
            self.active = False

    def detection(self, stamp, position, rotation, pixel):
        if not self.started:
            return
        elapsed = ""
        if self.active and stamp >= self.segment_start:
            elapsed = self.elapsed_at(stamp)
            self.counter.add(elapsed, rotation)
            self.detection_elapsed.append(elapsed)
            elapsed = round(elapsed, 3)
        self.trace.append([round(stamp - self.t0, 3), elapsed, *np.round(pixel, 1), *np.round(position, 4),
                           *np.round(rotation.as_quat(), 4)])

    def figures(self, goal_times):
        """For the dashboard."""
        if not self.started:
            return {}
        elapsed = self.elapsed()
        out = dict(elapsed=elapsed, turns=self.sign * self.counter.turns, drops=0,
                   speed=self.sign * self.counter.speed(elapsed, 5.0))
        if self.goals is not None:
            error = self.goals.error
            out.update(goals=len(goal_times), goal_error_deg=None if error is None else float(np.degrees(error)))
        return out

    def log_health(self, hand, state):
        now = time.time()
        if now - self.last_health < 1.0:
            return
        self.last_health = now
        temps, volts = hand.temperatures(), hand.voltage
        self.health.append((round(now - self.t0, 1), round(self.elapsed(), 2), state,
                            None if temps is None else temps.tolist(), None if volts is None else volts.tolist()))

    def collect(self, hand):
        joints, steps = hand.take_logs()
        self.joints += joints
        self.steps += steps

    def write(self, a, joint_ids, end_temps, goal_times, skipped, meta):
        """The run's files; returns its row of results.csv."""
        d, n = self.run_dir, len(joint_ids)
        nan = np.full(n, np.nan)
        start = nan if self.start_temps is None else np.asarray(self.start_temps, float)
        end = nan if end_temps is None else np.asarray(end_temps, float)
        peak = np.nanmax([np.asarray(t, float) for *_, t, _ in self.health if t is not None] + [start, end], axis=0)
        active = self.accumulated
        with open(d / "joints.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["policy_time_s", *[f"cmd_deg_{j}" for j in joint_ids], *[f"meas_deg_{j}" for j in joint_ids],
                        *[f"current_ma_{j}" for j in joint_ids]])
            for t, cmd, q, cur in self.joints:
                w.writerow([f"{t:.3f}", *np.round(np.degrees(cmd), 2), *np.round(np.degrees(q), 2),
                            *(np.round(cur, 1) if cur is not None and len(cur) == n else [""] * n)])
        with open(d / "steps.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["policy_time_s", "interval_ms", "compute_ms"])
            w.writerows([[f"{t:.3f}", f"{i:.1f}", f"{c:.2f}"] for t, i, c in self.steps])
        with open(d / "health.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["wall_s", "policy_time_s", "state", *[f"temp_c_{j}" for j in joint_ids], "voltage_min_v"])
            for wall, el, state, temps, volts in self.health:
                w.writerow([wall, el, state, *(np.round(temps, 1) if temps is not None else [""] * n),
                            "" if not volts or not np.any(np.isfinite(volts)) else round(float(np.nanmin(volts)), 2)])
        if self.trace:
            with open(d / "trace.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["wall_s", "policy_time_s", "px", "py", "x_m", "y_m", "z_m", "qx", "qy", "qz", "qw"])
                w.writerows(self.trace)
        if goal_times:
            (d / "goals.csv").write_text("policy_time_s\n" + "".join(f"{t:.2f}\n" for t in goal_times))
        intervals = np.array([i for _, i, _ in self.steps if i > 0])
        rate = len(self.steps) / active if active > 0 else 0.0
        turns = self.sign * self.counter.turns if a.camera else None
        row = dict(
            time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), mode=a.kind, camera=a.camera, run=self.label,
            model=a.policy.stem.removesuffix("_policy"), axis=self.axis if a.kind == "axes" else "",
            speed_rad_s=round(self.speed, 2) if a.kind != "goals" else "", outcome=self.outcome,
            active_s=round(active, 1), turns="" if turns is None else round(turns, 2),
            mean_speed_rad_s="" if turns is None or active <= 0 else round(turns * 2 * np.pi / active, 3),
            goals=len(goal_times) if self.goals is not None else "",
            s_per_goal=round(active / len(goal_times), 1) if self.goals is not None and goal_times else "",
            heat_pauses=self.heat_pauses, pauses=self.pauses, stops=self.stops,
            temp_max_start_c=round(float(np.nanmax(start)), 1) if np.any(np.isfinite(start)) else np.nan,
            temp_max_end_c=round(float(np.nanmax(end)), 1) if np.any(np.isfinite(end)) else np.nan,
            temp_peak_c=round(float(np.nanmax(peak)), 1) if np.any(np.isfinite(peak)) else np.nan,
            loop_hz=round(rate, 2), max_step_gap_ms=round(float(intervals.max()), 1) if len(intervals) else "",
            late_steps=int(np.sum(intervals > STALL_S * 1000)), skipped_steps=skipped,
            servo_acc="config" if a.servo_acc is None else a.servo_acc,
            max_current_ma="config" if a.max_current is None else a.max_current, action_alpha=a.action_alpha,
            run_dir=str(d))
        (d / "meta.json").write_text(json.dumps(dict(row, **meta, stalls=self.stalls), indent=1, default=str))
        return row


def append_row(path, row):
    """A row of results.csv (a new file, with the header, when the columns changed)."""
    if path.exists():
        with open(path, newline="") as f:
            header = next(csv.reader(f), None)
        if header != list(row):
            path = path.with_name(f"{path.stem}_{datetime.now():%Y%m%d_%H%M%S}{path.suffix}")
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


class Progress:
    """The run's progress line in the terminal, every ``every`` seconds of policy time."""

    def __init__(self, demo, run, duration, every):
        self.demo, self.run, self.duration, self.every = demo, run, duration, every
        self.next = every
        self.loop_hz = []           # the policy loop's rate at every report, for the end of the run
        self.last = (0.0, 0)        # (policy time, detections) at the last report

    def maybe_report(self):
        if self.every > 0 and self.run.elapsed() >= self.next:
            self.next += self.every
            self.report()

    def report(self, final=False):
        d, run, hand, a = self.demo, self.run, self.demo.hand, self.demo.args
        t = min(run.elapsed(), self.duration)
        parts = [f"{t:5.1f}/{self.duration:.0f} s"]
        f = run.figures(hand.goal_log)
        if run.goals is not None:
            err = f.get("goal_error_deg")
            parts.append(f"goals {f.get('goals', 0)}, to the goal {'-' if err is None else f'{err:.0f} deg'}")
        elif a.camera:
            parts.append(f"turns {f.get('turns', 0.0):+6.2f}, speed {f.get('speed', 0.0):+.2f}/{run.speed:+.1f} rad/s")
        if a.camera:
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
        track = d.tracking_error()
        if track is not None:
            parts.append(f"tracking {track:.1f} deg")
        temps = hand.temperatures()
        if temps is not None:
            hot = int(np.nanargmax(temps))
            parts.append(f"hottest {d.joint_ids[hot]} {temps[hot]:.0f} C")
        if final:
            late = len(run.stalls)
            parts.append(f"late steps (> {STALL_S * 1000:.0f} ms): {late}")
            if len(self.loop_hz) > 1:
                parts.append(f"loop over the run: mean {np.mean(self.loop_hz):.1f} Hz, min {np.min(self.loop_hz):.1f} Hz")
        style = "ok" if loop is None or loop["hz"] >= 18.0 else "warn"
        log("  " + " | ".join(parts), style)
        if loop is not None and loop["hz"] < 18.0:
            log(f"  WARNING: the policy loop runs at {loop['hz']:.1f} Hz instead of 20 (laptop busy? on battery?)",
                "warn")


# ---------------------------------------------------------------------------------------------------- the demo
class Demo:
    def __init__(self, a, plan):
        self.args, self.plan, self.kind = a, plan, a.kind
        config = yaml.safe_load(HAND_CONFIG.read_text())
        self.joint_ids = list(config["joint_ids"])
        roms = config["joint_roms"]
        low, high = roms["wrist"]
        wrist = float(a.start_wrist_deg) if a.start_wrist_deg is not None else high - 2.0  # flexed, 2 deg off the stop
        if not low <= wrist <= high:
            raise SystemExit(f"common.start_wrist_deg {wrist} is outside the wrist's range {low}..{high}")
        self.start_pose = np.array([float(config["neutral_position"][j]) for j in self.joint_ids])
        self.start_pose[self.joint_ids.index("wrist")] = wrist
        self.lo = [float(roms[j][0]) for j in self.joint_ids]
        self.hi = [float(roms[j][1]) for j in self.joint_ids]
        self.thermal = ThermalModel(self.joint_ids, a.temp_limit, a.temp_margin, a.out / "thermal_history.json")
        self.hand_T_camera = camera_in_hand(a)
        self.session_dir = a.out / f"{datetime.now():%Y%m%d_%H%M%S}_demo_{self.kind}"

        self.ros = None
        self.hand = None
        self.hand_process = self.hand_log = None
        self.vision_process = self.status_q = self.events_q = self.vision_stop = None
        self.twin_process = self.twin_state_q = self.twin_image_q = self.twin_stop = None
        self.twin_stopped = False
        self.last_twin_push = 0.0
        self.camera_info = {}
        self.status = dict(model="", phase="starting", joint_names=self.joint_ids, temp_limit=a.temp_limit,
                           drop_dist=0.06, trail_id=0)
        self.last_status_push = 0.0
        self.keys = deque()
        self.pose = None            # (time, position, Rotation) of the newest detection
        self.run = None

        self.results = [None] * len(plan["runs"])
        self.rows = [None] * len(plan["runs"])
        self.run_number, self.state, self.countdown = 0, "starting", None
        self.summarized = False
        self.policy_name, self.task = a.policy.stem.removesuffix("_policy"), ""
        acc = "config" if a.servo_acc is None else str(a.servo_acc)
        self.params = [["servo acc", acc], ["max current", "config" if a.max_current is None else f"{a.max_current} mA"],
                       ["action alpha", f"{a.action_alpha:g}"]]
        if self.kind == "goals":
            self.params.append(["goal tolerance", f"{np.degrees(a.goal_tolerance):.0f} deg"])
        self.params += [["start below", f"{a.start_temp:.0f} C"], ["limit", f"{a.temp_limit:.0f} C"]]

    # -- processes
    def start_vision(self):
        """The dashboard's process: the window and its keys, and with a camera the camera and the cube detector."""
        from cube_eval.detector import DEFAULT_CKPT
        a = self.args
        ctx = multiprocessing.get_context("spawn")
        self.status_q, self.events_q, self.vision_stop = ctx.Queue(), ctx.Queue(), ctx.Event()
        if a.twin:
            self.twin_image_q = ctx.Queue(maxsize=1)
        config = dict(mock=a.mock, mock_drop_after=None, exposure_us=a.exposure_us, iso=a.iso, camera_id=a.camera_id,
                      detector_ckpt=str(DEFAULT_CKPT), device=a.device, cube_edge=a.cube_edge, window=True,
                      camera=a.camera, dashboard=True, fullscreen=a.fullscreen)
        self.vision_process = ctx.Process(target=vision.run, args=(config, self.status_q, self.events_q,
                                                                   self.vision_stop, self.twin_image_q), daemon=True)
        self.vision_process.start()
        t0 = time.time()
        while not self.camera_info:
            self.pump()
            if time.time() - t0 > 120:
                raise RuntimeError("the dashboard's process did not get ready within 120 s")
            if not self.vision_process.is_alive():
                self.pump()
                raise RuntimeError("the dashboard's process stopped")
            time.sleep(0.05)
        i = self.camera_info
        if i["camera"] == "none":
            log("  dashboard ready (no camera)", "dim")
            return
        log(f"  {i['camera']} {i['camera_id']} (USB {i['usb']}), cube detector on {i['detector_device']}", "dim")
        if "HIGH" in i["usb"] or "FULL" in i["usb"] or "LOW" in i["usb"]:
            log("  WARNING: the camera is on a slow USB connection; use a USB 3 port and cable", "warn")

    def start_twin(self):
        """The perceived state: the hand model in its measured pose (and the cube where the policy sees it), drawn by
        twin_render in a process of its own from what this process sends it (no ROS)."""
        ctx = multiprocessing.get_context("spawn")
        self.twin_state_q, self.twin_stop = ctx.Queue(maxsize=1), ctx.Event()
        self.twin_process = ctx.Process(target=twin_render.run, daemon=True, args=(
            self.twin_state_q, self.twin_image_q, self.twin_stop, self.joint_ids, self.args.cube_edge, TWIN_TITLE))
        self.twin_process.start()

    def start_hand(self):
        a = self.args
        import rclpy
        from rclpy.signals import SignalHandlerOptions
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)  # Ctrl+C must reach our own clean-up
        self.ros = rclpy
        self.hand = Hand(self.joint_ids, a.cube_pose_timeout)
        log_dir = a.out / "node_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        if a.mock:
            cmd = [sys.executable, str(MOCK_HAND_NODE), "0.1"]
        else:
            params = log_dir / "hand_params.yaml"
            params.write_text(yaml.safe_dump({"/**": {"ros__parameters": {
                "hand_controller/hand_name": "orca_v2", "hand_controller/hand_chirality": "right",
                "hand_controller/hand_cfg_name": HAND_CFG_NAME, "hand_controller/calibrate": False,
                "hand_controller/compliant_test_mode": False, "hand_controller/torque_off_on_exit": True,
                "hand_controller/servo_acceleration_raw": -1 if a.servo_acc is None else int(a.servo_acc),
                "hand_controller/max_current_ma": -1 if a.max_current is None else int(a.max_current)}}}))
            cmd = [sys.executable, str(HAND_NODE), "--ros-args", "-r", "__ns:=/right", "--params-file", str(params)]
        self.hand_log = open(log_dir / "hand_controller.log", "a")
        self.hand_log.write(f"\n===== {datetime.now():%Y-%m-%d %H:%M:%S} {' '.join(cmd)}\n")
        self.hand_log.flush()
        # a session of its own: Ctrl+C in the terminal reaches only this script, which then stops the node
        self.hand_process = subprocess.Popen(cmd, stdout=self.hand_log, stderr=subprocess.STDOUT,
                                             start_new_session=True)
        self.wait("joint positions from the hand controller", lambda: self.hand.joints_age() < 1.0, 90)
        self.wait("motor temperatures from the hand controller", lambda: self.hand.temperatures() is not None, 20)

    def wait(self, what, check, timeout):
        t0 = time.time()
        while not check():
            if self.hand_process is not None and self.hand_process.poll() is not None:
                raise RuntimeError(f"the hand controller stopped; see {self.args.out / 'node_logs/hand_controller.log'}")
            if time.time() - t0 > timeout:
                raise RuntimeError(f"no {what} after {timeout:.0f} s")
            self.tick()

    def shutdown(self):
        signal.signal(signal.SIGINT, signal.SIG_IGN)  # a second Ctrl+C must not interrupt the clean-up
        if self.hand is not None:
            self.hand.stop_policy()
            try:
                if self.hand.joints_age() < STALE_JOINTS_S:
                    self.hand.set_torque(False)
                    log("  the torque is off")
            except Exception as e:
                log(f"  could not switch the torque off ({e}): SWITCH THE HAND'S POWER OFF if it holds a pose", "error")
        if self.vision_process is not None:
            try:
                self.status_q.put(dict(self.status, quit=True, record=None))
                self.vision_process.join(timeout=12)  # closes the video
            finally:
                self.vision_stop.set()
                if self.vision_process.is_alive():
                    self.vision_process.terminate()
        if self.hand_process is not None and self.hand_process.poll() is None:
            for sig, patience in ((signal.SIGINT, 8), (signal.SIGTERM, 3), (signal.SIGKILL, 2)):
                try:
                    os.killpg(self.hand_process.pid, sig)
                    self.hand_process.wait(timeout=patience)
                    break
                except subprocess.TimeoutExpired:
                    continue
                except ProcessLookupError:
                    break
        if self.twin_process is not None:
            self.twin_stop.set()
            self.twin_process.join(timeout=3)
            if self.twin_process.is_alive():
                self.twin_process.terminate()
        if self.hand is not None:
            self.hand.close()
        if self.ros is not None:
            self.ros.try_shutdown()

    # -- the main loop's body: events from the dashboard's process, status to it and to the twin
    def pump(self):
        while self.events_q is not None:
            try:
                event = self.events_q.get_nowait()
            except queue.Empty:
                return
            kind = event[0]
            if kind == "det":
                from scipy.spatial.transform import Rotation
                _, stamp, position, quat, pixel = event
                position, rotation = np.array(position), Rotation.from_quat(quat)
                self.pose = (stamp, position, rotation)
                if self.hand_T_camera is not None and self.hand is not None:
                    self.hand.cube_detection(stamp, self.hand_T_camera @ transform(position, rotation))
                if self.run is not None:
                    self.run.detection(stamp, position, rotation, pixel)
            elif kind == "key":
                self.keys.append(event[1])
            elif kind == "ready":
                self.camera_info = event[1]
            elif kind == "error":
                raise RuntimeError(f"dashboard's process: {event[1]}")

    def tick(self, sleep=0.02):
        self.pump()
        if self.vision_process is not None and not self.vision_process.is_alive():
            raise RuntimeError("the dashboard's process stopped")
        now = time.time()
        if self.twin_process is not None and now - self.last_twin_push >= 1.0 / twin_render.RATE_HZ:
            self.last_twin_push = now
            self.push_twin(now)
        if self.hand is not None:
            for t, interval, joints_ms in self.hand.take_stalls():
                if self.run is not None:
                    self.run.stalls.append((round(t, 2), round(interval), round(joints_ms)))
                late = joints_ms > 100
                log(f"  {t:5.1f} s: policy step {interval:.0f} ms after the one before (should be 50); the joint "
                    f"positions were {joints_ms:.0f} ms old" + (" -> the hand controller was late" if late else ""),
                    "warn")
        if self.status_q is not None and now - self.last_status_push >= 0.1:
            self.last_status_push = now
            temps = self.hand.temperatures() if self.hand is not None else None
            self.status["temps"] = None if temps is None else temps.tolist()
            if self.run is not None:
                self.status.update(self.run.figures(self.hand.goal_log))
            # goal task: a banner while the hand holds still at a reached goal (dashboard and video); only this
            # banner is cleared here, not one of set_state()
            hold_s = getattr(self.hand, "goal_hold_s", 0.0) if self.hand is not None else 0.0
            hold_banner = f"GOAL REACHED - holding {hold_s:g} s" if hold_s else None
            if self.hand is not None and getattr(self.hand, "holding", False):
                self.status.update(banner=hold_banner, banner_color="green")
            elif hold_banner and self.status.get("banner") == hold_banner:
                self.status.update(banner=None)
            self.status.update(self.dashboard_status())
            self.status_q.put(dict(self.status))
        if sleep:
            time.sleep(sleep)

    def push_twin(self, now):
        if not self.twin_stopped and not self.twin_process.is_alive():
            self.twin_stopped = True
            log("  the perceived state's renderer stopped (see the error above); the demo goes on", "warn")
            return
        q = self.hand.measured() if self.hand is not None else None
        measured = q is not None and self.hand.joints_age() < 1.0 and len(q) == len(self.joint_ids)
        start = np.deg2rad(self.start_pose)
        q = np.where(np.isfinite(q), q, start) if measured else start
        pos = quat = None
        tracked = False
        lines = []
        if self.args.camera:
            cube = self.hand.cube_state(now) if self.hand is not None else None
            if cube is None:
                lines.append("no cube pose yet")
            else:
                pos, quat, age = cube
                tracked = age <= self.args.cube_pose_timeout
                lines.append("cube tracked" if tracked else f"cube lost {age:.1f} s: last pose, flag 0")
        if not measured:
            lines.append("no joint angles: start pose shown")
        goals = self.run.goals if self.run is not None else None
        goal = None if goals is None or goals.quat is None else np.array(goals.quat, dtype=np.float64)
        if goal is not None:
            lines.append("goal orientation: the cube marked goal" + (
                "" if goals.error is None else f", {np.degrees(goals.error):.0f} deg away"))
        try:
            self.twin_state_q.get_nowait()
        except queue.Empty:
            pass
        try:
            self.twin_state_q.put_nowait((now, q, pos, quat, tracked, lines, goal))
        except queue.Full:
            pass

    def tracking_error(self):
        h = self.hand
        if h is None:
            return None
        with h.lock:
            q, cmd = h.q, h.last_command if h.policy_on else None
        if q is None or cmd is None or len(q) != len(cmd):
            return None
        return float(np.degrees(np.nanmean(np.abs(cmd - q))))

    def dashboard_status(self):
        h = self.hand
        q = cmd = current = None
        loop = voltage = None
        if h is not None:
            with h.lock:
                q = None if h.q is None else np.degrees(h.q)
                cmd = None if h.last_command is None or not h.policy_on else np.degrees(h.last_command)
                current = None if h.current is None else h.current.copy()
            loop = h.loop_stats()
            if h.voltage is not None and np.any(np.isfinite(h.voltage)):
                voltage = float(np.nanmin(h.voltage))
        track = float(np.nanmean(np.abs(cmd - q))) if q is not None and cmd is not None and len(q) == len(cmd) else None
        tolist = lambda x: None if x is None else [float(v) for v in x]  # noqa: E731  (NaN: no reading)
        a = self.args
        mode = "proprio_camera" if self.kind == "proprio" and a.camera else self.kind  # the dashboard's layout
        return dict(demo=dict(
            title="ORCA Hand - in-hand cube rotation", mode=mode, mode_label=MODE_LABELS[self.kind],
            model=self.policy_name, task=self.task, axis=self.run.axis if self.run and self.kind == "axes" else None,
            goal_tol_deg=float(np.degrees(a.goal_tolerance)) if self.kind == "goals" else None,
            run=self.run_number, runs=len(self.plan["runs"]), run_labels=[label for label, _ in self.plan["runs"]],
            results=self.results, state=self.state, countdown=self.countdown, camera=bool(a.camera),
            start_temp_c=a.start_temp, joint_lo_deg=self.lo, joint_hi_deg=self.hi, q_deg=tolist(q),
            cmd_deg=tolist(cmd), current_ma=tolist(current), voltage_v=voltage, loop=loop, track_err_deg=track,
            params=self.params, keys=KEYS))

    def show(self, phase, color="yellow", banner=None, banner_color="yellow"):
        self.status.update(phase=phase, phase_color=color, banner=banner, banner_color=banner_color)

    def set_state(self, state, banner=None, color="yellow", countdown=None, note=None):
        """The dashboard's state and banner; the terminal gets a line when the state changes (note: its text)."""
        if state != self.state:
            style = dict(running="ok", cooling="warn", stopped="error", done="head", countdown="ok").get(state, "info")
            log(f"  [{state.upper()}] {note or banner or ''}".rstrip(), style)
        self.state, self.countdown = state, countdown
        phase_color = dict(running="green", cooling="orange", stopped="red", countdown="green").get(state, "yellow")
        self.show(state, phase_color, banner, color)

    def pose_age(self):
        return float("inf") if self.pose is None else time.time() - self.pose[0]

    def demo_keys(self, running=False):
        """The keys pressed since the last call: q, n, e (and p while running) raise; returns True for SPACE."""
        space = False
        names = {"q": "Q (quit)", "n": "N (next run)", "e": "E (emergency stop)", "p": "P (pause)"}
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
        if self.hand_process is not None and self.hand_process.poll() is not None:
            raise RuntimeError(f"the hand controller stopped; see {self.args.out / 'node_logs/hand_controller.log'}")

    # -- the hand between runs
    def park(self):
        """To the start pose, then torque off: the cube stays in the open palm and the motors cool down."""
        self.show("moving to the start pose")
        self.hand.move_to(self.start_pose, tick=self.tick)
        self.hand.set_torque(False)

    def wait_start(self, remaining, first):
        """Until the motors are cool enough (at the start of a run also below start_temp) and SPACE was pressed, then
        the countdown. E (torque off, then wait again), N and Q work here; P does nothing."""
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
            if cool and first and np.nanmax(temps) > self.args.start_temp:
                cool = False
            if not cool:
                armed = False
                if temps is None:
                    self.set_state("cooling", "Waiting for the motor temperatures", "orange")
                else:
                    hot = int(np.nanargmax(temps))
                    below = self.args.start_temp if first else float(np.min(self.thermal.start_below(remaining)))
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
            left = self.args.countdown_s - (time.time() - t0)
            if left <= 0:
                return
            self.set_state("countdown", f"{'START' if first else 'CONTINUE'} IN {np.ceil(left):.0f} - hands away",
                           "green", countdown=float(left), note=f"{self.args.countdown_s:.0f} s, hands away")

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

    # -- the runs
    def demo_run(self, index, label, axis):
        from cube_spin_policy import CubeSpinPolicy, axis_in_camera
        a, hand = self.args, self.hand
        speed = 0.0 if self.kind == "goals" else speed_for(a, axis)
        goals = {}
        if self.kind == "goals":
            mode = a.goal_mode
            if mode is None:
                try:
                    info = json.loads(Path(a.policy).with_suffix(".json").read_text()).get("training") or {}
                    mode = ((info.get("commands") or {}).get("object_pose") or {}).get("goal_mode")
                except (OSError, ValueError):
                    mode = None
            goals = dict(goal_mode=mode or "uniform")
            log(f"  goals: {goals['goal_mode']}, the hand holds still {a.goal_hold_s:g} s at every goal reached")
        policy = CubeSpinPolicy(a.policy, self.joint_ids, goal_ang_vel=speed, use_cube_pose=None,
                                action_alpha=a.action_alpha, axis=axis, goal_tolerance=a.goal_tolerance, **goals)
        if policy.goals is not None:
            policy.goals.hold = a.goal_hold_s > 0  # a reached goal stays until the hand has held still
        t_warm = time.perf_counter()
        warm_up(policy)
        warm_ms = (time.perf_counter() - t_warm) * 1000
        self.task = (f"goal orientations within {np.degrees(policy.goals.tolerance):.0f} deg" if policy.goals is not None
                     else f"spin {speed:+.1f} rad/s about {axis}")
        run_dir = self.session_dir / f"{index + 1}_{label.replace(' ', '_')}_{self.policy_name}"
        run_dir.mkdir(parents=True)
        # turns about the camera's viewing axis; about the commanded axis as the camera sees it (optional/axes)
        up = axis_in_camera(self.hand_T_camera, axis) if self.kind == "axes" and self.hand_T_camera is not None \
            else (0.0, 0.0, -1.0)
        run = Run(label, axis, speed, run_dir, policy, up)
        self.run_number = index + 1
        with hand.lock:
            hand.goal_log, hand.skipped = [], 0
            hand.joint_log, hand.step_log, hand.stalls = [], [], []
        self.status.update(model=f"{self.policy_name}   {self.task}", duration=a.duration, goal=speed, home=None,
                           goals=None, trail_id=self.status["trail_id"] + 1, record=None, record_on=True,
                           running=False, elapsed=0.0, turns=0.0, speed=0.0, drops=0, goal_error_deg=None)
        print(flush=True)
        rule(f"RUN {index + 1}/{len(self.plan['runs'])} ({label})  {self.policy_name}: {self.task}, {a.duration:.0f} s")
        log(f"  policy: {policy.obs_dim} observations, {'with' if policy.use_cube_pose else 'without'} the cube pose, "
            f"action alpha {a.action_alpha:g}, warm-up {WARMUP_STEPS} steps in {warm_ms:.0f} ms", "dim")
        log(f"  files: {run_dir}", "dim")
        row = None
        try:
            log("  moving to the start pose, then torque off")
            self.park()
            self.wait_start(a.duration, first=True)
            run.begin(hand.temperatures())
            self.run = run
            temps = hand.temperatures()
            log(f"  start: motors mean {np.nanmean(temps):.1f} C, hottest {np.nanmax(temps):.1f} C"
                + (", cube seen in the palm" if self.pose_age() < 1.0 else ""), "ok")
            progress = Progress(self, run, a.duration, a.progress_every)
            self.status.update(record=str(run_dir / "video.mp4") if a.record_video else None)
            while run.elapsed() < a.duration:
                segment_temps = hand.temperatures()
                hand.start_policy(policy, run.elapsed, a.duration,  # torque on first: policy time from the first step
                                  goal_hold_s=a.goal_hold_s, on_hold=run.set_hold)
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
                run.collect(hand)
                self.tick(sleep=0)  # the late steps of this stretch into the terminal
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
                    run.pauses += 1
                    self.park()
                else:
                    run.stops += 1
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
            run.collect(hand)
            self.run = None
            if run.started:
                log(f"  run over ({run.outcome}): parking the hand and writing the files")
                row = self.finish(run)
            else:
                try:
                    run_dir.rmdir()
                except OSError:
                    pass
        if row is not None:
            self.rows[index] = row
            self.results[index] = headline(self.kind, self.args.camera, row)
            r = self.results[index]
            log(f"  RESULT run {index + 1} ({label}): {r['headline']} ({r['detail']}) in {row['active_s']} s, motors "
                f"{row['temp_max_start_c']:.0f} -> {row['temp_max_end_c']:.0f} C, {row['heat_pauses']} heat pauses, "
                f"{row['late_steps']} late policy steps (max interval {row['max_step_gap_ms'] or '-'} ms)",
                "ok" if row["outcome"] == "completed" else "warn")

    def finish(self, run):
        """Park the hand, close the video and write the run's files and its row in results.csv (returned)."""
        a = self.args
        self.show(f"finished: {run.outcome}", "white")
        end_temps = self.hand.temperatures()
        try:
            if self.hand.joints_age() < STALE_JOINTS_S:
                self.park()
        except Exception as e:
            log(f"  could not park the hand: {e}", "warn")
        self.status.update(record=None, home=None)
        t0 = time.time()
        while time.time() - t0 < 1.0:  # the dashboard's process closes the video
            self.tick()
        if run.outcome in ("completed",):
            self.thermal.learn(run.segments)
        meta = dict(config=self.plan, policy=str(a.policy), start_wrist_deg=float(self.start_pose[
            self.joint_ids.index("wrist")]), camera_info=self.camera_info, session=str(self.session_dir))
        try:
            row = run.write(a, self.joint_ids, end_temps, list(self.hand.goal_log), self.hand.skipped, meta)
        except Exception as e:
            log(f"  could not write the run's files: {type(e).__name__}: {e}", "error")
            return None
        append_row(a.out / "results.csv", row)
        if run.stalls:
            worst = max(run.stalls, key=lambda s: s[1])
            log(f"  {len(run.stalls)} late policy step(s); the worst {worst[1]} ms at {worst[0]} s (steps.csv)", "warn")
        return row

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
        log(f"  {'run':10s} {'result':>14s}  {'detail':30s} {'outcome':12s} {'policy s':>8s}  {'motors C':>9s}  "
            f"{'late steps':>10s}", "dim")
        for (label, _), r, row in zip(self.plan["runs"], self.results, self.rows):
            if row is None:
                log(f"  {label:10s} {'-':>14s}  not run")
                continue
            log(f"  {label:10s} {r['headline']:>14s}  {r['detail']:30s} {row['outcome']:12s} {row['active_s']:>8}  "
                f"{row['temp_max_start_c']:>3.0f} -> {row['temp_max_end_c']:<3.0f}  {row['late_steps']:>10}",
                "ok" if row["outcome"] == "completed" else "warn")
        (self.session_dir / "summary.json").write_text(json.dumps(dict(
            kind=self.kind, model=self.policy_name, config=self.plan, results=self.results, rows=self.rows),
            indent=1, default=str))
        log(f"  files in {self.session_dir}", "dim")


def headline(kind, camera, row):
    """The dashboard's line for a finished run."""
    if kind == "goals":
        per = f"{row['s_per_goal']} s per goal" if row.get("s_per_goal") not in ("", None) else "no goal"
        return dict(headline=f"{row['goals']} goals", detail=per)
    if not camera:
        return dict(headline=f"{row['active_s']:.0f} s", detail=f"{row['outcome']}, peak {row['temp_peak_c']:.0f} C")
    return dict(headline=f"{row['turns']:+.1f} turns", detail=f"{row['mean_speed_rad_s']:+.2f} rad/s mean")


def warm_up(policy):
    """A few steps of the network, then a reset: the first real steps are as fast as the others (TorchScript
    optimises on the first calls)."""
    import torch
    with torch.no_grad():
        x = torch.zeros(1, policy.obs_dim)
        for _ in range(WARMUP_STEPS):
            policy.policy(x)
    policy.reset()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    hands.add_hand_argument(p)
    p.add_argument("--mode", required=True, choices=("proprio", "extero", "optional"))
    p.add_argument("--task", choices=("axes", "goals"), help="With --mode optional: instead of optional.task.")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="The demo's YAML (default: %(default)s).")
    p.add_argument("--dry-run", action="store_true", help="Only the pre-flight check; nothing starts.")
    p.add_argument("--mock", action="store_true", help="Rehearsal without hardware: a fake hand and a drawn cube.")
    p.add_argument("--windowed", action="store_true", help="The dashboard in a window, not full screen.")
    p.add_argument("--progress-every", type=float, default=PROGRESS_EVERY_S, metavar="S", help="A progress line in "
                   "the terminal every S seconds of a run (default %(default)g; 0: none).")
    cli = p.parse_args(argv)
    global HAND_CONFIG, HAND_CFG_NAME
    HAND_CONFIG, HAND_CFG_NAME = hands.config_path(cli.hand), hands.cfg_name(cli.hand)
    plan = load_plan(cli.config, cli.mode, cli.task)
    a = settings(plan, cli)
    label = MODE_LABELS[a.kind] + (" + camera" if a.kind == "proprio" and a.camera else "")
    rule(f"ORCA HAND DEMO  {label}")
    if preflight(a, plan):
        log("Pre-flight check failed: fix the config, nothing was started.", "error")
        return 1
    if cli.dry_run:
        log("Dry run: nothing was started.", "ok")
        return 0

    import torch
    torch.set_num_threads(a.torch_threads)
    a.out.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, _terminate)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, _terminate)
    awake = keep_awake()
    log(f"Performance: torch on {a.torch_threads} thread(s), policy loop in its own thread, W&B off, "
        f"{'caffeinate on (no sleep, no App Nap)' if awake is not None else 'no caffeinate (not macOS)'}", "dim")
    demo = Demo(a, plan)
    log(f"Results: {demo.session_dir}", "dim")
    code = 0
    t_start = time.time()
    try:
        log("Starting the dashboard" + (" with the camera and the cube detector" if a.camera else " (no camera)")
            + " ...")
        demo.start_vision()
        if a.twin:
            log("Starting the perceived state (the hand as measured" + (", the cube as the policy sees it" if a.camera
                                                                        else "") + ") ...")
            demo.start_twin()
        log(f"Starting the hand controller (servo acc {'from config.yaml' if a.servo_acc is None else a.servo_acc}, "
            f"max current {'from config.yaml' if a.max_current is None else f'{a.max_current} mA'})"
            ": THE HAND MOVES to its neutral pose ...", "warn")
        demo.start_hand()
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
