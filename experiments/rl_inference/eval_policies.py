#!/usr/bin/env python3
"""Evaluate cube-spin policies on the real ORCA v2 hand, with the camera counting the cube's turns.

    python experiments/rl_inference/eval_policies.py                    # the models in models_proprio/ not yet on
                                                                        # W&B; Enter evaluates and uploads them
    python experiments/rl_inference/eval_policies.py walrus             # one model: models_proprio/walrus_policy.pt
    python experiments/rl_inference/eval_policies.py models/ibex_policy.pt models/crow_policy.pt
    python experiments/rl_inference/eval_policies.py --pose-only       # only move to the start pose
    python experiments/rl_inference/eval_policies.py --camera-only     # only the camera view, the hand is not used
    python experiments/rl_inference/eval_policies.py --mock models/policy.pt   # no hardware: drawn cube, fake hand
    python experiments/rl_inference/eval_policies.py --wandb models/ibex_policy.pt   # and upload the run to W&B
    python experiments/rl_inference/eval_policies.py --dry-run          # only the list of untested models
    python experiments/rl_inference/eval_policies.py --upload logs/real_eval/<run folder>   # upload a saved run
    python experiments/rl_inference/eval_policies.py --models-dir models_extero --calibration ~/camera_extrinsics.npy \
        --hand-to-tag X Y Z ROLL PITCH YAW                              # policies with the cube pose

The script starts the hand controller (ROS) and a camera process, then for each policy (TorchScript files of the
proprioceptive task, 74 observations, or of the extero task with the cube pose, 82 observations, see below):

1. moves the hand to the start pose (the neutral pose with the wrist flexed) and switches the torque off,
2. waits until the cube lies still in the palm and the motors are cool enough to finish the run below --temp-limit
   (cube_eval/thermal.py) and for a cold start (the hottest at most 42.5 deg C, else the run is not in the
   leaderboard), then counts down and starts,
3. runs the policy for --duration seconds of policy time at the model's goal speed (its goal on W&B, else 2 rad/s;
   --speed sets one for all). The clock stops while the run is paused:
   - the cube fell (or is not seen for --lost-timeout seconds): the hand moves to the start pose; the run continues
     when the cube lies still in the palm again (or when you press the space bar),
   - a motor reached --temp-limit: the torque is switched off where the hand is; the run continues when the motors
     are cool enough for the rest of the time,
4. moves back to the start pose, switches the torque off and writes the results.

Keys in the camera window: space = start / continue now, n = next policy, q = quit. Ctrl+C in the terminal stops
everything and leaves the hand with the torque off.

Results, in --out (default logs/real_eval): results.csv with one row per run, and a folder per run with video.mp4
(the camera view with the figures drawn in; cooling pauses are left out), trace.csv (every cube detection: pixel and
position of its centre, orientation), joints.csv (commands, measured angles and currents at 100 Hz),
temperatures.csv and health.csv (per motor, once per second), joint_tracking.csv, turns.csv and meta.json.

With --wandb (or --auto), every finished run also goes to W&B, rwr-skin/real_hand_eval, in the format of the other
real-hand evaluation (cube_eval/wandb_results.py), in a background process while the next policy starts.

Without policy files (= --auto) the script lists the models in --models-dir (default models_proprio/, filled by
download_policies.py) that have no run on W&B yet, and evaluates and uploads them after Enter.

Policies with the cube pose (task orca-v2-inhand-repose-extero-v0) get the camera's cube pose in the frame of the
simulation, as policy_inference_cube_spin_v2.py on feat/exteroception-deployment gives it (cube_spin_policy.CubePose):
the newest pose with the flag 1, or the last one with the flag 0 when it is older than --cube-pose-timeout. For that the
script needs the camera's pose relative to the hand: --calibration (cam_T_tag from apriltag_calibration.py, made with the
camera where it is now) and --hand-to-tag (the hand_to_tag_* arguments of cube_spin_v2.launch.py). --camera-only with
--calibration prints the cube's position in the frame of the simulation: with the cube in the palm it is a few cm from
(0, 0, -0.05), +z up. Such runs also write cube_obs.csv (what the policy saw at each step).

Works on Linux and macOS. The cube detector uses CUDA, Apple's GPU or the CPU, whichever is there.
"""

import argparse
import csv
import hashlib
import json
import multiprocessing
import os
import queue
import signal
import socket
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

from cube_eval import vision  # noqa: E402
from cube_eval.detector import DEFAULT_CKPT  # noqa: E402
from cube_eval import wandb_results  # noqa: E402
from cube_eval.metrics import joint_tracking, progress_metrics, status_names, tracking_summary  # noqa: E402
from cube_eval.thermal import TemperatureFilter, ThermalModel  # noqa: E402
from cube_eval.turns import DropMonitor, RestDetector, TurnCounter, lateral_offset  # noqa: E402

HAND_CONFIG = REPO / "src/hand_control/orca/configs/orcahand_v2/right/config.yaml"
HAND_NODE = REPO / "src/hand_control/hand_control_node.py"
MOCK_HAND_NODE = HERE / "cube_eval/mock_hand_node.py"
WANDB_RESULTS = HERE / "cube_eval/wandb_results.py"
POLICY_HZ = 20.0
COUNTDOWN_S = 3.0
STALE_JOINTS_S = 2.0


class Quit(Exception):
    """q was pressed: stop the session after cleaning up."""


class Skip(Exception):
    """n was pressed: go to the next policy."""


class HandLink:
    """The ROS side: the hand controller's topics and torque service, and the policy loop at 20 Hz."""

    def __init__(self, joint_ids, cube_pose_timeout):
        import rclpy
        from cube_spin_policy import CubePose
        from rclpy.callback_groups import ReentrantCallbackGroup
        from rclpy.executors import MultiThreadedExecutor
        from rclpy.node import Node
        from std_msgs.msg import Float32MultiArray, Int32MultiArray
        from std_srvs.srv import SetBool

        self._msg, self._SetBool = Float32MultiArray, SetBool
        self.joint_ids = list(joint_ids)
        self.lock = threading.Lock()
        self.q = self.qd = self.current = self.voltage = None
        self.q_time = 0.0
        self.temps, self.temps_time = None, 0.0
        self.temp_filter = TemperatureFilter()
        self.status_or = np.zeros(len(joint_ids), dtype=np.int64)  # servo status bits seen since reset_health()
        self.min_voltage = float("inf")
        self.policy, self.policy_on = None, False
        self.last_command = None
        self.clock = None       # callable: the run's policy time, for the joint log
        self.joint_log = []     # (policy time, command, measured, current) while the policy runs
        self.cube = CubePose(cube_pose_timeout)  # the cube pose for policies of the extero task
        self.cube_log = []      # (policy time, cube pose observation) while such a policy runs
        self.torque_on = None

        self.node = Node("cube_spin_evaluation")
        group = ReentrantCallbackGroup()
        sub = lambda kind, name, cb: self.node.create_subscription(kind, f"/right/hand/{name}", cb, 20,  # noqa: E731
                                                                   callback_group=group)
        sub(Float32MultiArray, "joint_position_from_motor", self._position)
        sub(Float32MultiArray, "joint_velocity_from_motor", self._velocity)
        sub(Float32MultiArray, "motor_current", self._current)
        sub(Float32MultiArray, "motor_temperature", self._temperature)
        sub(Float32MultiArray, "motor_voltage", self._voltage)
        sub(Int32MultiArray, "motor_status", self._status)
        self.command_pub = self.node.create_publisher(Float32MultiArray, "/right/hand/policy_output", 10)
        self.torque_client = self.node.create_client(SetBool, "/right/hand/set_torque_enabled", callback_group=group)
        self.node.create_timer(1.0 / POLICY_HZ, self._policy_step, callback_group=group)
        self.executor = MultiThreadedExecutor(num_threads=3)
        self.executor.add_node(self.node)
        threading.Thread(target=self.executor.spin, daemon=True).start()
        self._rclpy = rclpy

    # -- callbacks (executor threads)
    def _position(self, msg):
        q = np.array(msg.data, dtype=np.float64)
        with self.lock:
            self.q, self.q_time = q, time.time()
            if self.policy_on and self.last_command is not None and self.clock is not None:
                self.joint_log.append((self.clock(), self.last_command, q, self.current))

    def _velocity(self, msg):
        self.qd = np.array(msg.data, dtype=np.float64)

    def _current(self, msg):
        self.current = np.array(msg.data, dtype=np.float64)

    def _temperature(self, msg):
        with self.lock:
            self.temps, self.temps_time = self.temp_filter.update(msg.data).copy(), time.time()

    def _voltage(self, msg):
        self.voltage = np.array(msg.data, dtype=np.float64)
        if np.any(np.isfinite(self.voltage)):
            self.min_voltage = min(self.min_voltage, float(np.nanmin(self.voltage)))

    def _status(self, msg):
        with self.lock:
            self.status_or |= np.array(msg.data, dtype=np.int64)

    def _policy_step(self):
        with self.lock:
            if not self.policy_on or self.q is None or self.qd is None:
                return
            cube = self.cube.observation(time.time()) if self.policy.use_cube_pose else None
            out = self.policy.step(self.q, self.qd, cube)
            if out is None:
                return
            if cube is not None and self.clock is not None:
                self.cube_log.append((self.clock(), cube))
            self.last_command = np.asarray(out[0], dtype=np.float64)
            command = self.last_command
        self._publish(command)

    def _publish(self, joint_pos_rad):
        self.command_pub.publish(self._msg(data=np.asarray(joint_pos_rad, dtype=np.float32).tolist()))

    # -- from the main thread
    def joints_age(self):
        return time.time() - self.q_time if self.q_time else float("inf")

    def temperatures(self, max_age=5.0):
        """Filtered motor temperatures in joint order, or None without a recent reading."""
        with self.lock:
            if self.temps is None or time.time() - self.temps_time > max_age:
                return None
            return self.temps.copy()

    def cube_detection(self, stamp, hand_T_cube):
        with self.lock:
            self.cube.update(stamp, hand_T_cube)

    def reset_health(self):
        with self.lock:
            self.status_or[:] = 0
            self.min_voltage = float("inf")
            self.temp_filter.glitches = 0

    def start_policy(self, policy, clock):
        """Torque on at the present pose, then the policy takes over, starting as after a reset in the task."""
        self.set_torque(True)
        with self.lock:
            policy.reset()
            self.policy, self.clock, self.last_command, self.policy_on = policy, clock, None, True

    def stop_policy(self):
        with self.lock:
            self.policy_on = False

    def set_torque(self, enabled, timeout=5.0):
        if enabled:  # make the measured pose the goal first, so the hand does not jump to an old goal
            q = self.q
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
        start = self.q.copy() if self.q is not None and np.all(np.isfinite(self.q)) else target
        steps = max(1, int(seconds * POLICY_HZ))
        for i in range(1, steps + 1):
            self._publish(start + (target - start) * i / steps)
            time.sleep(1.0 / POLICY_HZ)
            if tick is not None:
                tick(sleep=0.0)
        time.sleep(0.5)


class Evaluation:
    def __init__(self, args):
        self.args = args
        config = yaml.safe_load(HAND_CONFIG.read_text())
        self.joint_ids = list(config["joint_ids"])
        low, high = config["joint_roms"]["wrist"]
        wrist = args.start_wrist_deg if args.start_wrist_deg is not None else high - 2.0  # flexed, 2 deg off the stop
        if not low <= wrist <= high:
            raise SystemExit(f"--start-wrist-deg {wrist} is outside the wrist's range {low}..{high}")
        self.start_pose = np.array([float(config["neutral_position"][j]) for j in self.joint_ids])
        self.start_pose[self.joint_ids.index("wrist")] = wrist
        args.start_wrist_deg = wrist  # so that meta.json has the angle that was used
        self.out = Path(args.out)
        self.thermal = ThermalModel(self.joint_ids, args.temp_limit, args.temp_margin, self.out / "thermal_history.json")
        self.hand_T_camera = camera_in_hand(args)  # None: no policies with the cube pose
        self.last_cube_print = 0.0

        self.hand = None
        self.hand_process = None
        self.vision_process = None
        self.status_q = self.events_q = self.vision_stop = None
        self.camera_info = {}
        self.status = dict(model="", phase="starting", joint_names=self.joint_ids, temp_limit=args.temp_limit,
                           drop_dist=args.drop_dist, trail_id=0)
        self.last_status_push = 0.0
        self.keys = deque()
        self.pose = None            # (time, position, Rotation) of the newest detection
        self.rest = RestDetector()
        self.gravity = []           # accelerometer directions (unit vectors) during the session
        self.run = None             # the running policy's Run
        self.uploads = []           # W&B upload processes (--wandb)

    # -- processes
    def start_vision(self):
        ctx = multiprocessing.get_context("spawn")
        self.status_q, self.events_q, self.vision_stop = ctx.Queue(), ctx.Queue(), ctx.Event()
        a = self.args
        config = dict(mock=a.mock, mock_drop_after=a.mock_drop_after, exposure_us=a.exposure_us, iso=a.iso,
                      camera_id=a.camera_id, detector_ckpt=str(a.detector), device=a.device, cube_edge=a.cube_edge,
                      window=not a.no_window)
        self.vision_process = ctx.Process(target=vision.run, args=(config, self.status_q, self.events_q, self.vision_stop),
                                          daemon=True)
        self.vision_process.start()
        print("Starting the camera and the cube detector ...")
        t0 = time.time()
        while not self.camera_info:
            self.pump()
            if time.time() - t0 > 120:
                raise RuntimeError("the camera process did not get ready within 120 s")
            if not self.vision_process.is_alive():
                self.pump()
                raise RuntimeError("the camera process stopped")
            time.sleep(0.05)
        i = self.camera_info
        print(f"  {i['camera']} {i['camera_id']} (USB {i['usb']}), detector on {i['detector_device']}")
        if "HIGH" in i["usb"] or "FULL" in i["usb"] or "LOW" in i["usb"]:
            print("  WARNING: the camera is on a slow USB connection; use a USB 3 port and cable")

    def start_hand(self):
        import rclpy
        from rclpy.signals import SignalHandlerOptions
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)  # Ctrl+C must reach our own clean-up
        self.hand = HandLink(self.joint_ids, self.args.cube_pose_timeout)
        a = self.args
        if not a.no_hand_launch:
            log_dir = self.out / "node_logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            if a.mock:
                cmd = [sys.executable, str(MOCK_HAND_NODE), str(a.mock_heat_rate)]
            else:
                params = log_dir / "hand_params.yaml"
                params.write_text(yaml.safe_dump({"/**": {"ros__parameters": {
                    "hand_controller/hand_name": "orca_v2", "hand_controller/hand_chirality": "right",
                    "hand_controller/hand_cfg_name": "orcahand_v2", "hand_controller/calibrate": False,
                    "hand_controller/compliant_test_mode": False, "hand_controller/torque_off_on_exit": True}}}))
                cmd = [sys.executable, str(HAND_NODE), "--ros-args", "-r", "__ns:=/right", "--params-file", str(params)]
                print("Starting the hand controller: THE HAND MOVES to its neutral pose, then to the start pose.")
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
                raise RuntimeError(f"the hand controller stopped; see {self.out / 'node_logs/hand_controller.log'}")
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
                    print("The torque is off.")
            except Exception as e:
                print(f"Could not switch the torque off ({e}); switch the hand's power off if it holds a pose.")
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
        if self.hand is not None:
            self.hand._rclpy.try_shutdown()

    # -- the loop body: events from the camera process, status to it
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
                self.rest.add(stamp, position)
                if self.hand is not None and self.hand_T_camera is not None:
                    self.hand.cube_detection(stamp, self.hand_T_camera @ transform(position, rotation))
                if self.run is not None:
                    self.run.detection(stamp, position, rotation, pixel)
            elif kind == "key":
                self.keys.append(event[1])
            elif kind == "imu":
                acc = np.array(event[1])
                if np.linalg.norm(acc) > 1.0:
                    self.gravity.append(acc / np.linalg.norm(acc))
            elif kind == "ready":
                self.camera_info = event[1]
            elif kind == "error":
                raise RuntimeError(f"camera process: {event[1]}")

    def tick(self, sleep=0.02):
        self.pump()
        if self.vision_process is not None and not self.vision_process.is_alive():
            raise RuntimeError("the camera process stopped")
        now = time.time()
        if self.status_q is not None and now - self.last_status_push >= 0.1:
            self.last_status_push = now
            temps = self.hand.temperatures() if self.hand is not None else None
            self.status["temps"] = None if temps is None else temps.tolist()
            if self.run is not None:
                self.status.update(self.run.figures())
            self.status_q.put(dict(self.status))
        if sleep:
            time.sleep(sleep)

    def key(self):
        """The next key pressed in the window ("" if none); q and n end the session or the policy."""
        while self.keys:
            k = self.keys.popleft()
            if k == "q":
                raise Quit
            if k == "n":
                raise Skip
            if k == " ":
                return " "
        return ""

    def show(self, phase, color="yellow", banner=None, banner_color="yellow"):
        self.status.update(phase=phase, phase_color=color, banner=banner, banner_color=banner_color)

    def pose_age(self):
        return float("inf") if self.pose is None else time.time() - self.pose[0]

    def camera_tilt(self):
        """Degrees between the camera's viewing axis and gravity, and how much the accelerometer's direction changed
        during the session (a bumped camera). (None, None) without the IMU."""
        if len(self.gravity) < 3:
            return None, None
        g = np.array(self.gravity)
        mean = g.mean(axis=0) / np.linalg.norm(g.mean(axis=0))
        tilt = np.degrees(np.arccos(np.clip(abs(mean[2]), 0, 1)))
        moved = np.degrees(np.arccos(np.clip(g @ np.median(g[:5], axis=0) / np.linalg.norm(np.median(g[:5], axis=0)), -1, 1)))
        return float(tilt), float(np.max(moved))

    def print_cube_pose(self):
        """--camera-only with --calibration: once per second, the cube pose as a policy with the cube pose gets it."""
        from cube_spin_policy import CubePose
        now = time.time()
        if self.hand_T_camera is None or self.pose_age() > 0.5 or now - self.last_cube_print < 1.0:
            return
        self.last_cube_print = now
        cube = CubePose()
        cube.update(self.pose[0], self.hand_T_camera @ transform(self.pose[1], self.pose[2]))
        print(f"  cube in the frame of the simulation: pos {np.round(cube.pos, 3)} m, quat (w, x, y, z) "
              f"{np.round(cube.quat, 2)}")

    # -- the hand between runs
    def park(self):
        """To the start pose, then torque off: the cube stays in the open palm and the motors cool down."""
        self.show("moving to the start pose")
        self.hand.move_to(self.start_pose, tick=self.tick)
        self.hand.set_torque(False)

    # -- waiting for cube and temperature; returns the cube's resting position
    def wait_ready(self, remaining, home=None, home_radius=None, what="start", cold_start=False):
        """Until the motors are cool enough for ``remaining`` seconds and the cube lies still (within home_radius of
        home, if given), then a countdown. Space skips the wait for the cube. Returns the resting position.
        cold_start: also until the hottest motor is at most wandb_results.MAX_START_C (else not in the leaderboard)."""
        countdown_from, forced = None, False
        self.status["want_cube"] = True
        try:
            while True:
                self.tick()
                if self.hand.joints_age() > STALE_JOINTS_S:
                    raise RuntimeError("the hand controller stopped sending joint positions")
                now = time.time()
                forced = forced or (self.key() == " " and (home is not None or self.pose_age() < 1.0))
                temps = self.hand.temperatures()
                cool = temps is not None and self.thermal.ready(temps, remaining)
                too_warm = cool and cold_start and np.nanmax(temps) > wandb_results.MAX_START_C
                rest = self.rest.resting(now)
                if rest is not None and home is not None and home_radius and lateral_offset(rest, home) > home_radius:
                    rest = None
                if forced and rest is None:
                    rest = home if home is not None else self.pose[1]
                if not cool:
                    countdown_from = None
                    if temps is None:
                        self.show("waiting for motor temperatures", "orange")
                    else:
                        worst = int(np.argmin(self.thermal.headroom(temps, remaining)))
                        self.show("cooling down", "orange",
                                  f"COOLING: {self.joint_ids[worst]} {temps[worst]:.0f} C, {what} below "
                                  f"{self.thermal.start_below(remaining)[worst]:.0f} C", "orange")
                elif too_warm:
                    countdown_from = None
                    hottest = int(np.nanargmax(temps))
                    self.show("cooling down", "orange",
                              f"COOLING: {self.joint_ids[hottest]} {temps[hottest]:.0f} C, {what} below "
                              f"{wandb_results.MAX_START_C:.1f} C (cold start)", "orange")
                elif rest is None:
                    countdown_from = None
                    self.show("waiting for the cube", "yellow", "Put the cube in the palm  (space = " + what + " now)")
                else:
                    countdown_from = countdown_from or now
                    left = COUNTDOWN_S - (now - countdown_from)
                    if left <= 0:
                        return rest
                    self.show(f"{what} in {np.ceil(left):.0f} s", "green", f"{what.upper()} IN {np.ceil(left):.0f} - hands away",
                              "green")
        finally:
            self.status["want_cube"] = False

    # -- one policy
    def evaluate(self, policy_path, speed):
        from cube_spin_policy import CubeSpinPolicy
        a = self.args
        name = policy_path.stem.removesuffix("_policy")
        policy = CubeSpinPolicy(policy_path, self.joint_ids, goal_ang_vel=speed, use_cube_pose=None)
        run_dir = self.out / f"{datetime.now():%Y%m%d_%H%M%S}_{name}"
        run_dir.mkdir(parents=True)
        run = Run(name, policy_path, run_dir, a, self.joint_ids, policy.obs_dim, self.hand_T_camera)
        label = f"{name}{' (cube pose)' if policy.use_cube_pose else ''}"
        self.status.update(model=f"{label}   goal {speed:+.1f} rad/s", duration=a.duration, goal=speed, home=None,
                           trail_id=self.status["trail_id"] + 1, record=None, record_on=True, running=False)
        print(f"\n=== {label}: goal {speed:+.1f} rad/s, {a.duration:.0f} s")
        hand = self.hand
        try:
            self.park()
            hand.reset_health()
            home = self.wait_ready(a.duration, cold_start=True)
            run.begin(home, hand.temperatures(), speed)
            self.run = run
            self.status.update(home=home.tolist(), record=str(run_dir / "video.mp4"))
            while run.elapsed() < a.duration:
                # ---- a stretch of policy time
                segment_temps = hand.temperatures()
                hand.start_policy(policy, run.elapsed)
                run.resume()
                self.status.update(running=True, record_on=True)
                reason = ""
                while not reason:
                    self.tick()
                    now = time.time()
                    self.show("RUNNING   (n = next policy, q = quit)", "green")
                    run.log_health(hand, "running")
                    temps = hand.temperatures()
                    if run.elapsed() >= a.duration:
                        reason = "time"
                    elif hand.joints_age() > STALE_JOINTS_S:
                        reason = "controller"
                    elif temps is not None and self.thermal.over_limit(temps) is not None:
                        reason = "heat"
                    else:
                        reason = run.drop.check(now)  # "", "dropped" or "lost"
                    self.key()  # n and q raise
                hand.stop_policy()
                run.pause()
                self.status.update(running=False)
                run.segments.append((run.elapsed() - run.segment_start_elapsed, segment_temps, hand.temperatures()))
                if reason == "time":
                    break
                if reason == "controller":
                    raise RuntimeError("the hand controller stopped sending joint positions")

                # ---- a pause
                t_pause = time.time()
                remaining = a.duration - run.elapsed()
                if reason == "heat":
                    hot = self.thermal.over_limit(hand.temperatures())
                    print(f"  {run.elapsed():5.1f} s: {self.joint_ids[hot]} reached {hand.temperatures()[hot]:.0f} C, "
                          "cooling with the torque off")
                    hand.set_torque(False)  # where it is: no movement
                    self.status.update(record_on=False)
                    run.heat_pauses += 1
                    while True:
                        self.tick()
                        run.log_health(hand, "cooling")
                        temps = hand.temperatures()
                        if hand.joints_age() > STALE_JOINTS_S:
                            raise RuntimeError("the hand controller stopped sending joint positions")
                        if temps is not None and self.thermal.ready(temps, remaining):
                            break
                        if temps is not None:
                            worst = int(np.argmin(self.thermal.headroom(temps, remaining)))
                            self.show("cooling down", "orange",
                                      f"COOLING: {self.joint_ids[worst]} {temps[worst]:.0f} C, continue below "
                                      f"{self.thermal.start_below(remaining)[worst]:.0f} C", "orange")
                        self.key()
                    run.pause_heat_s += time.time() - t_pause
                    self.status.update(record_on=True, banner=None)
                    # with the torque off the cube may have slipped out: then go on as after a drop
                    if self.pose_age() < 1.0 and lateral_offset(self.pose[1], run.home) <= a.drop_dist:
                        continue
                    reason, t_pause = "dropped", time.time()
                run.drops += 1
                run.drop_times.append(round(run.elapsed(), 1))
                print(f"  {run.elapsed():5.1f} s: the cube {'fell' if reason == 'dropped' else 'is not seen'} "
                      f"(drop {run.drops})")
                if run.drops > a.max_drops:
                    run.outcome = "too many drops"
                    break
                self.park()
                run.home = self.wait_ready(remaining, home=run.home, home_radius=a.home_radius, what="continue")
                run.drop.home = run.home
                self.status.update(home=run.home.tolist(), banner=None)
                run.pause_drop_s += time.time() - t_pause
            if run.outcome == "":
                run.outcome = "completed"
        except Skip:
            run.outcome = "skipped"
            print("  skipped (n)")
        except Quit:
            run.outcome = "aborted"
            raise
        except KeyboardInterrupt:
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
                self.finish(run)
            else:
                try:
                    run_dir.rmdir()
                except OSError:
                    pass

    def finish(self, run):
        """Park the hand, close the video and write the run's files and its row in results.csv."""
        self.show(f"finished: {run.outcome}", "white")
        end_temps = self.hand.temperatures()
        try:
            if self.hand.joints_age() < STALE_JOINTS_S:
                self.park()
        except Exception as e:
            print(f"  could not park the hand: {e}")
        self.status.update(record=None, home=None)
        self.tick()
        time.sleep(1.0)  # the camera process closes the video
        tilt, moved = self.camera_tilt()
        rates = self.thermal.learn(run.segments) if run.outcome in ("completed", "too many drops") else None
        row = run.write(self.hand, end_temps, rates, self.camera_info, tilt, moved)
        append_row(self.out / "results.csv", row)
        if row["drops"]:
            print(f"  before the first drop: {row['turns_before_first_drop']:+.2f} turns in "
                  f"{row['duration_before_first_drop_s']} s (the result on W&B)")
        print(f"  {row['outcome'].upper()}: {row['turns']:+.2f} turns in {row['active_s']} s "
              f"({row['mean_speed_rad_s']:+.2f} rad/s, goal {row['goal_rad_s']:+.1f}), {row['drops']} drops, "
              f"tracked {row['tracked_fraction']:.0%}, motors {row['temp_max_start_c']:.0f} -> {row['temp_max_end_c']:.0f} C")
        if row["cube_obs_tracked_fraction"] != "":
            print(f"  the policy had a cube pose at most {self.args.cube_pose_timeout} s old in "
                  f"{row['cube_obs_tracked_fraction']:.0%} of its steps (cube_obs.csv)")
        print(f"  files in {run.run_dir}")
        if self.args.wandb:
            self.upload(run.run_dir, row["outcome"])

    def upload(self, run_dir, outcome):
        """The run to W&B in a background process (log: node_logs/wandb_upload.log), so the next run can start."""
        if self.args.mock:
            print("  not uploaded to W&B: a --mock run")
            return
        if outcome not in wandb_results.UPLOADED_OUTCOMES:
            print(f"  not uploaded to W&B: {outcome}")
            return
        log_dir = self.out / "node_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(WANDB_RESULTS), "--entity", self.args.wandb_entity, "--project",
               self.args.wandb_project, "upload", str(run_dir)] + (["--video"] if self.args.wandb_video else [])
        with open(log_dir / "wandb_upload.log", "a") as log:
            log.write(f"\n===== {datetime.now():%Y-%m-%d %H:%M:%S} {' '.join(cmd)}\n")
            log.flush()
            self.uploads.append((run_dir, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                                           start_new_session=True)))
        print(f"  uploading to W&B ({self.args.wandb_entity}/{self.args.wandb_project}) in the background")

    def wait_for_uploads(self, timeout=600):
        """After the hand is off: wait for the uploads, and say which ones failed."""
        if not self.uploads:
            return
        if any(p.poll() is None for _, p in self.uploads):
            print("Waiting for the W&B upload(s) (Ctrl+C: stop waiting; they go on by themselves) ...")
        t0 = time.time()
        try:
            while any(p.poll() is None for _, p in self.uploads) and time.time() - t0 < timeout:
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        failed = [str(d) for d, p in self.uploads if p.poll() not in (0, None)]
        if failed:
            print(f"W&B upload failed (see {self.out / 'node_logs/wandb_upload.log'}). Upload again with:\n"
                  f"  python experiments/rl_inference/eval_policies.py --upload {' '.join(failed)}")
        elif all(p.poll() == 0 for _, p in self.uploads):
            print(f"Uploaded to W&B: https://wandb.ai/{self.args.wandb_entity}/{self.args.wandb_project}")


class Run:
    """What one policy's run measures and writes."""

    def __init__(self, name, policy_path, run_dir, args, joint_ids, observations, hand_T_camera=None):
        self.name, self.policy_path, self.run_dir, self.args, self.joint_ids = name, Path(policy_path), run_dir, args, joint_ids
        self.observations, self.hand_T_camera = observations, hand_T_camera
        self.outcome = ""
        self.started = False
        self.active = False
        self.accumulated = 0.0       # policy seconds of the finished stretches
        self.segment_start = None    # wall time of the running stretch
        self.segment_start_elapsed = 0.0
        self.segments = []           # (policy seconds, temperatures at start, at end)
        self.drops, self.drop_times = 0, []
        self.heat_pauses, self.pause_heat_s, self.pause_drop_s = 0, 0.0, 0.0
        self.trace = []              # every detection
        self.detection_elapsed = []  # policy time of the detections during policy time
        self.health = []             # (wall seconds, policy seconds, state, temperatures, voltages, status bits)
        self.last_health = 0.0

    def begin(self, home, temps, speed):
        self.started = True
        self.t0 = time.time()
        self.home = np.asarray(home, dtype=float)
        self.start_temps = temps
        self.speed = speed
        self.sign = 1.0 if speed >= 0 else -1.0
        self.counter = TurnCounter()
        self.drop = DropMonitor(self.home, self.args.drop_dist, self.args.drop_confirm, self.args.lost_timeout)

    def elapsed(self):
        """Policy time: seconds the policy has run."""
        if self.active:
            return self.accumulated + time.time() - self.segment_start
        return self.accumulated

    def resume(self):
        self.segment_start, self.segment_start_elapsed = time.time(), self.accumulated
        self.active = True
        self.counter.rebase()
        self.drop.start(self.segment_start)

    def pause(self):
        if self.active:
            self.accumulated += time.time() - self.segment_start
            self.active = False

    def detection(self, stamp, position, rotation, pixel):
        if not self.started:
            return
        elapsed, accepted = "", ""
        if self.active and stamp >= self.segment_start:
            elapsed = self.accumulated + stamp - self.segment_start
            accepted = int(self.counter.add(elapsed, rotation))
            self.drop.add(stamp, position)
            self.detection_elapsed.append(elapsed)
            elapsed = round(elapsed, 3)
        self.trace.append([round(stamp - self.t0, 3), elapsed, *np.round(pixel, 1), *np.round(position, 4),
                           *np.round(rotation.as_quat(), 4), accepted,
                           round(lateral_offset(position, self.home) * 1000, 1)])

    def figures(self):
        """For the view."""
        if not self.started:
            return {}
        elapsed = self.elapsed()
        return dict(elapsed=elapsed, turns=self.sign * self.counter.turns, drops=self.drops,
                    speed=self.sign * self.counter.speed(elapsed, 5.0))

    def log_health(self, hand, state):
        now = time.time()
        if now - self.last_health < 1.0:
            return
        self.last_health = now
        temps = hand.temperatures()
        self.health.append((round(now - self.t0, 1), round(self.elapsed(), 2), state, temps, hand.voltage,
                            hand.status_or.copy()))

    def write(self, hand, end_temps, rates, camera_info, camera_tilt, camera_moved):
        a, joints, d = self.args, self.joint_ids, self.run_dir
        active = self.accumulated
        counter = self.counter
        n = len(joints)
        nan = np.full(n, np.nan)
        start = nan if self.start_temps is None else np.asarray(self.start_temps)
        end = nan if end_temps is None else np.asarray(end_temps)
        peak = np.nanmax([t for _, _, _, t, _, _ in self.health if t is not None] + [start, end], axis=0)
        hottest = int(np.nanargmax(peak)) if np.any(np.isfinite(peak)) else 0

        # how well the cube was seen during policy time
        stamps = np.array([0.0] + self.detection_elapsed + [active])
        gaps = np.diff(stamps) if active > 0 else np.zeros(1)
        tracked = 1.0 - float(np.sum(np.maximum(0.0, gaps - 0.5))) / active if active > 0 else 0.0
        offsets = np.array(self.drop.offsets) if self.drop.offsets else np.zeros(1)

        with hand.lock:
            log = list(hand.joint_log)
            hand.joint_log = []
            cube_log = list(hand.cube_log)
            hand.cube_log = []
            status = hand.status_or.copy()
        if cube_log:
            with open(d / "cube_obs.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["policy_time_s", "x_m", "y_m", "z_m", "qw", "qx", "qy", "qz", "tracked"])
                w.writerows([[f"{t:.3f}", *np.round(cube[:7], 4), int(cube[7])] for t, cube in cube_log])
        cube_obs_tracked = round(float(np.mean([cube[7] for _, cube in cube_log])), 3) if cube_log else ""
        tracking = []
        if log:
            times = np.array([r[0] for r in log])
            commands, measured = np.array([r[1] for r in log]), np.array([r[2] for r in log])
            currents = None if any(r[3] is None for r in log) else np.array([r[3] for r in log])
            tracking = joint_tracking(times, commands, measured, currents, joints)
            with open(d / "joints.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["policy_time_s", *[f"cmd_deg_{j}" for j in joints], *[f"meas_deg_{j}" for j in joints],
                            *[f"current_ma_{j}" for j in joints]])
                for i in range(len(times)):
                    w.writerow([f"{times[i]:.3f}", *np.round(np.degrees(commands[i]), 2),
                                *np.round(np.degrees(measured[i]), 2),
                                *(np.round(currents[i]).astype(int) if currents is not None else [""] * n)])
        if tracking:
            with open(d / "joint_tracking.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(tracking[0]))
                w.writeheader()
                w.writerows(tracking)
        with open(d / "trace.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time_s", "policy_time_s", "pixel_x", "pixel_y", "x_m", "y_m", "z_m", "qx", "qy", "qz", "qw",
                        "accepted", "offset_mm"])
            w.writerows(self.trace)
        with open(d / "temperatures.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time_s", "policy_time_s", "state", *joints])
            w.writerows([[t, pt, state, *(np.round(temps, 1) if temps is not None else [""] * n)]
                         for t, pt, state, temps, _, _ in self.health])
        with open(d / "health.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time_s", "policy_time_s", "state", *[f"voltage_v_{j}" for j in joints],
                        *[f"status_{j}" for j in joints]])
            w.writerows([[t, pt, state, *(np.round(volt, 1) if volt is not None else [""] * n), *bits.tolist()]
                         for t, pt, state, _, volt, bits in self.health])
        # the turns at every counted detection, over policy time (positive = the commanded direction)
        curve = [(t, self.sign * angle / (2 * np.pi)) for t, angle in counter.history]
        with open(d / "turns.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["policy_time_s", "turns"])
            w.writerows([[round(t, 3), round(c, 4)] for t, c in curve])
        # the other evaluation ends a run at the first drop: these figures are up to there
        until = self.drop_times[0] if self.drop_times else active
        before_drop = self.sign * counter.angle_at(until) / (2 * np.pi) if self.drop_times else self.sign * counter.turns
        progress = progress_metrics([t for t, _ in curve if t <= until], [c for t, c in curve if t <= until], until)

        errors = "; ".join(f"{j}: {'+'.join(status_names(b))}" for j, b in zip(joints, status) if b)
        row = dict(
            time=datetime.fromtimestamp(self.t0).strftime("%Y-%m-%d %H:%M:%S"), model=self.name,
            policy_file=str(self.policy_path), policy_sha256=sha256(self.policy_path)[:12], git_commit=git_commit(),
            observations=self.observations, goal_rad_s=self.speed, outcome=self.outcome, active_s=round(active, 1),
            wall_s=round(time.time() - self.t0, 1),
            turns=round(self.sign * counter.turns, 2),
            mean_speed_rad_s=round(self.sign * counter.angle / active, 3) if active > 0 else 0.0,
            turns_before_first_drop=round(before_drop, 2), duration_before_first_drop_s=round(until, 1),
            speed_before_first_drop_rad_s=round(before_drop * 2 * np.pi / until, 3) if until > 0 else 0.0, **progress,
            drops=self.drops, first_drop_s=self.drop_times[0] if self.drop_times else "",
            pause_drop_s=round(self.pause_drop_s, 1), heat_pauses=self.heat_pauses,
            pause_heat_s=round(self.pause_heat_s, 1),
            tracked_fraction=round(max(0.0, tracked), 3), cube_obs_tracked_fraction=cube_obs_tracked,
            detections_per_s=round(len(self.detection_elapsed) / active, 1) if active > 0 else 0.0,
            rejected_share=round(counter.rejected_share, 3), longest_gap_s=round(float(np.max(gaps)), 2),
            gap_s=round(counter.gap_s, 1),
            tilt_turns=round(counter.tilt_angle / (2 * np.pi), 2),
            cube_offset_mean_mm=round(float(np.mean(offsets)) * 1000, 1),
            cube_offset_max_mm=round(float(np.max(offsets)) * 1000, 1),
            camera_tilt_deg="" if camera_tilt is None else round(camera_tilt, 1),
            camera_moved_deg="" if camera_moved is None else round(camera_moved, 1),
            temp_mean_start_c=round(float(np.nanmean(start)), 1), temp_mean_end_c=round(float(np.nanmean(end)), 1),
            temp_max_start_c=round(float(np.nanmax(start)), 1), temp_max_end_c=round(float(np.nanmax(end)), 1),
            temp_peak_c=round(float(peak[hottest]), 1), hottest_joint=joints[hottest],
            heating_max_c_per_min="" if rates is None else round(float(np.max(rates)) * 60, 1),
            heating_max_joint="" if rates is None else joints[int(np.argmax(rates))],
            temperature_glitches=hand.temp_filter.glitches,
            **tracking_summary(tracking),
            min_voltage_v="" if not np.isfinite(hand.min_voltage) else round(hand.min_voltage, 1),
            servo_errors=errors, note=a.note, run_dir=str(d))
        meta = dict(row, policy_sha256_full=sha256(self.policy_path), host=socket.gethostname(),
                    hand_config_sha256=sha256(HAND_CONFIG), policy_time_segments=[round(s[0], 1) for s in self.segments], drop_times_s=self.drop_times,
                    start_temps_c=dict(zip(joints, np.round(start, 1).tolist())),
                    end_temps_c=dict(zip(joints, np.round(end, 1).tolist())),
                    heating_c_per_s=None if rates is None else dict(zip(joints, np.round(rates, 4).tolist())),
                    hand_T_camera=None if self.hand_T_camera is None else np.round(self.hand_T_camera, 5).tolist(),
                    camera=camera_info, options={k: str(v) for k, v in vars(a).items()})
        (d / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
        return row


def transform(position, rotation):
    """4x4 from a position and a scipy Rotation."""
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = rotation.as_matrix(), position
    return T


def camera_in_hand(args):
    """hand_T_camera (4x4) for policies with the cube pose, or None without --calibration. With --mock: a camera that
    looks down at the palm and sees the drawn cube resting where the task commands it, (0, 0, -0.05)."""
    from cube_spin_policy import OBS_T_HAND, hand_T_camera
    if args.calibration is not None:
        return hand_T_camera(np.load(args.calibration), args.hand_to_tag)
    if args.mock:
        obs_T_camera = np.diag([1.0, -1.0, -1.0, 1.0])
        obs_T_camera[:3, 3] = np.array([0.0, 0.0, -0.05]) - obs_T_camera[:3, :3] @ vision.MockCamera.HOME
        return np.linalg.inv(OBS_T_HAND) @ obs_T_camera
    return None


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_commit():
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True,
                                timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO, capture_output=True,
                               text=True, timeout=5).stdout.strip()
        return commit + ("+changes" if dirty else "")
    except (OSError, subprocess.TimeoutExpired):
        return ""


def append_row(path, row):
    """Append to a CSV; columns that an older file does not have are added to it."""
    path = Path(path)
    fields = list(row)
    if path.exists():
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        old = list(rows[0]) if rows else []
        if set(fields) - set(old) and rows:
            fields = old + [k for k in fields if k not in old]
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields, restval="")
                w.writeheader()
                w.writerows(rows)
        elif old:
            fields = old
    new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="", extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("policies", nargs="*", type=Path, help="Exported policy files (.pt) or model names in --models-dir "
                   "(walrus = models_proprio/walrus_policy.pt), run in this order. The model name is the file name "
                   "without .pt and _policy. None: the untested models, as --auto.")
    p.add_argument("--speed", type=float, help="Commanded spin speed for every policy (rad/s; the sign is the "
                   "direction). Default: each model's goal speed on W&B (the speed of its runs there, e.g. 1.5 for "
                   f"ibex), else {wandb_results.DEFAULT_SPEED}. A run counts in the leaderboard only at that speed.")
    p.add_argument("--duration", type=float, default=60.0, help="Seconds of policy time per run.")
    p.add_argument("--out", default=str(REPO / "logs/real_eval"), help="Results folder.")
    p.add_argument("--note", default="", help="Free text saved with every run (cube, setup, what changed).")
    p.add_argument("--start-wrist-deg", type=float, help="Wrist angle of the start pose (default: flexed, 2 deg "
                   "before the end of its range in config.yaml).")
    p.add_argument("--temp-limit", type=float, default=65.0, help="Pause the run when a motor reaches this (deg C).")
    p.add_argument("--temp-margin", type=float, default=2.0, help="Start only if every motor is expected to stay "
                   "this far below the limit.")
    p.add_argument("--drop-dist", type=float, default=0.06, help="Cube this far from where it rested = fallen (m, "
                   "across the camera's view).")
    p.add_argument("--drop-confirm", type=float, default=0.5, help="... for this long (s).")
    p.add_argument("--lost-timeout", type=float, default=3.0, help="Cube not detected for this long = fallen (s).")
    p.add_argument("--home-radius", type=float, default=0.04, help="After a drop the cube must lie this close to "
                   "where it rested before (m); space overrides.")
    p.add_argument("--max-drops", type=int, default=2, help="End the run after more drops than this.")
    p.add_argument("--cube-edge", type=float, default=0.045, help="Edge length of the cube (m).")
    p.add_argument("--exposure-us", type=int, default=5000, help="Camera exposure time; 0 = automatic (blurs a "
                   "turning cube).")
    p.add_argument("--iso", type=int, default=400, help="Camera sensitivity with a fixed exposure (100-1600).")
    p.add_argument("--camera-id", default="", help="OAK ID to use; default: the first one found.")
    p.add_argument("--calibration", type=Path, help="For policies with the cube pose: the camera calibration cam_T_tag "
                   "(.npy from apriltag_calibration.py), made with the camera where it is now. Policies without the "
                   "cube pose do not need it.")
    p.add_argument("--hand-to-tag", type=float, nargs=6, default=[0.0, 0.05, 0.055, 90.0, 0.0, 180.0],
                   metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"), help="With --calibration: the tag's pose measured "
                   "from the mount (m, deg), the hand_to_tag_* arguments of cube_spin_v2.launch.py. Default: the "
                   "launch file's, measured for the ORCA v1b mount; measure them for your setup.")
    p.add_argument("--cube-pose-timeout", type=float, default=0.25, help="A policy with the cube pose gets an older "
                   "pose with the flag 0 (lost); = timeout_s of CUBE_TRACKER in faive_lab (s).")
    p.add_argument("--detector", type=Path, default=DEFAULT_CKPT, help="Checkpoint of the cube detector.")
    p.add_argument("--device", default="auto", help="Detector device: auto (cuda, then mps, then cpu), cuda, mps, cpu.")
    p.add_argument("--no-window", action="store_true", help="No live window (then there are no keys either).")
    p.add_argument("--no-hand-launch", action="store_true", help="Use a hand controller that already runs.")
    p.add_argument("--pose-only", action="store_true", help="Move to the start pose, switch the torque off and exit.")
    p.add_argument("--camera-only", action="store_true", help="Only show the camera view; the hand is not started.")
    p.add_argument("--wandb", action="store_true", help="Upload every finished run to W&B (rwr-skin/real_hand_eval, "
                   "the format of the other real-hand evaluation) in the background. Needs `wandb login`.")
    p.add_argument("--auto", action="store_true", help="Instead of policy files (the default without them): list the "
                   "models in --models-dir (*_policy.pt) that have no run on W&B yet (no run of the same file or the "
                   "same model name), then after Enter evaluate them in alphabetical order. Implies --wandb.")
    p.add_argument("--dry-run", action="store_true", help="With --auto: only list, start nothing.")
    p.add_argument("--models-dir", type=Path, default=REPO / "models_proprio", help="Where --auto and model names look "
                   "for models (download_policies.py fills it from W&B).")
    p.add_argument("--upload", nargs="+", type=Path, metavar="RUN_DIR", help="Only upload these saved run folders to "
                   "W&B (e.g. after a failed upload); runs already there are skipped.")
    p.add_argument("--wandb-video", action=argparse.BooleanOptionalAction, default=True,
                   help="Also upload video.mp4 (about 25 MB per run); with --upload, also to runs already on W&B "
                   "without it.")
    p.add_argument("--wandb-entity", default=wandb_results.ENTITY, help=argparse.SUPPRESS)
    p.add_argument("--wandb-project", default=wandb_results.PROJECT, help=argparse.SUPPRESS)
    p.add_argument("--mock", action="store_true", help="No hardware: a drawn cube and a fake hand.")
    p.add_argument("--mock-drop-after", type=float, help="With --mock: the cube falls once after this many seconds.")
    p.add_argument("--mock-heat-rate", type=float, default=0.1, help="With --mock: heating of the fake motors.")
    args = p.parse_args(argv)
    if args.auto and args.policies:
        p.error("give policy files or --auto, not both")
    args.auto = args.auto or not (args.policies or args.pose_only or args.camera_only or args.upload)
    if args.dry_run and not args.auto:
        p.error("--dry-run goes with --auto")
    args.wandb = args.wandb or args.auto
    args.policies = [model_file(f, args.models_dir) for f in args.policies]
    missing = [str(d) for d in args.upload or [] if not (d / "meta.json").is_file()]
    if missing:
        p.error(f"not run folders of this script (no meta.json): {', '.join(missing)}")
    missing = [str(f) for f in args.policies if not f.is_file()]
    if missing:
        p.error(f"not found: {', '.join(missing)}")
    if args.calibration is not None:
        args.calibration = args.calibration.expanduser()
        if not args.calibration.is_file():
            p.error(f"--calibration: {args.calibration} not found")
        if np.load(args.calibration).shape != (4, 4):
            p.error(f"--calibration: {args.calibration} is not a 4x4 cam_T_tag")
    return args


def model_file(path, models_dir):
    """A policy argument: a file, else a model name or file name in models_dir (walrus, walrus_policy.pt)."""
    if path.is_file() or len(path.parts) > 1:
        return path
    for candidate in (models_dir / path.name, models_dir / f"{path.name}_policy.pt"):
        if candidate.is_file():
            return candidate
    return path


def not_deployable(policy_path, calibrated):
    """Why the evaluation cannot run a policy file ("" = it can): it feeds the observations of the proprioceptive task,
    or of the extero task if calibrated (the camera's pose relative to the hand is known), and expects a target per
    joint (cube_spin_policy.py). Teachers and history students on W&B take others."""
    import torch
    from cube_spin_policy import NUM_JOINTS, OBS_DIM, OBS_DIM_EXTERO, observation_size
    try:
        policy = torch.jit.load(str(policy_path), map_location="cpu").eval()
        size = observation_size(policy)
        if size is None:
            return f"takes neither the {OBS_DIM} proprioceptive observations nor the {OBS_DIM_EXTERO} with the cube pose"
        if size == OBS_DIM_EXTERO and not calibrated:
            return "takes the cube pose: give --calibration and --hand-to-tag"
        with torch.no_grad():
            out = policy(torch.zeros(1, size))
    except Exception as e:
        return f"cannot be run ({type(e).__name__}: {e})"
    if out.shape[-1] != NUM_JOINTS:
        return f"outputs {out.shape[-1]} actions, not {NUM_JOINTS}"
    return ""


def _terminate(*_):
    raise KeyboardInterrupt


def wandb_records(args):
    """All runs in the W&B project (as wandb_results.Project.records()); exits if W&B cannot be reached."""
    try:
        project = wandb_results.Project(args.wandb_entity, args.wandb_project)
        return project.records()
    except Exception as e:  # not installed, not logged in, offline
        raise SystemExit(f"W&B is not reachable ({type(e).__name__}: {e}).\nLog in once with `wandb login` (see "
                         "the README's installation)" + ("." if args.auto else ", or give --speed to run without W&B."))


def untested_policies(args, known):
    """--auto: the models in --models-dir without a run on W&B, after listing them all."""
    if not any(Path(args.models_dir).glob("*_policy.pt")):
        raise SystemExit(f"No models (*_policy.pt) in {args.models_dir}. Download them from W&B first:\n"
                         "  python experiments/rl_inference/download_policies.py")
    todo, done = wandb_results.split_models(args.models_dir, known)
    print(f"Models in {args.models_dir} against W&B {args.wandb_entity}/{args.wandb_project}:")
    for path, why in done:
        print(f"  tested    {path.name:28s} {why}")
    calibrated = args.calibration is not None or args.mock
    unusable = {path: why for path in todo if (why := not_deployable(path, calibrated))}
    for path in todo:
        print(f"  untested  {path.name:28s} {'not evaluated: ' + unusable[path] if path in unusable else ''}".rstrip())
    todo = [path for path in todo if path not in unusable]
    print(f"{len(todo)} to evaluate" + (f": {', '.join(wandb_results.model_name(t) for t in todo)}" if todo else ""))
    return todo


def main(argv=None):
    args = parse_args(argv)
    if args.upload:
        try:
            wandb_results.upload_runs(args.upload, video=args.wandb_video, entity=args.wandb_entity,
                                      project=args.wandb_project)
        except Exception as e:
            print(f"W&B upload failed: {type(e).__name__}: {e}")
            return 1
        return 0
    known = {}
    if args.auto or (args.policies and args.speed is None):
        known = wandb_results.known_models(wandb_records(args))
    if args.auto:
        args.policies = untested_policies(args, known)
    calibrated = args.calibration is not None or args.mock
    unusable = [f"{p}: {why}" for p in args.policies if (why := not_deployable(p, calibrated))]
    if unusable:
        raise SystemExit("Cannot evaluate:\n  " + "\n  ".join(unusable))
    speeds = {}
    for policy in args.policies:
        speeds[policy], source = wandb_results.goal_speed(known, policy, args.speed)
        print(f"  {wandb_results.model_name(policy)}: {speeds[policy]:+.1f} rad/s ({source})")
    if args.dry_run or (args.auto and not args.policies):
        return 0
    if args.auto:
        try:
            input(f"Enter: evaluate {'it' if len(args.policies) == 1 else f'these {len(args.policies)}'} on the hand "
                  "and upload the runs (Ctrl+C: cancel) ")
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.")
            return 0
    Path(args.out).mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, _terminate)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, _terminate)  # the terminal was closed
    ev = Evaluation(args)
    code = 0
    try:
        if not args.pose_only:
            ev.start_vision()
        if args.camera_only:
            ev.show("camera only, the hand is not used", "white", "q = quit")
            while True:
                ev.tick()
                ev.key()
                ev.print_cube_pose()
        ev.start_hand()
        if args.pose_only:
            ev.park()
            print(f"The hand is in the start pose (wrist {ev.start_pose[ev.joint_ids.index('wrist')]:+.0f} deg), "
                  "torque off.")
            return 0
        for policy in args.policies:
            ev.evaluate(policy, speeds[policy])
        print(f"\nDone. Results: {Path(args.out) / 'results.csv'}")
    except (Quit, Skip):
        print("\nStopped (q).")
    except KeyboardInterrupt:
        print("\nStopped (Ctrl+C).")
    except RuntimeError as e:
        print(f"\nError: {e}")
        code = 1
    finally:
        ev.shutdown()
        ev.wait_for_uploads()
    return code


if __name__ == "__main__":
    sys.exit(main())
