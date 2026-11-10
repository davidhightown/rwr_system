#!/usr/bin/env python3
"""Evaluate exported proprioceptive policies on the real hand, with the camera measuring the cube.

One command starts the camera with the cube detector, the hand controller and the live view, then tests the models one
after another, newest first:

    experiments/rl_inference/run_real_evaluation.sh                  # all models in --models-dir
    experiments/rl_inference/run_real_evaluation.sh --models 06 20   # only these
    experiments/rl_inference/run_real_evaluation.sh --live           # camera only: count while you turn the cube by hand

--models-dir holds one folder per model with its policy.pt and a models.json, newest first:
[{"idx": 6, "name": "wombat", "dir": "/path/to/06_wombat", "goal": 1.5}, ...] (goal: the test speed in rad/s).

Between runs the hand rests in the start pose (the neutral pose with the wrist raised, --start-wrist-deg) with the torque
off: the light cube stays in the open palm and the motors cool down. A run

1. starts once every motor is below --start-temp and the cube lies still in the palm,
2. switches the torque on, starts the policy node (without the cube pose) and records the camera video,
3. counts the cube's turns from the detected pose (real_eval/cube_tracking.py),
4. stops after --duration seconds, when the cube falls, when it is not detected for --lost-timeout seconds, when a motor
   reaches --abort-temp, when the camera, the hand controller or the policy node stops, or when you press Enter,
5. moves the hand back to the start pose, switches the torque off and asks whether to save the result. "Save and
   start the next model" starts the next model by itself (step 1), without a second Enter.

A stopped camera or hand controller is restarted before the next run.

Results: <out>/results.csv (one row per run, also the ones not saved), <out>/summary.md (the saved ones),
<models-dir>/all_results.csv (the saved runs of all sessions) and per run a folder with video.mp4 (the live view),
turns_vs_time.png, joints.png, trace.csv (cube poses), joints.csv (commanded and measured joints and motor currents,
100 Hz), joint_tracking.csv (delay, error and current per joint), temperatures.csv, motor_status.csv (supply voltage and
servo status bits), meta.json (model, file hash, options, git commit, start temperatures, result), policy_log.csv and
policy_node.log. <out>/code_changes_<time>.diff holds the uncommitted code changes. --out with an earlier folder
continues it: models with a saved result there are skipped, except the ones named with --models.
"""

import argparse
import csv
import faulthandler
import json
import os
import select
import shutil
import signal
import sys
import termios
import time
from collections import deque
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import rclpy
from rclpy.signals import SignalHandlerOptions
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from real_eval.analysis import (joint_summary, joint_tracking, plot_run, progress_metrics,  # noqa: E402
                                quality_warnings)
from real_eval.cube_tracking import DropMonitor, TurnCounter, resting_position  # noqa: E402
from real_eval.live_view import ViewState, VideoRecorder, render  # noqa: E402
from real_eval.processes import Processes, find_leftovers, stop_pids, wait_or_fail  # noqa: E402
from real_eval.ros_interface import RosInterface  # noqa: E402
from real_eval.run_info import session_info, update_meta_result, write_meta  # noqa: E402

faulthandler.enable()  # a crash in a C library prints the Python stack of every thread

WS = Path(__file__).resolve().parents[4]  # ros_ws
REPO = Path(__file__).resolve().parents[2]  # rwr_system
HAND_CONFIG = REPO / "src/hand_control/orca/configs/orcahand_v2/right/config.yaml"
DETECTOR = Path(__file__).resolve().parent / "pose_estimation/finetuned_models/maskrcnn-finetuned_lightning_new_adamw.ckpt"
CUBE_EDGE = 0.045  # m, as cube_edge_length of the cube detector

# outcomes that say nothing about the policy: the run is repeated
INTERRUPTED = ("not started", "overheated", "aborted", "camera lost", "controller lost", "policy failed")
STALE_S = {"image": 3.0, "joints": 2.0}  # no camera image / joint position for this long = the node stopped


# the status byte of the Feetech servos (orca_core feetech_client.hardware_error_bits)
SERVO_ERROR_BITS = ((0x01, "input_voltage"), (0x02, "angle_sensor"), (0x04, "overheating"), (0x08, "overcurrent"),
                    (0x20, "overload"))


def servo_error_names(byte: int) -> list[str]:
    names = [name for bit, name in SERVO_ERROR_BITS if byte & bit]
    rest = byte & ~sum(bit for bit, _ in SERVO_ERROR_BITS)
    return names + ([f"0x{rest:02X}"] if rest else [])


class RunInterrupted(KeyboardInterrupt):
    """Ctrl+C during a run: the run was finished (hand parked, files written); its result is saved, then we stop."""

    def __init__(self, result):
        super().__init__()
        self.result = result


class Evaluation:
    def __init__(self, args, need_hand=True):
        self.args = args
        self.need_hand = need_hand
        config = yaml.safe_load(open(HAND_CONFIG))
        self.joint_ids = list(config["joint_ids"])
        self.start_pose = np.array([float(config["neutral_position"].get(j, 0.0)) for j in self.joint_ids])
        self.start_pose[self.joint_ids.index("wrist")] = args.start_wrist_deg

        self.ros = RosInterface()
        self.view = ViewState(joint_names=self.joint_ids, start_temp=args.start_temp, abort_temp=args.abort_temp,
                              drop_dist=args.drop_dist)
        self.counter = None  # TurnCounter during a run or a live count
        self.trail = deque(maxlen=60)
        self.recorder = None
        self.ros.on_detection = self._on_detection
        self.ros.on_tick = self._on_tick
        self.processes = None
        self.hand_ready = False
        # during a run: hottest temperature, motor status once per second, servo faults until the hand parks
        self._temp_max = 0.0
        self.motor_status = []  # (time, voltages or None, status bytes or None)
        self._run_errors = None
        self.session = None

    # -- callbacks from the ROS thread
    def _on_detection(self, now, position, rotation):
        self.trail.append(position)
        counter = self.counter
        if counter is not None:
            counter.add(now, position, rotation)

    def _on_tick(self, now):
        with self.ros.lock:
            image, pose, temps = self.ros.image, self.ros.pose, self.ros.temps
            K, dist = self.ros.K, self.ros.dist
        if image is None:
            return
        v = self.view
        v.temps = temps
        v.pose_age = None if pose is None else now - pose[0]
        if pose is not None:
            v.position, v.rotation = pose[1], pose[2]
        v.trail = list(self.trail)
        v.detections_per_s = self.ros.detections_per_s()
        counter = self.counter
        v.turns = None if counter is None else counter.turns_camera_axis
        v.rejected_share = 0.0 if counter is None else counter.rejected / max(1, counter.accepted + counter.rejected)
        frame = render(image, v, K, dist, CUBE_EDGE)
        self.ros.publish_view(frame)
        recorder = self.recorder
        if recorder is not None:
            recorder.write(frame)

    # -- the nodes: start, watch, restart
    def launch(self, out: Path):
        self.processes = Processes(out / "node_logs")
        self.session = session_info(self.args, REPO, HAND_CONFIG, out)
        if self.args.no_launch:
            return
        self._stop_leftovers()
        self._start_camera()
        if not self.args.no_viewer:
            self.processes.start("viewer", ["ros2", "run", "rqt_image_view", "rqt_image_view", "/real_eval/view"])
        if self.need_hand:
            self._start_hand()

    def _stop_leftovers(self):
        """Nodes of an earlier session that still run hold the camera and the motor port: offer to stop them."""
        leftovers = find_leftovers()
        if not leftovers:
            return
        print("These nodes still run (from an earlier session, or started in another terminal):")
        for pid, cmd in sorted(leftovers.items()):
            print(f"  {pid:7d}  {cmd[:150]}")
        if ask("Stop them? (a hand controller started with the evaluation switches the torque off) "
               "[Enter = stop them, q = quit] ", "", "q") == "q":
            raise KeyboardInterrupt
        if not stop_pids(leftovers):
            raise RuntimeError("could not stop the old nodes: " + ", ".join(map(str, find_leftovers())))
        print("Stopped.")

    def _start_camera(self):
        a = self.args
        for attempt in range(1, 4):
            print(f"Starting the camera and the cube detector (attempt {attempt}) ...")
            self.processes.start("camera", ["ros2", "launch", "experiments", "cube_spin_v2.launch.py",
                                            f"model_path:={DETECTOR}", f"exposure_us:={a.exposure_us}", f"iso:={a.iso}"]
                                 + ([f"camera_id:={a.camera_id}"] if a.camera_id else []))
            if wait_or_fail(lambda: self.ros.ages()["image"] < 1.0, 40):
                print("  camera images arrive" + (f" (exposure {a.exposure_us} us, ISO {a.iso or 800})"
                                                   if a.exposure_us > 0 else " (auto exposure)"))
                return
            self.processes.stop("camera")
            if attempt < 3:
                print("  the camera is not ready yet, retrying in 5 s")
                time.sleep(5)
        raise RuntimeError(f"no camera images; see {self.processes.log_dir / 'camera.log'}")

    def _start_hand(self):
        print(f"Starting the hand controller in 5 s: the hand moves to the start pose "
              f"(wrist {self.args.start_wrist_deg:+.0f} deg). Ctrl+C to cancel.")
        time.sleep(5)
        self.processes.start("hand", ["ros2", "launch", "hand_control", "orcahand_v2_controller_launch.yaml",
                                      "hand_chirality:=right", "launch_rviz:=false",
                                      f"neutral_wrist_deg:={self.args.start_wrist_deg}",
                                      "torque_off_on_exit:=true"])  # also when this script crashes

    def prepare_hand(self):
        self.ros.wait_for("joint positions from the hand controller", lambda: self.ros.ages()["joints"] < 1.0, 60)
        self.ros.wait_for("motor temperatures from the hand controller", lambda: self.ros.ages()["temps"] < 3.0, 20)
        self.hand_ready = True
        self.view.phase = "moving to the start pose"
        self.ros.park(self.start_pose)
        self.view.phase = "start pose, torque off"

    def problem(self) -> str:
        """The node that stopped ("camera", "hand controller") or ""."""
        ages = self.ros.ages()
        launched = not self.args.no_launch
        if (launched and not self.processes.alive("camera")) or ages["image"] > STALE_S["image"]:
            return "camera"
        if self.need_hand and ((launched and not self.processes.alive("hand")) or ages["joints"] > STALE_S["joints"]):
            return "hand controller"
        return ""

    def ensure_running(self):
        """Restart a camera or hand controller that stopped."""
        for _ in range(3):
            problem = self.problem()
            if not problem:
                return
            if self.args.no_launch:
                raise RuntimeError(f"the {problem} stopped; restart it (this evaluation runs with --no-launch)")
            print(f"  the {problem} stopped (log in {self.processes.log_dir}); restarting it")
            self.view.phase = f"restarting the {problem}"
            if problem == "camera":
                self.processes.stop("camera")
                self._start_camera()
            else:
                self.hand_ready = False
                self.processes.stop("hand")
                self._start_hand()
                self.prepare_hand()
        raise RuntimeError(f"the {self.problem()} keeps stopping")

    # -- one run
    def wait_until_cool(self):
        """The hottest motor's temperature once it is below --start-temp; None if the temperatures stop arriving."""
        last_print, no_temps_since = 0.0, None
        while True:
            temp, joint = self.ros.hottest()
            if temp is not None and temp < self.args.start_temp:
                return temp
            if temp is None:
                no_temps_since = no_temps_since or time.time()
                if time.time() - no_temps_since > 10.0:
                    return None
            else:
                no_temps_since = None
                self.view.phase = f"cooling down: {self.joint_ids[joint]} {temp:.0f} C, start below {self.args.start_temp:.0f} C"
            if time.time() - last_print > 30:
                last_print = time.time()
                print(f"  {self.view.phase}" if temp is not None else "  waiting for motor temperatures")
            time.sleep(0.1)  # the temperatures arrive once per second: start right after the first one below the limit

    def wait_until_cube_rests(self, timeout=15.0):
        self.view.phase = "waiting for the cube to lie still in the palm"
        t0 = time.time()
        while time.time() - t0 < timeout:
            age = self.ros.pose_age()
            samples = list(self.trail)[-15:] if age is not None and age < 0.3 else []
            rest = resting_position(samples)
            if rest is not None:
                return rest
            time.sleep(0.1)
        return None

    def run(self, model, speed, run_dir: Path) -> dict:
        a = self.args
        ctrl_c = False
        self._temp_max = 0.0
        name = f"{model['idx']:02d}_{model['name']}"
        result = dict(time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), model=name, goal_ang_vel=speed,
                      outcome="not started", stop_reason="", duration_s=0.0, turns=0.0, speed_along_cmd=0.0,
                      turns_30s="", best_10s_rad_per_s="", longest_stall_s="", turns_camera_axis=0.0,
                      spin_axis_tilt_deg=0.0, tracked_fraction=0.0, poses_per_s=0.0, rejected_poses=0.0,
                      longest_gap_s=0.0, dropped=False, median_lag_ms="", worst_joint="", worst_joint_error_deg="",
                      stuck_joints="", total_current_a="", highest_current_joint="", start_temp_max=0.0,
                      run_temp_max=0.0, hottest_joint="", warnings="", valid="", min_voltage_v="", servo_errors="",
                      servo_errors_parking="", temperature_glitches=0, note=a.note, exposure_us=a.exposure_us,
                      run_dir=str(run_dir))
        with self.ros.lock:
            temps = None if self.ros.temps is None else self.ros.temps.copy()
        start_temps = {} if temps is None else {j: round(float(t), 1) for j, t in zip(self.joint_ids, temps)}
        write_meta(run_dir, self.session, model, speed, start_temps)
        result["start_temp_max"] = result["run_temp_max"] = max(start_temps.values(), default=0.0)

        rest = self.wait_until_cube_rests()
        if rest is None:
            result["stop_reason"] = "the cube was not detected lying still in the palm"
            update_meta_result(run_dir, result)
            return result
        self.view.rest_position = rest

        self.recorder = VideoRecorder(run_dir / "video.mp4")
        counter = TurnCounter(up=np.array([0.0, 0.0, a.axis_sign]), max_step_deg=a.max_step_deg,
                              max_tilt_deg=a.max_tilt_deg)
        drops = DropMonitor(rest, a.drop_dist, a.drop_confirm)
        temperatures = []
        self._run_errors, self.motor_status = None, []
        t0 = t_end = time.time()
        self.view.phase = "starting the policy"
        self.ros.set_torque(True)
        self.ros.policy_messages = 0
        self.ros.start_joint_log()
        self.processes.start("policy", ["ros2", "run", "experiments", "policy_inference_cube_spin_v2.py", "--ros-args",
                                        "-p", f"policy_ckpt_path:={model['dir']}/policy.pt", "-p", "use_cube_pose:=false",
                                        "-p", f"goal_ang_vel:={speed}", "-p", "dry_run:=false",
                                        "-p", f"csv_log_path:={run_dir / 'policy_log.csv'}"], log_name=f"policy_{name}")
        if not wait_or_fail(lambda: self.ros.policy_messages > 0 or not self.processes.alive("policy"), 30):
            result["stop_reason"], result["outcome"] = "the policy node sent no command within 30 s", "policy failed"
        elif not self.processes.alive("policy"):
            result["stop_reason"], result["outcome"] = "the policy node exited at the start", "policy failed"
        else:
            flush_stdin()
            print("  running; Enter = stop this run")
            t0 = time.time()
            self.counter = counter
            try:
                self._run_loop(result, counter, drops, temperatures, speed, t0)
            except KeyboardInterrupt:
                result["stop_reason"], result["outcome"], ctrl_c = "Ctrl+C", "aborted", True
                signal.signal(signal.SIGINT, signal.SIG_IGN)  # finish the run (park, files) without a second one
            t_end = time.time()
            self.counter = None
            result.update(self._measure(counter, speed, t0, t_end, self._temp_max))

        joint_log = self.ros.stop_joint_log()
        self.view.phase = f"stopped: {result['stop_reason']}"
        self.processes.stop("policy")
        log = self.processes.log_dir / f"policy_{name}.log"
        if log.exists():
            log.replace(run_dir / "policy_node.log")
        time.sleep(1.0)
        self.recorder.close()
        self.recorder = None
        with self.ros.lock:  # the servo faults of the run (status bytes arrive once per second: after the 1 s above)
            self._run_errors = None if self.ros.status_errors is None else self.ros.status_errors.copy()
        if result["outcome"] != "controller lost":
            self.view.phase = "back to the start pose"
            try:
                self.ros.park(self.start_pose)
                self.view.phase = "start pose, torque off"
            except RuntimeError as e:
                print(f"  could not park the hand: {e}")
        self.view.rest_position = None

        result.update(self._motor_status_summary())
        result["temperature_glitches"] = self.ros.temp_glitches
        run_log = [row for row in joint_log if row[0] >= t0]
        tracking = joint_tracking(run_log, self.joint_ids)
        result.update(joint_summary(tracking))
        if result["outcome"] not in INTERRUPTED:
            result["warnings"] = "; ".join(quality_warnings(result, a.min_tracked, a.max_gap))
        self._write_run_files(run_dir, counter, temperatures, joint_log, t0, tracking)
        try:
            times, progress = counter.progress(counter.result().axis)
            sign = 1.0 if speed >= 0 else -1.0
            plot_run(run_dir, f"{name}: {result['turns']:+.2f} turns in {result['duration_s']} s ({result['outcome']})",
                     times, sign * progress, t0, t_end, speed, temperatures, self.joint_ids, run_log,
                     result["stop_reason"])
        except Exception as e:  # a plot must not stop the evaluation
            print(f"  could not plot the run: {e}")
        update_meta_result(run_dir, result)
        if ctrl_c:
            signal.signal(signal.SIGINT, _interrupt)
            raise RunInterrupted(result)
        return result

    def _run_loop(self, result, counter, drops, temperatures, speed, t0):
        """Watch the run until a stop reason is set in result; the hottest temperature ends in self._temp_max."""
        a = self.args
        self._temp_max, last_temp_log = result["start_temp_max"], 0.0
        while True:
            time.sleep(0.05)
            now = time.time()
            self.view.phase = f"RUNNING {now - t0:4.1f} / {a.duration:.0f} s   goal {speed:+.1f} rad/s   (Enter = stop)"
            temp, joint = self.ros.hottest()
            if temp is not None:
                self._temp_max = max(self._temp_max, temp)
                if now - last_temp_log >= 1.0:
                    last_temp_log = now
                    with self.ros.lock:
                        temperatures.append([now - t0, *self.ros.temps.tolist()])
                        volt, errors = self.ros.voltage, self.ros.status_errors
                        self.motor_status.append((now - t0, None if volt is None else volt.copy(),
                                                  None if errors is None else errors.copy()))
            age = self.ros.pose_age()
            problem = self.problem()
            if problem:
                result["stop_reason"] = f"the {problem} stopped"
                result["outcome"] = "camera lost" if problem == "camera" else "controller lost"
            elif not self.processes.alive("policy"):
                result["stop_reason"], result["outcome"] = "the policy node exited", "policy failed"
            elif enter_pressed():
                result["stop_reason"], result["outcome"] = "stopped by you", "aborted"
            elif temp is not None and temp >= a.abort_temp:
                result["stop_reason"] = f"{self.joint_ids[joint]} reached {temp:.0f} C"
                result["outcome"] = "overheated"
            elif drops.update(now, counter):
                result["stop_reason"], result["dropped"], result["outcome"] = "dropped", True, "dropped"
            elif age is None or age > a.lost_timeout:
                result["stop_reason"] = f"cube not detected for {a.lost_timeout:.0f} s"
                result["outcome"] = "lost"
                result["dropped"] = drops.distance > a.drop_dist / 2
            elif now - t0 >= a.duration:
                result["stop_reason"], result["outcome"] = "time", "completed"
            if result["stop_reason"]:
                break

    def _motor_status_summary(self) -> dict:
        """results.csv fields: lowest supply voltage during the run, servo faults (status bits) per joint during the
        run, and the ones that appeared only while the hand moved back to the start pose afterwards."""
        with self.ros.lock:
            all_errors = None if self.ros.status_errors is None else self.ros.status_errors.copy()
        run_errors = self._run_errors
        volts = [v for _, v, _ in self.motor_status if v is not None]
        out = dict(min_voltage_v=round(float(np.nanmin(volts)), 1) if volts else "")

        def names(errors):
            return "; ".join(f"{j}: {'+'.join(servo_error_names(e))}" for j, e in zip(self.joint_ids, errors) if e)

        if run_errors is not None:
            out["servo_errors"] = names(run_errors)
        if all_errors is not None:
            out["servo_errors_parking"] = names(all_errors & ~run_errors if run_errors is not None else all_errors)
        return out

    def _measure(self, counter, speed, t0, t_end, temp_max) -> dict:
        duration = t_end - t0
        turns = counter.result()
        direction = 1.0 if speed >= 0 else -1.0
        times = [t0] + [d.time for d in counter.detections if d.time >= t0] + [t_end]
        tracked = sum(min(dt, 0.25) for dt in np.diff(times)) / duration
        temps_now = self.ros.temps
        p_times, progress = counter.progress(turns.axis)
        return dict(duration_s=round(duration, 1), turns=round(direction * turns.turns, 2),
                    turns_camera_axis=round(direction * turns.turns_camera_axis, 2),
                    spin_axis_tilt_deg=round(turns.spin_axis_tilt_deg, 1),
                    speed_along_cmd=round(direction * turns.turns * 2 * np.pi / duration, 3),
                    tracked_fraction=round(min(1.0, tracked), 2),
                    poses_per_s=round(len(counter.detections) / duration, 1),
                    rejected_poses=round(turns.rejected_share, 2),
                    longest_gap_s=round(float(np.max(np.diff(times))), 2), run_temp_max=round(temp_max, 1),
                    hottest_joint=self.joint_ids[int(np.argmax(temps_now))] if temps_now is not None else "",
                    **progress_metrics(p_times, direction * progress, t0, t_end))

    def _write_run_files(self, run_dir, counter, temperatures, joint_log, t0, tracking):
        with open(run_dir / "trace.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time", "x", "y", "z", "qx", "qy", "qz", "qw", "accepted", "step_rx", "step_ry", "step_rz"])
            for d in counter.detections:
                w.writerow([f"{d.time:.3f}", *np.round(d.position, 5), *np.round(d.rotation.as_quat(), 5),
                            int(d.accepted), *np.round(d.step, 5)])
        with open(run_dir / "temperatures.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time_s", *self.joint_ids])
            w.writerows([[round(v, 1) for v in row] for row in temperatures])
        with open(run_dir / "motor_status.csv", "w", newline="") as f:  # supply voltage, status bytes since the start
            w = csv.writer(f)
            w.writerow(["time_s", *[f"voltage_v_{j}" for j in self.joint_ids],
                        *[f"errors_{j}" for j in self.joint_ids]])
            n = len(self.joint_ids)
            for t, volt, err in self.motor_status:
                w.writerow([round(t, 1), *(np.round(volt, 1) if volt is not None else [""] * n),
                            *(err.tolist() if err is not None else [""] * n)])
        with open(run_dir / "joints.csv", "w", newline="") as f:  # degrees, time from the start of the run
            w = csv.writer(f)
            w.writerow(["time_s", *[f"cmd_{j}" for j in self.joint_ids], *[f"meas_{j}" for j in self.joint_ids],
                        *[f"current_ma_{j}" for j in self.joint_ids]])
            for t, cmd, meas, *cur in joint_log:
                cur = cur[0] if cur and cur[0] is not None else None
                w.writerow([f"{t - t0:.3f}", *np.round(np.degrees(cmd), 2), *np.round(np.degrees(meas), 2),
                            *(np.round(cur).astype(int) if cur is not None else [""] * len(self.joint_ids))])
        if tracking:
            with open(run_dir / "joint_tracking.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(asdict(tracking[0])))
                w.writeheader()
                w.writerows(asdict(j) for j in tracking)

    # -- modes
    def live_count(self):
        print("Turn the cube by hand: counterclockwise seen from above counts up. Ctrl+C to stop.")
        self.counter = TurnCounter(up=np.array([0.0, 0.0, self.args.axis_sign]), max_step_deg=self.args.max_step_deg,
                                   max_tilt_deg=self.args.max_tilt_deg)
        self.view.phase = "live count, the hand does not move"
        while True:
            time.sleep(0.5)
            r = self.counter.result()
            age = self.ros.pose_age()
            state = "tracked" if age is not None and age <= 0.25 else "NOT DETECTED"
            print(f"\r  turns {r.turns:+6.2f} (camera axis {r.turns_camera_axis:+6.2f}, spin axis tilt "
                  f"{r.spin_axis_tilt_deg:3.0f} deg)  {state:12s} {self.ros.detections_per_s():4.1f} poses/s  "
                  f"rejected {r.rejected_share:.0%}   ", end="", flush=True)

    def evaluate(self, models, out: Path):
        results_path = out / "results.csv"
        try:
            self._evaluate(models, out, results_path)
        finally:  # also after Ctrl+C or an error
            self._summary(results_path)

    def _evaluate(self, models, out: Path, results_path: Path):
        done = set()
        # skip models already saved in this folder, but never the ones named with --models: those run again
        if results_path.exists() and not self.args.models:
            done = {r["model"] for r in csv.DictReader(open(results_path)) if r.get("valid") == "yes"}
        start_next = False  # after "save and next model": no second Enter, the next model starts when ready
        for model in models:
            name = f"{model['idx']:02d}_{model['name']}"
            if name in done:
                print(f"\n=== {name}: already saved in {out.name}, skipped")
                continue
            speed = self.args.speed if self.args.speed is not None else float(model["goal"])
            self.view.title = f"{name}   goal {speed:+.1f} rad/s"
            self.view.phase = "put the cube in the palm, then Enter in the terminal"
            print(f"\n=== {name}, goal {speed:+.2f} rad/s")
            if start_next:
                print("  starts once every motor is below the start temperature and the cube lies in the palm "
                      "(Ctrl+C = stop)")
                choice = ""
            else:
                choice = ask("  Put the cube in the palm. Enter = start, n = next model, q = quit: ", "", "n", "q")
            start_next = False
            while choice == "":
                self.ensure_running()
                temp = self.wait_until_cool()  # starts by itself once every motor is below --start-temp
                if temp is None:  # the temperatures stopped arriving: restart the controller and wait again
                    continue
                self.ensure_running()
                print(f"  hottest motor {temp:.0f} C, starting")
                run_dir = new_run_dir(out, name)
                try:
                    result = self.run(model, speed, run_dir)
                except RunInterrupted as e:
                    self._report(e.result)
                    e.result["valid"] = "no"
                    self._save(results_path, e.result)
                    print(f"  not saved as a result (row in {results_path.name}, files in {run_dir.name})")
                    raise
                self._report(result)
                self.view.phase = "Enter in the terminal: save / run again / next model"
                if result["outcome"] in INTERRUPTED:
                    # not a measurement of the policy: run it again once the motors have cooled down
                    result["valid"] = "no"
                    self._save(results_path, result)
                    choice = ask(f"  Interrupted ({result['stop_reason']}). Put the cube back. Enter = run again "
                                 f"(starts once every motor is below {self.args.start_temp:.0f} C), n = next model, "
                                 f"q = quit: ", "", "n", "q")
                    continue
                choice = ask("  Enter = save and start the next model, r = run again, n = next model without saving, "
                             "q = quit: ", "", "r", "n", "q")
                keep = choice == ""
                if choice == "q":
                    keep = ask("  Save this run before quitting? Enter = save, n = do not save: ", "", "n") == ""
                result["valid"] = "yes" if keep else "no"
                self._save(results_path, result)
                if choice == "r":
                    print("  Put the cube back in the palm; starting again.")
                    choice = ""
                else:
                    start_next = choice == ""
                    choice = "n" if choice == "" else choice
            if choice == "q":
                break

    @staticmethod
    def _report(r):
        print(f"  {r['outcome'].upper()}: {r['stop_reason']} after {r['duration_s']} s: {r['turns']:+.2f} turns "
              f"({r['turns_camera_axis']:+.2f} about the camera axis, spin axis tilted {r['spin_axis_tilt_deg']:.0f} "
              f"deg), tracked {r['tracked_fraction']:.0%}, {r['poses_per_s']} poses/s, longest gap "
              f"{r['longest_gap_s']} s, motors {r['start_temp_max']:.0f} -> {r['run_temp_max']:.0f} C")
        if r["duration_s"] >= 10:
            print(f"  turns after 30 s {r['turns_30s'] if r['turns_30s'] != '' else '-'}, fastest 10 s "
                  f"{r['best_10s_rad_per_s']} rad/s, longest stall {r['longest_stall_s']} s")
        if r["median_lag_ms"] != "":
            print(f"  joints: median delay {r['median_lag_ms']} ms, worst {r['worst_joint']} "
                  f"({r['worst_joint_error_deg']} deg rms)" +
                  (f", barely moving: {r['stuck_joints']}" if r["stuck_joints"] else ""))
        if r["warnings"]:
            print(f"  WARNING, the count may be wrong: {r['warnings']}. Consider r = run again.")

    def _save(self, path, result):
        """The run's row in the session's results.csv; a saved run also goes into all_results.csv of --models-dir."""
        update_meta_result(Path(result["run_dir"]), result)
        append_row(path, result)
        if result.get("valid") == "yes":
            append_row(Path(self.args.models_dir) / "all_results.csv", dict(session=path.parent.name) | result)

    @staticmethod
    def _summary(path):
        if not path.exists():
            return
        rows = [r for r in csv.DictReader(open(path)) if r.get("valid") == "yes"]
        rows.sort(key=lambda r: float(r["turns"]), reverse=True)
        cols = [("model", "model"), ("goal (rad/s)", "goal_ang_vel"), ("turns", "turns"), ("rad/s", "speed_along_cmd"),
                ("turns 30 s", "turns_30s"), ("best 10 s (rad/s)", "best_10s_rad_per_s"),
                ("longest stall (s)", "longest_stall_s"), ("duration (s)", "duration_s"), ("outcome", "outcome"),
                ("tracked", "tracked_fraction"), ("joint delay (ms)", "median_lag_ms"),
                ("current (A)", "total_current_a"), ("barely moving", "stuck_joints"), ("max temp (C)", "run_temp_max"),
                ("min voltage (V)", "min_voltage_v"), ("servo errors", "servo_errors"), ("warnings", "warnings")]
        lines = ["| " + " | ".join(c for c, _ in cols) + " |", "|" + "---|" * len(cols)]
        lines += ["| " + " | ".join(str(r.get(k, "")) for _, k in cols) + " |" for r in rows]
        (path.parent / "summary.md").write_text("\n".join(lines) + "\n")
        print("\n" + "\n".join(lines))

    def shutdown(self):
        # a second Ctrl+C (or the signal reaching both pixi and this process) must not interrupt the clean-up
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        if self.processes is not None:
            self.processes.stop("policy")
        if self.recorder is not None:
            self.recorder.close()
        if self.hand_ready:
            try:  # leave the hand in the start pose with the torque off
                self.view.phase = "back to the start pose"
                self.ros.park(self.start_pose)
                print("The hand rests in the start pose with the torque off.")
            except Exception as e:
                print(f"Could not park the hand ({e}); switch the hand's power off if it keeps holding a pose.")
        if self.processes is not None:
            self.processes.stop_all()


def append_row(path: Path, row: dict):
    """Append a row to a CSV file; new columns (a file of an older version) are added to the header."""
    new = not path.exists()
    fields = list(row) if new else next(csv.reader(open(path)))
    if not new and set(row) - set(fields):  # a file of an older version: add the new columns
        rows = list(csv.DictReader(open(path)))
        fields += [k for k in row if k not in fields]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, restval="")
            w.writeheader()
            w.writerows(rows)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", restval="")
        if new:
            w.writeheader()
        w.writerow(row)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="*", help="Numbers (e.g. 06) or names, run in this order; default: all, newest "
                   "trained first. Named models run even when --out already has a saved result for them (a second "
                   "saved row is added).")
    p.add_argument("--oldest-first", action="store_true")
    p.add_argument("--models-dir", default=os.path.expanduser("~/policies/real_test"))
    p.add_argument("--speed", type=float, help="Commanded speed (rad/s, sign = direction) for every model; default: "
                   "each model's own test speed.")
    p.add_argument("--duration", type=float, default=60.0, help="Seconds per run.")
    p.add_argument("--start-temp", type=float, default=42.0, help="Start a run only when every motor is below this (C).")
    p.add_argument("--abort-temp", type=float, default=58.0, help="Stop a run when a motor reaches this (C).")
    p.add_argument("--start-wrist-deg", type=float, default=30.0, help="Wrist angle of the start pose.")
    p.add_argument("--drop-dist", type=float, default=0.06, help="Cube this far from where it rested = fallen (m).")
    p.add_argument("--drop-confirm", type=float, default=1.0, help="... for this long (s).")
    p.add_argument("--lost-timeout", type=float, default=5.0, help="Stop when the cube is not detected this long (s).")
    p.add_argument("--max-step-deg", type=float, default=40.0, help="Larger turns between two detections are "
                   "treated as wrong detections (above 45 the cube's symmetry makes them ambiguous anyway).")
    p.add_argument("--max-tilt-deg", type=float, default=30.0, help="The same for rotations about the other axes.")
    p.add_argument("--axis-sign", type=float, default=-1.0, help="-1: the camera looks down; +1 flips the direction.")
    p.add_argument("--min-tracked", type=float, default=0.5, help="Warn when less of the run was tracked.")
    p.add_argument("--max-gap", type=float, default=3.0, help="Warn about a longer time without detection (s).")
    p.add_argument("--exposure-us", type=int, default=0, help="Fixed camera exposure time (us), less motion blur; "
                   "0: auto exposure. Try 4000-8000 with --live.")
    p.add_argument("--camera-id", default="", help="OAK-D ID (mxid) to use; default: the one in "
                   "experiments/cfgs/cams/oakd_cube_task.yaml, or the only connected OAK-D.")
    p.add_argument("--iso", type=int, default=0, help="Camera sensitivity (100-1600) with --exposure-us; 0: 800.")
    p.add_argument("--note", default="", help="Free text saved with every run of this session: cube, setup, "
                   "what changed (e.g. \"new 45 mm cube, re-calibrated thumb\").")
    p.add_argument("--out", help="Results folder; default: <models-dir>/eval_<date_time>. An earlier folder continues.")
    p.add_argument("--live", action="store_true", help="Only show the turn count while you turn the cube by hand.")
    p.add_argument("--pose-only", action="store_true", help="Only move the hand to the start pose, then exit.")
    p.add_argument("--no-launch", action="store_true", help="Use a camera and hand controller that already run.")
    p.add_argument("--no-viewer", action="store_true", help="Do not open the live view window.")
    return p.parse_args()


def load_models(args):
    models = json.load(open(Path(args.models_dir) / "models.json"))
    if args.models:
        # in the order given on the command line; "9" and "09" both mean model 09
        by_key = {f"{m['idx']:02d}": m for m in models} | {m["name"]: m for m in models}
        keys = [f"{int(k):02d}" if k.isdigit() else k for k in args.models]
        missing = [k for k in keys if k not in by_key]
        if missing:
            raise SystemExit(f"unknown models: {missing}")
        models = list({id(by_key[k]): by_key[k] for k in keys}.values())  # without repeats
    return models[::-1] if args.oldest_first else models


def new_run_dir(out: Path, name: str) -> Path:
    base = out / f"{name}_{datetime.now():%H%M%S}"
    run_dir, k = base, 1
    while run_dir.exists():
        k += 1
        run_dir = base.with_name(f"{base.name}_{k}")
    run_dir.mkdir(parents=True)
    return run_dir


def ask(prompt, *allowed):
    """input() until the answer is one of the allowed ones (lower case; "" = Enter)."""
    flush_stdin()
    while True:
        answer = input(prompt).strip().lower()
        if answer in allowed:
            return answer
        print(f"  please answer one of: {', '.join(a or 'Enter' for a in allowed)}")


def flush_stdin():
    """Forget keys pressed earlier, e.g. an Enter pressed twice."""
    if sys.stdin.isatty():
        termios.tcflush(sys.stdin, termios.TCIFLUSH)


def enter_pressed() -> bool:
    if not sys.stdin.isatty():
        return False
    ready, _, _ = select.select([sys.stdin], [], [], 0)
    if ready:
        sys.stdin.readline()
        return True
    return False


def _interrupt(*_):
    raise KeyboardInterrupt


def _hangup(*_):
    """The terminal was closed: clean up like Ctrl+C, without writing to the terminal that is gone."""
    devnull = open(os.devnull, "w")
    sys.stdout = sys.stderr = devnull
    raise KeyboardInterrupt


def main():
    args = parse_args()
    models = [] if args.live or args.pose_only else load_models(args)
    out = Path(args.out) if args.out else Path(args.models_dir) / f"eval_{datetime.now():%Y%m%d_%H%M%S}"
    new_folder = not out.exists()
    out.mkdir(parents=True, exist_ok=True)
    failed = False
    signal.signal(signal.SIGTERM, _interrupt)  # kill stops like Ctrl+C
    signal.signal(signal.SIGHUP, _hangup)  # closing the terminal, too

    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)  # rclpy's handlers stop the executor on Ctrl+C:
    # the hand could then not be parked (no answer from the torque service). The handlers here raise
    # KeyboardInterrupt in the main thread, and shutdown() parks the hand while ROS still runs.
    ev = Evaluation(args, need_hand=not args.live)
    ev.ros.start()
    try:
        ev.launch(out)
        ev.ros.wait_for("camera images", lambda: ev.ros.image is not None, 30)
        if args.live:
            ev.live_count()
        ev.prepare_hand()
        if args.pose_only:
            print(f"The hand rests in the start pose (wrist {args.start_wrist_deg:+.0f} deg), torque off.")
            return
        print(f"{len(models)} models, results in {out}")
        ev.evaluate(models, out)
    except KeyboardInterrupt:
        print("\nStopped.")
    except RuntimeError as e:
        failed = True
        print(f"\nError: {e}")
    finally:
        ev.shutdown()
        rclpy.try_shutdown()
        remove_if_empty(out, new_folder, failed)


def remove_if_empty(out: Path, new_folder: bool, failed: bool):
    """A new session folder without any run is deleted, unless an error stopped it (its node logs say why)."""
    if not new_folder or failed or not out.exists() or any(p.is_dir() and p.name[:2].isdigit() for p in out.iterdir()):
        return
    shutil.rmtree(out, ignore_errors=True)
    print(f"No run in this session: removed {out}")


if __name__ == "__main__":
    main()
