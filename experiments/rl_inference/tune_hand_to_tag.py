#!/usr/bin/env python3
"""Find the six --hand-to-tag values by eye: RViz shows the hand and the live cube as eval_policies.py's digital twin
does, and six sliders set the tag's pose from the mount until the virtual cube sits where the real one does.

    python experiments/rl_inference/tune_hand_to_tag.py \
        --calibration experiments/cfgs/calibration/camera_extrinsics_2026-11-13.npy

The script starts the hand controller as eval_policies.py does: THE HAND MOVES to its neutral pose, then to the
evaluation's start pose, and holds it (torque off at the end). RViz shows the measured joint angles. Then put the cube
in the palm; keep the camera and the tag where they were at the calibration. --no-hand: no hand controller (RViz then
shows the start pose, or the angles of a hand controller started elsewhere). RViz also shows the tag (dark square) and
the mount (yellow dot), and the camera window draws the mount's axes (x red, y green, z blue) and the task's goal
position (green) into the image with the current values.

The mount is the centre of the forearm's flat end face: x towards the little finger, y out of the palm, z along the
forearm towards the fingertips. Angles: intrinsic XYZ (roll about x, then pitch, then yaw), deg.

Keys in the camera window: s = save into the .yaml next to the calibration (hand_to_tag), r = back to the start
values, q / Esc = quit. The values are printed as --hand-to-tag arguments at every save and at the end.
"""

import argparse
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # before torch: Mask R-CNN's ops without an MPS kernel

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from cube_eval.detector import DEFAULT_CKPT, CubePoseEstimator  # noqa: E402
from cube_eval.hud import CUBE_EDGES, GREEN, WHITE, YELLOW, project, text  # noqa: E402
from cube_spin_policy import GOAL_POS, MOUNT_IN_HAND, OBS_T_HAND, hand_T_camera  # noqa: E402

HAND_CONFIG = REPO / "src/hand_control/orca/configs/orcahand_v2/right/config.yaml"
HAND_NODE = REPO / "src/hand_control/hand_control_node.py"
LOG_DIR = REPO / "logs/tune_hand_to_tag"
DEFAULT_CALIBRATION = REPO / "experiments/cfgs/calibration/camera_extrinsics_2026-11-13.npy"
WINDOW = "hand-to-tag"
SLIDERS = [("x mm", 500), ("y mm", 500), ("z mm", 500), ("roll deg", 180), ("pitch deg", 180), ("yaw deg", 180)]
CUBE_POSE_TIMEOUT_S = 1.0  # shown as lost (red) after this long without a detection


def start_values(args):
    """--start, else hand_to_tag of the .yaml next to the calibration, else the measurement of 2026-11-13."""
    if args.start is not None:
        return list(args.start)
    if args.calibration.with_suffix(".yaml").exists():
        values = yaml.safe_load(args.calibration.with_suffix(".yaml").read_text()).get("hand_to_tag")
        if values and len(values) == 6:
            return [float(v) for v in values]
    return [0.0, -0.04, 0.423, 60.0, 0.0, 0.0]


def to_sliders(values):
    return [round(v * 1000) if i < 3 else round(v) for i, v in enumerate(values)]


def from_sliders():
    raw = [cv2.getTrackbarPos(name, WINDOW) - offset for name, offset in SLIDERS]
    return [r / 1000 for r in raw[:3]] + [float(r) for r in raw[3:]]


def hand_T_tag(values):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("XYZ", values[3:], degrees=True).as_matrix()
    T[:3, 3] = np.asarray(values[:3]) + MOUNT_IN_HAND
    return T


def arguments(values):
    return "--hand-to-tag " + " ".join(f"{v:g}" for v in [round(v, 3) for v in values[:3]] + values[3:])


def save(yaml_path, values):
    line = f"hand_to_tag: [{', '.join(f'{v:.3f}' if i < 3 else f'{v:.1f}' for i, v in enumerate(values))}]"
    content = yaml_path.read_text() if yaml_path.exists() else ""
    if re.search(r"^hand_to_tag:.*$", content, flags=re.M):
        content = re.sub(r"^hand_to_tag:.*$", line, content, flags=re.M)
    else:
        content += ("" if content.endswith("\n") or not content else "\n") + line + "\n"
    yaml_path.write_text(content)


class Detection:
    """Camera and cube detector in a thread: the newest image and the newest cube pose in the camera frame."""

    def __init__(self, args):
        from cube_eval.vision import OakCamera
        self.detector = CubePoseEstimator(args.detector, args.cube_edge, args.device)
        self.camera = OakCamera(args.exposure_us, args.iso, camera_id=args.camera_id)
        self.K, self.dist = self.camera.K, self.camera.dist
        self.lock = threading.Lock()
        self.image, self.cube, self.cube_time = None, None, None  # cube: (cam_T_cube 4x4, corner pixels)
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while not self.stop.is_set():
            image = self.camera.read()
            if image is None:
                time.sleep(0.005)
                continue
            stamp = time.time()
            with self.lock:
                self.image = image
            result = self.detector.detect(image, self.K, self.dist)
            if result is not None:
                T = np.eye(4)
                T[:3, :3], T[:3, 3] = result[0], result[1]
                with self.lock:
                    self.cube, self.cube_time = (T, result[2]), stamp

    def latest(self):
        with self.lock:
            return (None if self.image is None else self.image.copy()), self.cube, self.cube_time

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2.0)
        self.camera.close()


class Hand:
    """The hand controller (hand_control_node.py, the parameters of eval_policies.py) as a child process, driven over
    ROS on the given node (spun elsewhere): joint angles in, position commands and the torque switch out."""

    def __init__(self, node):
        from std_msgs.msg import Float32MultiArray
        from std_srvs.srv import SetBool
        self.Float32MultiArray, self.SetBool = Float32MultiArray, SetBool
        self.pub = node.create_publisher(Float32MultiArray, "/right/hand/policy_output", 10)
        self.torque_client = node.create_client(SetBool, "/right/hand/set_torque_enabled")
        node.create_subscription(Float32MultiArray, "/right/hand/joint_position_from_motor", self._joints, 10)
        self.q, self.q_time, self.process = None, 0.0, None

    def _joints(self, msg):
        self.q, self.q_time = np.array(msg.data, dtype=np.float64), time.time()

    def start(self):
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        params = LOG_DIR / "hand_params.yaml"
        params.write_text(yaml.safe_dump({"/**": {"ros__parameters": {
            "hand_controller/hand_name": "orca_v2", "hand_controller/hand_chirality": "right",
            "hand_controller/hand_cfg_name": "orcahand_v2", "hand_controller/calibrate": False,
            "hand_controller/compliant_test_mode": False, "hand_controller/torque_off_on_exit": True}}}))
        log = open(LOG_DIR / "hand_controller.log", "a")
        cmd = [sys.executable, str(HAND_NODE), "--ros-args", "-r", "__ns:=/right", "--params-file", str(params)]
        log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n")
        log.flush()
        self.process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        t0 = time.time()
        while time.time() - self.q_time > 1.0:
            if self.process.poll() is not None:
                raise RuntimeError(f"the hand controller stopped; see {LOG_DIR / 'hand_controller.log'}")
            if time.time() - t0 > 90:
                raise RuntimeError("no joint angles from the hand controller after 90 s")
            time.sleep(0.1)

    def set_torque(self, enabled, timeout=5.0):
        if enabled and self.q is not None and np.all(np.isfinite(self.q)):
            self.command(self.q)  # the measured pose as the goal first: no jump to an old goal
            time.sleep(0.1)
        if not self.torque_client.wait_for_service(timeout_sec=timeout):
            raise RuntimeError("no torque service: is the hand controller running?")
        future = self.torque_client.call_async(self.SetBool.Request(data=bool(enabled)))
        t0 = time.time()
        while not future.done():
            if time.time() - t0 > timeout:
                raise RuntimeError("the hand controller did not answer the torque request")
            time.sleep(0.02)

    def command(self, q_rad):
        self.pub.publish(self.Float32MultiArray(data=np.asarray(q_rad, dtype=np.float32).tolist()))

    def move_to(self, pose_rad, seconds=2.0):
        self.set_torque(True)
        start = self.q.copy() if self.q is not None and np.all(np.isfinite(self.q)) else pose_rad
        steps = int(seconds * 20)
        for i in range(1, steps + 1):
            self.command(start + (pose_rad - start) * i / steps)
            time.sleep(0.05)

    def stop(self):
        if self.process is None:
            return
        try:
            if time.time() - self.q_time < 2.0:
                self.set_torque(False)
                print("The torque is off.")
        except Exception as e:
            print(f"Could not switch the torque off ({e}); switch the hand's power off if it holds a pose.")
        for sig, patience in ((signal.SIGINT, 5), (signal.SIGKILL, 2)):
            if self.process.poll() is not None:
                break
            try:
                os.killpg(self.process.pid, sig)
                self.process.wait(timeout=patience)
            except subprocess.TimeoutExpired:
                continue
            except ProcessLookupError:
                break


class TagMarkers:
    """The tag and the mount in RViz, on the twin's marker topic (namespaces of their own)."""

    def __init__(self, node, tag_size):
        from visualization_msgs.msg import Marker, MarkerArray
        from cube_eval.twin import FRAME, MARKER_TOPIC
        self.Marker, self.MarkerArray, self.frame, self.node = Marker, MarkerArray, FRAME, node
        self.tag_size = tag_size
        self.pub = node.create_publisher(MarkerArray, MARKER_TOPIC, 10)

    def publish(self, values):
        M = self.Marker
        stamp = self.node.get_clock().now().to_msg()
        markers = []
        for ns, kind, T, scale, rgba in (
            ("tag", M.CUBE, OBS_T_HAND @ hand_T_tag(values), (self.tag_size, self.tag_size, 0.002), (0.1, 0.1, 0.1, 0.9)),
            ("mount", M.SPHERE, OBS_T_HAND @ hand_T_tag([0, 0, 0, 0, 0, 0]), (0.01,) * 3, (1.0, 0.85, 0.1, 1.0)),
        ):
            m = M(ns=ns, id=0, type=kind, action=M.ADD)
            m.header.frame_id, m.header.stamp = self.frame, stamp
            m.pose.position.x, m.pose.position.y, m.pose.position.z = (float(v) for v in T[:3, 3])
            x, y, z, w = Rotation.from_matrix(T[:3, :3]).as_quat()
            m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z, m.pose.orientation.w = x, y, z, w
            m.scale.x, m.scale.y, m.scale.z = (float(v) for v in scale)
            m.color.r, m.color.g, m.color.b, m.color.a = (float(v) for v in rgba)
            markers.append(m)
        self.pub.publish(self.MarkerArray(markers=markers))


def draw(image, K, dist, cam_T_hand, cube, cube_age, values):
    """The camera image with the detected cube, the mount's axes and the goal position for the current values."""
    if cube is not None:
        corners = cube[1].astype(int)
        color = GREEN if cube_age < CUBE_POSE_TIMEOUT_S else (60, 60, 255)
        for a, b in CUBE_EDGES:
            cv2.line(image, tuple(corners[a]), tuple(corners[b]), color, 2, cv2.LINE_AA)
    mount = cam_T_hand @ hand_T_tag([0, 0, 0, 0, 0, 0])
    axes = [mount[:3, 3] + 0.05 * mount[:3, i] for i in range(3)]
    pixels = project([mount[:3, 3], *axes], K, dist)
    if pixels is not None:
        o = tuple(int(v) for v in pixels[0])
        for p, color in zip(pixels[1:], ((60, 60, 255), (80, 220, 80), (255, 120, 60))):
            cv2.arrowedLine(image, o, tuple(int(v) for v in p), color, 2, cv2.LINE_AA, tipLength=0.15)
        text(image, "mount", (o[0] + 8, o[1] - 8), YELLOW)
    goal = cam_T_hand @ np.linalg.inv(OBS_T_HAND) @ np.array([*GOAL_POS, 1.0])
    pixel = project(goal[:3], K, dist)
    if pixel is not None:
        cv2.circle(image, tuple(int(v) for v in pixel[0]), 6, GREEN, 2, cv2.LINE_AA)
        text(image, "goal", (int(pixel[0][0]) + 9, int(pixel[0][1]) + 5), GREEN)
    status = "no cube yet" if cube is None else f"cube seen {cube_age:.1f} s ago"
    text(image, arguments(values), (10, 25), WHITE, 0.6)
    text(image, f"{status}    s = save, r = reset, q = quit", (10, 50), WHITE, 0.5)
    return image


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION, help="cam_T_tag of apriltag_calibration.py.")
    p.add_argument("--start", type=float, nargs=6, metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"),
                   help="Start values (m, deg). Default: hand_to_tag of the .yaml next to the calibration.")
    p.add_argument("--tag-size", type=float, default=0.08, help="Black square of the tag (m), for RViz only.")
    p.add_argument("--cube-edge", type=float, default=0.045)
    p.add_argument("--camera-id", default="")
    p.add_argument("--exposure-us", type=int, default=5000)
    p.add_argument("--iso", type=int, default=400)
    p.add_argument("--detector", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--device", default="auto")
    p.add_argument("--no-hand", action="store_true", help="Do not start the hand controller.")
    p.add_argument("--start-wrist-deg", type=float, help="Wrist angle of the start pose; default: as eval_policies.py.")
    args = p.parse_args()
    args.calibration = args.calibration.expanduser()
    cam_T_tag = np.load(args.calibration)
    yaml_path = args.calibration.with_suffix(".yaml")
    values = start_values(args)

    config = yaml.safe_load(HAND_CONFIG.read_text())
    joint_ids = list(config["joint_ids"])
    start_pose = np.deg2rad([float(config["neutral_position"][j]) for j in joint_ids])
    wrist = args.start_wrist_deg if args.start_wrist_deg is not None else config["joint_roms"]["wrist"][1] - 2.0
    start_pose[joint_ids.index("wrist")] = np.deg2rad(wrist)

    import rclpy
    from rclpy.signals import SignalHandlerOptions
    from cube_eval.twin import Twin
    detection = twin = hand = None
    window = False
    try:
        print("Starting the camera and the cube detector ...")
        detection = Detection(args)
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        twin = Twin(joint_ids, start_pose, args.cube_edge, CUBE_POSE_TIMEOUT_S, LOG_DIR)
        tags = TagMarkers(twin.node, args.tag_size)
        twin.start()
        if not args.no_hand:
            print("Starting the hand controller: THE HAND MOVES to its neutral pose, then to the start pose.")
            hand = Hand(twin.node)
            hand.start()
            hand.move_to(start_pose)
            print(f"The hand holds the start pose (wrist {wrist:+.0f} deg). Put the cube in the palm.")

        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, 960, 760)
        for (name, offset), v in zip(SLIDERS, to_sliders(values)):
            cv2.createTrackbar(name, WINDOW, int(np.clip(v + offset, 0, 2 * offset)), 2 * offset, lambda _: None)
        window = True
        print(f"RViz and the window '{WINDOW}' are open. Start: {arguments(values)}")
        while True:
            current = from_sliders()
            cam_T_hand = np.linalg.inv(hand_T_camera(cam_T_tag, current))
            image, cube, cube_time = detection.latest()
            if cube is not None:
                twin.cube_detection(cube_time, np.linalg.inv(cam_T_hand) @ cube[0])
            tags.publish(current)
            if image is not None:
                age = float("inf") if cube_time is None else time.time() - cube_time
                cv2.imshow(WINDOW, draw(image, detection.K, detection.dist, cam_T_hand, cube, age, current))
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27) or twin.stopped():
                break
            if hand is not None and hand.process.poll() is not None:
                print(f"The hand controller stopped; see {LOG_DIR / 'hand_controller.log'}")
                break
            if key == ord("s"):
                save(yaml_path, current)
                print(f"Saved to {yaml_path}: {arguments(current)}")
            if key == ord("r"):
                for (name, offset), v in zip(SLIDERS, to_sliders(values)):
                    cv2.setTrackbarPos(name, WINDOW, int(np.clip(v + offset, 0, 2 * offset)))
    except KeyboardInterrupt:
        pass
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)  # a second Ctrl+C must not interrupt the clean-up
        current = from_sliders() if window else None
        cv2.destroyAllWindows()
        if hand is not None:
            hand.stop()
        if detection is not None:
            detection.close()
        if twin is not None:
            twin.stop()
        rclpy.try_shutdown()
        if current is not None:
            print(f"Last values: {arguments(current)}")


if __name__ == "__main__":
    main()
