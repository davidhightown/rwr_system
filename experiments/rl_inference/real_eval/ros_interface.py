"""Everything the evaluation needs from ROS: camera image and intrinsics, cube pose (TF from the cube detector), motor
temperatures, joint positions, the torque service and joint commands of the hand controller, and the live view topic."""

import threading
import time
from collections import deque

import numpy as np
import tf2_ros
from cv_bridge import CvBridge
from ingress.msg import CameraIntrinsics
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import SetBool

TICK_HZ = 20.0


class RosInterface(Node):
    def __init__(self, camera_topic="/oakd_bottom_view/color", camera_frame="oakd_bottom_view_frame",
                 cube_frame="cube", hand="right"):
        super().__init__("real_policy_evaluation")
        self.lock = threading.Lock()
        self.bridge = CvBridge()
        self.camera_frame, self.cube_frame = camera_frame, cube_frame
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.image, self.image_time = None, 0.0
        self.K, self.dist = None, None
        self.temps, self.temps_time = None, 0.0
        self.joint_pos, self.joint_time = None, 0.0
        self.command = None  # latest joint command of the policy (rad, the hand controller's joint order)
        self.joint_log = None  # list of (time, command, measured) while recording
        self.pose = None  # (time, position, rotation) of the latest detection
        self._pose_stamp = None
        self.detection_times = deque(maxlen=200)
        self.on_detection = None  # callback(time, position, rotation)
        self.on_tick = None  # callback(now), at TICK_HZ: view and video
        self.policy_messages = 0
        self._own_commands = False
        self.torque_on = None  # unknown until set

        self.create_subscription(Image, camera_topic, self._image_cb, qos_profile_sensor_data)
        self.create_subscription(CameraIntrinsics, camera_topic.replace("color", "intrinsics"), self._intrinsics_cb,
                                 QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                            durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(Float32MultiArray, f"/{hand}/hand/motor_temperature", self._temps_cb, 10)
        self.create_subscription(Float32MultiArray, f"/{hand}/hand/joint_position_from_motor", self._joints_cb, 10)
        self.create_subscription(Float32MultiArray, f"/{hand}/hand/policy_output", self._policy_cb, 10)
        self.command_pub = self.create_publisher(Float32MultiArray, f"/{hand}/hand/policy_output", 10)
        self.view_pub = self.create_publisher(Image, "/real_eval/view", 1)
        self.torque_client = self.create_client(SetBool, f"/{hand}/hand/set_torque_enabled")
        self.create_timer(1.0 / TICK_HZ, self._tick)
        self._executor = None

    # -- life cycle
    def start(self):
        self._executor = MultiThreadedExecutor()
        self._executor.add_node(self)
        threading.Thread(target=self._executor.spin, daemon=True).start()

    # -- callbacks
    def _image_cb(self, msg):
        image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        with self.lock:
            self.image, self.image_time = image, time.time()

    def _intrinsics_cb(self, msg):
        with self.lock:
            self.K = np.array([[msg.fx, 0, msg.cx], [0, msg.fy, msg.cy], [0, 0, 1]], dtype=np.float64)
            self.dist = np.array(msg.disto, dtype=np.float64)

    def _temps_cb(self, msg):
        with self.lock:
            self.temps, self.temps_time = np.array(msg.data, dtype=np.float32), time.time()

    def _joints_cb(self, msg):
        now = time.time()
        q = np.array(msg.data, dtype=np.float64)
        with self.lock:
            self.joint_pos, self.joint_time = q, now
            if self.joint_log is not None and self.command is not None:
                self.joint_log.append((now, self.command, q))

    def _policy_cb(self, msg):
        if not self._own_commands:
            self.policy_messages += 1
            with self.lock:
                self.command = np.array(msg.data, dtype=np.float64)

    def _tick(self):
        now = time.time()
        try:
            tf = self.tf_buffer.lookup_transform(self.camera_frame, self.cube_frame, Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            tf = None
        if tf is not None:
            stamp = (tf.header.stamp.sec, tf.header.stamp.nanosec)
            if stamp != self._pose_stamp:
                self._pose_stamp = stamp
                t, q = tf.transform.translation, tf.transform.rotation
                position = np.array([t.x, t.y, t.z])
                rotation = Rotation.from_quat([q.x, q.y, q.z, q.w])
                with self.lock:
                    self.pose = (now, position, rotation)
                    self.detection_times.append(now)
                if self.on_detection is not None:
                    self.on_detection(now, position, rotation)
        if self.on_tick is not None:
            self.on_tick(now)

    # -- readings
    def hottest(self, max_age=5.0):
        """(temperature, joint index) of the hottest motor, or (None, None) without recent temperatures."""
        with self.lock:
            temps, age = self.temps, time.time() - self.temps_time
        if temps is None or age > max_age:
            return None, None
        return float(np.max(temps)), int(np.argmax(temps))

    def start_joint_log(self):
        with self.lock:
            self.command, self.joint_log = None, []

    def stop_joint_log(self):
        """[(time, command, measured)] at the rate of the joint positions (100 Hz) since start_joint_log()."""
        with self.lock:
            log, self.joint_log = self.joint_log or [], None
        return log

    def ages(self):
        """Seconds since the last camera image, joint position and motor temperature (inf if never)."""
        now = time.time()
        with self.lock:
            return {name: (now - t if t > 0 else float("inf"))
                    for name, t in (("image", self.image_time), ("joints", self.joint_time), ("temps", self.temps_time))}

    def pose_age(self):
        with self.lock:
            return None if self.pose is None else time.time() - self.pose[0]

    def detections_per_s(self, window=2.0):
        now = time.time()
        with self.lock:
            return sum(1 for t in self.detection_times if now - t <= window) / window

    def publish_view(self, frame):
        self.view_pub.publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))

    def wait_for(self, what, check, timeout):
        t0 = time.time()
        if not check():
            print(f"  waiting for {what} (up to {timeout:.0f} s) ...", flush=True)
        while not check():
            if time.time() - t0 > timeout:
                raise RuntimeError(f"no {what} after {timeout:.0f} s")
            time.sleep(0.1)

    # -- hand
    def _command(self, joint_pos_rad):
        self.command_pub.publish(Float32MultiArray(data=np.asarray(joint_pos_rad, dtype=np.float32).tolist()))

    def hold_present_pose(self):
        """Make the measured pose the goal, so that switching the torque on does not jump to an old goal."""
        with self.lock:
            q = None if self.joint_pos is None else self.joint_pos.copy()
        if q is None or np.any(np.isnan(q)):
            return
        self._own_commands = True
        for _ in range(3):
            self._command(q)
            time.sleep(0.05)
        self._own_commands = False
        time.sleep(0.2)

    def set_torque(self, enabled, timeout=5.0):
        if enabled:
            self.hold_present_pose()
        if not self.torque_client.wait_for_service(timeout_sec=timeout):
            raise RuntimeError("no set_torque_enabled service: is the hand controller running?")
        future = self.torque_client.call_async(SetBool.Request(data=enabled))
        t0 = time.time()
        while not future.done():
            if time.time() - t0 > timeout:
                raise RuntimeError("set_torque_enabled did not answer")
            time.sleep(0.05)
        if not future.result().success:
            raise RuntimeError(f"set_torque_enabled failed: {future.result().message}")
        self.torque_on = enabled

    def move_to(self, pose_deg, seconds=1.5):
        """Ramp from the measured pose to pose_deg (the hand controller's joint order)."""
        with self.lock:
            start = None if self.joint_pos is None else self.joint_pos.copy()
        target = np.deg2rad(np.asarray(pose_deg, dtype=float))
        if start is None or start.shape != target.shape or np.any(np.isnan(start)):
            start = target
        self._own_commands = True
        try:
            steps = max(1, int(seconds * TICK_HZ))
            for i in range(1, steps + 1):
                self._command(start + (target - start) * i / steps)
                time.sleep(1.0 / TICK_HZ)
        finally:
            self._own_commands = False

    def park(self, pose_deg):
        """Move to pose_deg and switch the torque off: the hand keeps the pose, holds the light cube and cools down."""
        if self.torque_on is not True:
            self.set_torque(True)
        self.move_to(pose_deg)
        time.sleep(0.5)
        self.set_torque(False)
