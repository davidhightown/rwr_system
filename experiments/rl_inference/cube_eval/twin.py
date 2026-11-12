"""The digital twin of eval_policies.py: an RViz window with the hand model in its measured pose and the cube where a
policy with the cube pose sees it, in the frame the simulation observes the cube in (sim_world: +z up, the task's goal
at cube_spin_policy.GOAL_POS).

The twin starts robot_state_publisher with the ORCA v2 URDF of the viz package and rviz2 (cfgs/rviz/eval_twin.rviz) as
child processes, and its own ROS node publishes:
- /right/hand/joint_states: the angles of /right/hand/joint_position_from_motor (the URDF's joints are the hand
  driver's joint_ids), or the start pose while no hand controller sends them (--camera-only),
- TF sim_world -> hand_root (cube_spin_policy.OBS_T_HAND),
- /cube_eval/twin (MarkerArray): the cube as cube_spin_policy.CubePose gives it to the policy (textured while tracked,
  red and transparent while lost, when the policy gets the last pose with the flag 0), the goal position and a status
  text.
With panel_q, the same state also goes to twin_render, which draws the twin into the evaluation's video.
"""

import os
import queue
import signal
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

RVIZ_CONFIG = Path(__file__).resolve().parents[2] / "cfgs/rviz/eval_twin.rviz"
FRAME = "sim_world"
MARKER_TOPIC = "/cube_eval/twin"
CUBE_MESH = "package://viz/models/dexcube/dex_cube_instanceable.dae"
CUBE_MESH_EDGE = 0.06  # edge length of the mesh (m)
RATE_HZ = 30.0
JOINTS_TIMEOUT_S = 1.0


def hand_urdf():
    import xacro
    from ament_index_python.packages import get_package_share_directory
    xacro_path = Path(get_package_share_directory("viz")) / "models/orcahand_v2/orcahand_v2.urdf.xacro"
    return xacro.process_file(str(xacro_path), mappings={"hand_chirality": "right"}).toxml()


class Twin:
    """rclpy must be initialised. joint_ids: the hand driver's joint order; start_pose_rad: shown without joint angles."""

    def __init__(self, joint_ids, start_pose_rad, cube_edge, cube_pose_timeout, log_dir, panel_q=None):
        from cube_spin_policy import OBS_T_HAND, CubePose
        from geometry_msgs.msg import TransformStamped
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Float32MultiArray
        from tf2_ros import StaticTransformBroadcaster
        from visualization_msgs.msg import Marker, MarkerArray

        self._Marker, self._MarkerArray = Marker, MarkerArray
        self.joint_ids = list(joint_ids)
        self.start_pose = np.asarray(start_pose_rad, dtype=np.float64)
        self.cube_edge = float(cube_edge)
        self.lock = threading.Lock()
        self.cube = CubePose(cube_pose_timeout)
        self.q, self.q_time = None, 0.0
        self.log_dir = Path(log_dir)
        self.panel_q = panel_q  # the state for twin_render (the twin in the video), or None
        self.processes = []

        self.node = Node("cube_eval_twin")
        self.node.create_subscription(Float32MultiArray, "/right/hand/joint_position_from_motor", self._joints, 10)
        self.joint_pub = self.node.create_publisher(JointState, "/right/hand/joint_states", 10)
        self.marker_pub = self.node.create_publisher(MarkerArray, MARKER_TOPIC, 10)
        self.joint_msg = JointState(name=self.joint_ids)

        tf = TransformStamped()
        tf.header.stamp = self.node.get_clock().now().to_msg()
        tf.header.frame_id, tf.child_frame_id = FRAME, "hand_root"
        tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = OBS_T_HAND[:3, 3].tolist()
        x, y, z, w = Rotation.from_matrix(OBS_T_HAND[:3, :3]).as_quat()
        tf.transform.rotation.x, tf.transform.rotation.y, tf.transform.rotation.z, tf.transform.rotation.w = x, y, z, w
        self.static_tf = StaticTransformBroadcaster(self.node)
        self.static_tf.sendTransform(tf)

        self.node.create_timer(1.0 / RATE_HZ, self._publish)
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        threading.Thread(target=self.executor.spin, daemon=True).start()

    def start(self):
        """Start robot_state_publisher and RViz; their output goes to twin.log in log_dir."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        params = self.log_dir / "twin_params.yaml"
        params.write_text(yaml.safe_dump({"/**": {"ros__parameters": {"robot_description": hand_urdf()}}}))
        log = open(self.log_dir / "twin.log", "a")
        for cmd in (
            ["ros2", "run", "robot_state_publisher", "robot_state_publisher", "--ros-args",
             "-r", "__ns:=/right/hand", "-r", "__node:=twin_state_publisher", "--params-file", str(params)],
            ["ros2", "run", "rviz2", "rviz2", "-d", str(RVIZ_CONFIG), "--ros-args", "-r", "__node:=twin_rviz"],
        ):
            log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n")
            log.flush()
            # a session of its own: Ctrl+C in the terminal reaches only eval_policies.py, which then stops it
            self.processes.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True))

    def stopped(self):
        """The command of a child process that exited, or ""."""
        for p in self.processes:
            if p.poll() is not None:
                return " ".join(p.args[:4])
        return ""

    def stop(self):
        for p in self.processes:
            for sig, patience in ((signal.SIGINT, 3), (signal.SIGKILL, 2)):
                if p.poll() is not None:
                    break
                try:
                    os.killpg(p.pid, sig)
                    p.wait(timeout=patience)
                except subprocess.TimeoutExpired:
                    continue
                except ProcessLookupError:
                    break
        self.executor.shutdown(timeout_sec=1.0)
        self.node.destroy_node()

    def cube_detection(self, stamp, hand_T_cube):
        with self.lock:
            self.cube.update(stamp, hand_T_cube)

    # -- the executor thread
    def _joints(self, msg):
        q = np.array(msg.data, dtype=np.float64)
        if len(q) == len(self.joint_ids):
            with self.lock:
                self.q, self.q_time = q, time.time()

    def _publish(self):
        now = time.time()
        with self.lock:
            measured = self.q is not None and now - self.q_time < JOINTS_TIMEOUT_S
            q = self.q.copy() if measured else self.start_pose.copy()
            pos = None if self.cube.pos is None else self.cube.pos.copy()
            quat = None if self.cube.quat is None else self.cube.quat.copy()
            age, timeout = self.cube.age(now), self.cube.timeout_s
        q = np.where(np.isfinite(q), q, self.start_pose)
        tracked = age <= timeout
        lines = self._status_lines(pos is not None, tracked, age, measured)
        self.joint_msg.header.stamp = self.node.get_clock().now().to_msg()
        self.joint_msg.position = q.tolist()
        self.joint_pub.publish(self.joint_msg)
        self.marker_pub.publish(self._markers(pos, quat, tracked, lines))
        if self.panel_q is not None:
            try:
                self.panel_q.get_nowait()
            except queue.Empty:
                pass
            try:
                self.panel_q.put_nowait((now, q, pos, quat, tracked, lines))
            except queue.Full:
                pass

    @staticmethod
    def _status_lines(have_cube, tracked, age, measured):
        lines = ["cube tracked" if tracked else f"cube lost {age:.1f} s: last pose, flag 0" if have_cube
                 else "no cube pose yet"]
        if not measured:
            lines.append("no joint angles: start pose shown")
        return lines

    def _markers(self, pos, quat, tracked, lines):
        from cube_spin_policy import GOAL_POS
        M = self._Marker
        stamp = self.node.get_clock().now().to_msg()

        def marker(ns, kind, position, scale, rgba):
            m = M(ns=ns, id=0, type=kind, action=M.ADD)
            m.header.frame_id, m.header.stamp = FRAME, stamp
            m.pose.position.x, m.pose.position.y, m.pose.position.z = (float(v) for v in position)
            m.pose.orientation.w = 1.0
            m.scale.x, m.scale.y, m.scale.z = (float(v) for v in scale)
            m.color.r, m.color.g, m.color.b, m.color.a = (float(v) for v in rgba)
            return m

        markers = [marker("goal", M.SPHERE, GOAL_POS, [0.008] * 3, (0.2, 0.9, 0.3, 0.8))]
        if pos is None:
            markers.append(M(ns="cube", id=0, action=M.DELETE))
        else:
            s = self.cube_edge / CUBE_MESH_EDGE
            # embedded materials with a colour of all zeros: the mesh's own texture
            cube = marker("cube", M.MESH_RESOURCE, pos, [s] * 3, (0, 0, 0, 0) if tracked else (1.0, 0.2, 0.2, 0.45))
            w, x, y, z = (float(v) for v in quat)
            cube.pose.orientation.x, cube.pose.orientation.y, cube.pose.orientation.z, cube.pose.orientation.w = x, y, z, w
            cube.mesh_resource, cube.mesh_use_embedded_materials = CUBE_MESH, tracked
            markers.append(cube)
        text = marker("status", M.TEXT_VIEW_FACING, np.add(GOAL_POS, (0.0, 0.0, 0.09)), (0, 0, 0.01), (1, 1, 1, 1))
        text.text = "\n".join(lines)
        markers.append(text)
        return self._MarkerArray(markers=markers)
