#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Policy Inference Node for the ORCA v2 hand - runs an rsl_rl policy trained in faive_lab on the real hand. The number of
inputs of the policy selects the observation (cube_spin_policy.py):
    74  proprioception only (orca-v2-inhand-repose-v0)
    82  + the tracked cube pose; velocity command about x, y or z (orca-v2-inhand-all-axes-vel-extero-v0, lemur)
    84  + the tracked cube pose; goal orientations as in DeXtreme (orca-v2-inhand-goal-v0, manta)

The policy is the TorchScript file that faive_lab's play.py exports (logs/rsl_rl/orca_v2_repose/<run>/exported/policy.pt);
it includes the observation normalisation, so it takes the raw observation vector.

Every 1/policy_hz seconds:
1. Read the hand's joint positions and velocities (radians, in the order of the hand's config.yaml joint_ids)
2. Build the policy observation exactly as in the simulation (USD joint order):
       joint_pos       joint positions normalised to [-1, 1] over the joint limits
       joint_vel       joint velocities * 0.2
       cube_pose       position (3), orientation w, x, y, z (4), tracked flag (1); 82 and 84 inputs
       goal_command    [0, 0, -0.05, angular velocity about x, y, z]; 74 and 82 inputs
       goal_quat       goal orientation (4), and the rotation from the goal to the tracked orientation (4); 84 inputs
       last_action     the policy's previous raw output
       last_joint_cmd  the previous joint position command (radians)
3. Run the policy, map its output onto the joint limits and smooth it with the same moving average as the simulation
   (target = alpha * new + (1 - alpha) * previous)
4. Publish the joint position command on /right/hand/policy_output, which hand_control_node executes (unless dry_run)

Cube pose: every step the node reads the cube pose from TF (hand_root -> cube, published by cube_spin_v2.launch.py and
cube_detection_node) and converts it to the frame of the simulation (mdp/cube_tracker.py in faive_lab). The tracked flag
is 1 if the newest pose is at most cube_pose_timeout_s old. While the cube is lost, the node holds the last pose with the
flag 0. The simulated tracker starts each episode with the cube found, so a policy with the cube pose starts only at the
first cube pose: until then, the node commands nothing.

Goal orientations (84 inputs): the node samples the goals, takes the next one when the tracked orientation is within
goal_tolerance, and publishes the goal as the TF frame cube_goal (at the cube, in hand_root) for RViz.

The joint limits, observation order and scales in cube_spin_policy.py must match the training task (orca_v2_repose_env_cfg.py).

Usage (hand driver running, e.g. `ros2 launch hand_control orcahand_v2_controller_launch.yaml`):
    python policy_inference_cube_spin_v2.py --ros-args -p policy_ckpt_path:=/path/to/exported/policy.pt
"""

import csv
from pathlib import Path

import numpy as np
import rclpy
import rclpy.executors
import tf2_ros
import yaml
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.time import Time
from scipy.spatial.transform import Rotation
from std_msgs.msg import Float32MultiArray

from faive_system.src.common.ros_utils import numpy_to_float32_multiarray, float32_multiarray_to_numpy

from cube_spin_policy import CubeSpinPolicy, quat_angle  # next to this file: the observation and action processing

DEFAULT_HAND_CONFIG = (
    Path(__file__).resolve().parents[2] / "src" / "hand_control" / "orca" / "configs" / "orcahand_v2" / "right" / "config.yaml"
)

# If in your simulation environment you use `mdp.root_pos_w` for the cube position,
# you need to represent your cube position in the simulator's world frame.
# Read your robot position from the env cfg and set the obs_T_hand accordingly.
# Here: robot init_state pos (0.0004, 0.1512, -0.0187), rot (w, x, y, z) (-0.2607, 0.9654, 0, 0) of the
# root link right_forearm, and hand_root -> forearm 90 deg about x (orcahand_v2.urdf.xacro): 120.2 deg about x.
# If your simulator observes the cube relative to the hand instead, note that the robot's root link is
# forearm, not hand_root: use forearm_T_hand_root (90 deg about -x) @ hand_T_cube.
OBS_T_HAND = np.array([
    [1.0,  0.0,     0.0,     0.0004],
    [0.0, -0.5034, -0.8641,  0.1512],
    [0.0,  0.8641, -0.5034, -0.0187],
    [0.0,  0.0,     0.0,     1.0],
])


class PolicyInferenceNodeV2(Node):
    def __init__(self):
        super().__init__("policy_inference_node_v2")

        # ========== ROS2 Parameters ==========
        self.declare_parameter("policy_ckpt_path", "")
        self.declare_parameter("hand_config_path", str(DEFAULT_HAND_CONFIG))
        self.declare_parameter("policy_hz", 20.0)          # the task runs the policy at 20 Hz (decimation 6, dt 1/120)
        self.declare_parameter("action_alpha", 0.2)        # moving average of the task's action term
        self.declare_parameter("velocity_obs_scale", 0.2)  # joint_vel observation scale in the task
        self.declare_parameter("goal_ang_vel", 1.0)        # rad/s, the sign sets the direction; use the training range
        self.declare_parameter("goal_axis", "z")           # x, y or z of the simulation's env frame (all-axes policies)
        self.declare_parameter("goal_flip_period_s", 0.0)  # > 0: flip the spin direction every this many seconds
        self.declare_parameter("goal_mode", "uniform")     # goal orientations: uniform (as trained) or face90
        self.declare_parameter("goal_tolerance", 0.4)      # rad; a goal counts within this angle (DeXtreme's test value)
        self.declare_parameter("dry_run", False)           # true: compute and log actions, but do not command the hand
        self.declare_parameter("log_period_s", 1.0)        # print the observation summary this often (0: never)
        self.declare_parameter("csv_log_path", "")         # non-empty: log observations and commands to this CSV
        self.declare_parameter("command_topic", "/right/hand/policy_output")  # what hand_control_node subscribes to

        # Frame IDs
        self.declare_parameter("hand_frame_id", "hand_root")
        self.declare_parameter("cube_frame_id", "cube")
        self.declare_parameter("goal_frame_id", "cube_goal")
        self.declare_parameter("cube_pose_timeout_s", 0.25)  # an older cube pose counts as not tracked; = timeout_s of CUBE_TRACKER

        self.policy_ckpt_path = self.get_parameter("policy_ckpt_path").value
        self.policy_hz = float(self.get_parameter("policy_hz").value)
        self.action_alpha = float(self.get_parameter("action_alpha").value)
        self.velocity_obs_scale = float(self.get_parameter("velocity_obs_scale").value)
        self.goal_flip_period_s = float(self.get_parameter("goal_flip_period_s").value)
        self.dry_run = bool(self.get_parameter("dry_run").value)
        self.log_period_s = float(self.get_parameter("log_period_s").value)
        self.hand_frame = self.get_parameter("hand_frame_id").value
        self.cube_frame = self.get_parameter("cube_frame_id").value
        self.goal_frame = self.get_parameter("goal_frame_id").value
        self.cube_pose_timeout_s = float(self.get_parameter("cube_pose_timeout_s").value)

        # ========== Policy ==========
        # the hand driver publishes and accepts joints in the order of config.yaml's joint_ids
        hand_config = yaml.safe_load(Path(self.get_parameter("hand_config_path").value).read_text())
        if not self.policy_ckpt_path:
            raise ValueError("Set policy_ckpt_path to the exported policy.pt (faive_lab logs/.../exported/policy.pt)")
        self.core = CubeSpinPolicy(self.policy_ckpt_path, hand_config["joint_ids"],
                                   goal_ang_vel=float(self.get_parameter("goal_ang_vel").value),
                                   action_alpha=self.action_alpha, velocity_obs_scale=self.velocity_obs_scale,
                                   goal_axis=self.get_parameter("goal_axis").value,
                                   goal_mode=self.get_parameter("goal_mode").value,
                                   goal_tolerance=float(self.get_parameter("goal_tolerance").value))
        self.joint_order_usd = self.core.joint_order_usd
        if self.core.layout == "goal":
            command = f"goal orientations ({self.core.goal_mode}, tolerance {self.core.goal_tolerance:.2f} rad)"
            if self.goal_flip_period_s > 0:
                self.get_logger().warn("goal_flip_period_s is ignored: the policy takes goal orientations")
                self.goal_flip_period_s = 0.0
        else:
            command = f"angular velocity {np.round(self.core.goal_command[3:6], 2).tolist()} rad/s"
        self.get_logger().info(f"Loaded policy {self.policy_ckpt_path}: {self.core.obs_dim} observations "
                               f"({self.core.layout}) -> 17 actions, {command}")

        # ========== ROS2 Publishers and Subscribers ==========
        self.hand_pub = self.create_publisher(Float32MultiArray, self.get_parameter("command_topic").value, 10)
        self.create_subscription(Float32MultiArray, "/right/hand/joint_position_from_motor", self.hand_pos_callback, 10)
        self.create_subscription(Float32MultiArray, "/right/hand/joint_velocity_from_motor", self.hand_vel_callback, 10)

        # ========== TF2 ==========
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.last_object_pos = None  # the last tracked cube pose (simulation frame), held while the cube is lost
        self.last_object_quat = None
        self.last_hand_t_cube = None  # its position in hand_root, for the goal frame

        # ========== State ==========
        self.current_joint_pos = None  # robot order, radians
        self.current_joint_vel = None
        self.start_time = None
        self.last_flip_time = None
        self.last_log_time = None
        self.goals_logged = 0

        self.csv_writer = None
        csv_log_path = self.get_parameter("csv_log_path").value
        if csv_log_path:
            self.csv_file = open(csv_log_path, "w", newline="")
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(["time"] + [f"obs_{i}" for i in range(self.core.obs_dim)]
                                     + [f"cmd_{name}" for name in self.joint_order_usd]
                                     + ["cube_x", "cube_y", "cube_z", "cube_qw", "cube_qx", "cube_qy", "cube_qz",
                                        "cube_tracked", "cube_age_s", "goals_reached"])

        self.create_timer(1.0 / self.policy_hz, self.process_once)
        mode = "DRY RUN (not commanding the hand)" if self.dry_run else "COMMANDING THE HAND"
        self.get_logger().info(f"Policy inference v2 running at {self.policy_hz} Hz, alpha {self.action_alpha}: {mode}")

    def hand_pos_callback(self, msg: Float32MultiArray):
        self.current_joint_pos = float32_multiarray_to_numpy(msg)

    def hand_vel_callback(self, msg: Float32MultiArray):
        self.current_joint_vel = float32_multiarray_to_numpy(msg)

    def lookup_transform_from_tf(self, parent_frame, child_frame):
        """Latest parent_T_child from TF as a 4x4 matrix with its time stamp (seconds), or (None, None)."""
        try:
            tf_msg = self.tf_buffer.lookup_transform(parent_frame, child_frame, Time())
        except tf2_ros.TransformException:
            return None, None
        t, q = tf_msg.transform.translation, tf_msg.transform.rotation
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()  # ROS order is (x, y, z, w)
        transform[:3, 3] = [t.x, t.y, t.z]
        return transform, Time.from_msg(tf_msg.header.stamp).nanoseconds * 1e-9

    def publish_goal_frame(self):
        """The goal orientation as the TF frame goal_frame_id, at the last cube position, in hand_root."""
        w, x, y, z = self.core.goal_quat
        hand_R_goal = OBS_T_HAND[:3, :3].T @ Rotation.from_quat([x, y, z, w]).as_matrix()
        qx, qy, qz, qw = Rotation.from_matrix(hand_R_goal).as_quat()
        msg = TransformStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.hand_frame
        msg.child_frame_id = self.goal_frame
        msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z = map(
            float, self.last_hand_t_cube)
        msg.transform.rotation.x, msg.transform.rotation.y = float(qx), float(qy)
        msg.transform.rotation.z, msg.transform.rotation.w = float(qz), float(qw)
        self.tf_broadcaster.sendTransform(msg)

    def process_once(self):
        if self.current_joint_pos is None or self.current_joint_vel is None:
            self.get_logger().info("Waiting for hand state...", throttle_duration_sec=2.0)
            return
        if not (np.all(np.isfinite(self.current_joint_pos)) and np.all(np.isfinite(self.current_joint_vel))):
            self.get_logger().warn("Non-finite joint state, skipping this step", throttle_duration_sec=1.0)
            return

        now = self.get_clock().now().nanoseconds * 1e-9

        # === cube pose ===
        hand_T_cube, stamp = self.lookup_transform_from_tf(self.hand_frame, self.cube_frame)
        cube_age = now - stamp if hand_T_cube is not None else float("nan")
        tracked = cube_age <= self.cube_pose_timeout_s  # False for nan
        if tracked:
            obs_T_cube = OBS_T_HAND @ hand_T_cube
            self.last_hand_t_cube = hand_T_cube[:3, 3].copy()
            self.last_object_pos = obs_T_cube[:3, 3].astype(np.float32)
            # Convert quaternion from ROS (x,y,z,w) to policy (w,x,y,z)
            x, y, z, w = Rotation.from_matrix(obs_T_cube[:3, :3]).as_quat()
            object_quat = np.array([w, x, y, z], dtype=np.float32)
            # q and -q are the same rotation; the simulation's quaternion changes continuously, so keep the previous sign
            if self.last_object_quat is not None and np.dot(object_quat, self.last_object_quat) < 0:
                object_quat = -object_quat
            self.last_object_quat = object_quat

        if self.core.uses_cube_pose and self.last_object_pos is None:
            self.get_logger().info(
                f"Waiting for the first cube pose ({self.hand_frame} -> {self.cube_frame}): check the camera, the detector "
                "and the calibration. The hand is not commanded yet.", throttle_duration_sec=2.0)
            return

        if self.start_time is None:
            # the policy core starts as at a reset in the task: moving average from the current pose, last action zero
            self.start_time = self.last_flip_time = self.last_log_time = now
        if self.goal_flip_period_s > 0 and now - self.last_flip_time >= self.goal_flip_period_s:
            self.core.flip_direction()
            self.last_flip_time = now
            self.get_logger().info(
                f"Spin direction flipped: goal angular velocity {np.round(self.core.goal_command[3:6], 2).tolist()} rad/s")

        # === observation, policy and action processing (cube_spin_policy.py) ===
        cube_pose = (self.last_object_pos, self.last_object_quat, tracked) if self.core.uses_cube_pose else None
        target, obs, action_usd, target_usd = self.core.step(self.current_joint_pos, self.current_joint_vel, cube_pose)

        if not self.dry_run:
            self.hand_pub.publish(numpy_to_float32_multiarray(target))
        if self.core.layout == "goal":
            self.publish_goal_frame()
            if self.core.goals_reached > self.goals_logged:
                self.goals_logged = self.core.goals_reached
                self.get_logger().info(f"Goal {self.goals_logged} reached after {now - self.start_time:.1f} s")

        has_cube_pose = self.last_object_pos is not None
        if self.csv_writer is not None:
            cube = np.concatenate([self.last_object_pos, self.last_object_quat]).tolist() if has_cube_pose else [float("nan")] * 7
            self.csv_writer.writerow([now - self.start_time] + obs.tolist() + target_usd.tolist() + cube
                                     + [int(tracked), cube_age, self.core.goals_reached])
        if self.log_period_s > 0 and now - self.last_log_time >= self.log_period_s:
            self.last_log_time = now
            norm = obs[:17]
            cube = "never seen"
            if has_cube_pose:
                state = "tracked" if tracked else f"LOST, held, age {cube_age:.2f} s"
                cube = f"pos {np.round(self.last_object_pos, 3)} quat {np.round(self.last_object_quat, 2)} ({state})"
            goal = ""
            if self.core.layout == "goal":
                goal = (f" | goal {np.degrees(quat_angle(self.core.goal_diff(self.last_object_quat))):.0f} deg away, "
                        f"{self.core.goals_reached} reached")
            self.get_logger().info(
                "joint_pos (normalised): " + " ".join(f"{n}={v:+.2f}" for n, v in zip(self.joint_order_usd, norm))
                + f" | |action| max {np.abs(action_usd).max():.2f} | cube {cube}{goal}"
            )

    def destroy_node(self):
        if self.csv_writer is not None:
            self.csv_file.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PolicyInferenceNodeV2()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        print("\nShutting down...")
    finally:
        node.destroy_node()
        if rclpy.ok():  # a signal may already have shut the context down
            rclpy.shutdown()


if __name__ == "__main__":
    main()
