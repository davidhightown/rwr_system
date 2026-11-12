#!/usr/bin/env python3
"""Lets RViz draw the 3D scene (hand model, cube, tag, frames) over the camera image, to check the calibration and
the hand_to_tag_* values of cube_spin_v2.launch.py.

The OAK-D node publishes its images with frame_id "world" and its intrinsics as ingress/CameraIntrinsics, but RViz's
Camera display needs the image in the camera's (optical) TF frame and a sensor_msgs/CameraInfo next to it. This node
republishes:
    <camera>/overlay/image        the color image, frame_id <camera_name>_frame
    <camera>/overlay/camera_info  the matching intrinsics
    <camera>/overlay/markers      the tag's black square (in tag_frame) and the mount (yellow sphere)

In RViz, the Camera display "Camera overlay" shows them. If the calibration and hand_to_tag are right, the grey square
covers the real tag and the hand model covers the real hand.
"""

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import Marker, MarkerArray
from ingress.msg import CameraIntrinsics


class CameraOverlay(Node):
    def __init__(self):
        super().__init__("camera_overlay")
        self.declare_parameter("camera_name", "oakd_bottom_view")
        self.declare_parameter("tag_size", 0.08)
        camera = self.get_parameter("camera_name").value
        self.frame_id = f"{camera}_frame"
        self.tag_size = float(self.get_parameter("tag_size").value)
        self.intrinsics = None

        best_effort = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT,
                                 history=HistoryPolicy.KEEP_LAST, durability=DurabilityPolicy.VOLATILE)
        latched = QoSProfile(depth=2, reliability=ReliabilityPolicy.RELIABLE,
                             history=HistoryPolicy.KEEP_LAST, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(CameraIntrinsics, f"/{camera}/intrinsics", self.on_intrinsics, latched)
        self.create_subscription(Image, f"/{camera}/color", self.on_image, best_effort)
        self.image_pub = self.create_publisher(Image, f"/{camera}/overlay/image", best_effort)
        self.info_pub = self.create_publisher(CameraInfo, f"/{camera}/overlay/camera_info", best_effort)
        self.marker_pub = self.create_publisher(MarkerArray, f"/{camera}/overlay/markers", latched)
        self.create_timer(1.0, self.publish_markers)
        self.get_logger().info(f"Waiting for /{camera}/color and /{camera}/intrinsics")

    def on_intrinsics(self, msg):
        if self.intrinsics is None:
            self.get_logger().info(f"Intrinsics: fx {msg.fx:.1f} fy {msg.fy:.1f} cx {msg.cx:.1f} cy {msg.cy:.1f}")
        self.intrinsics = msg

    def on_image(self, msg):
        if self.intrinsics is None:
            return
        msg.header.frame_id = self.frame_id
        k = self.intrinsics
        info = CameraInfo()
        info.header = msg.header
        info.width, info.height = msg.width, msg.height
        info.distortion_model = "plumb_bob"
        info.d = [float(v) for v in list(k.disto)[:5]] + [0.0] * max(0, 5 - len(k.disto))
        info.k = [k.fx, 0.0, k.cx, 0.0, k.fy, k.cy, 0.0, 0.0, 1.0]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [k.fx, 0.0, k.cx, 0.0, 0.0, k.fy, k.cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        self.info_pub.publish(info)
        self.image_pub.publish(msg)

    def publish_markers(self):
        tag = Marker()
        tag.header.frame_id = "tag_frame"
        tag.ns, tag.id, tag.type, tag.action = "tag", 0, Marker.CUBE, Marker.ADD
        tag.pose.orientation.w = 1.0
        tag.scale.x = tag.scale.y = self.tag_size
        tag.scale.z = 0.001
        tag.color.r = tag.color.g = tag.color.b = 0.6
        tag.color.a = 0.7

        mount = Marker()
        mount.header.frame_id = "mount"
        mount.ns, mount.id, mount.type, mount.action = "mount", 0, Marker.SPHERE, Marker.ADD
        mount.pose.orientation.w = 1.0
        mount.scale.x = mount.scale.y = mount.scale.z = 0.015
        mount.color.r, mount.color.g, mount.color.b, mount.color.a = 1.0, 0.85, 0.0, 1.0
        self.marker_pub.publish(MarkerArray(markers=[tag, mount]))


def main():
    rclpy.init()
    node = CameraOverlay()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
