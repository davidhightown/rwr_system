#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import cv2
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, MultiArrayLayout
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError
from threading import RLock
from faive_system.src.ingress.oakd.oakd_ingress import OakDDriver
import depthai as dai
from copy import deepcopy
import time
from faive_system.src.common.utils import numpy_to_float32_multiarray
from faive_system.src.common.ros_utils import restore_library_path_on_macos
from ingress.msg import CameraIntrinsics
from datetime import datetime
import os
import yaml

restore_library_path_on_macos()

class OakDPublisher(Node):
    def __init__(self, camera_dict=None):
        super().__init__("oakd_publisher")
        self.declare_parameter("use_depth", False)
        self.declare_parameter("config_file", "")
        # > 0: fixed exposure time (us) and ISO of the color camera; 0: the camera's entry in the config file, else auto
        self.declare_parameter("exposure_us", 0)
        self.declare_parameter("iso", 0)
        # a camera ID (mxid) to use instead of the one in the config file, e.g. another camera of the same setup
        self.declare_parameter("camera_id", "")
        cfg_path: str = self.get_parameter("config_file").get_parameter_value().string_value
        if not cfg_path:
            cfg_path = os.path.join(os.path.dirname(__file__), "oakd_cams.yaml")
        if not os.path.isfile(cfg_path):
            raise FileNotFoundError(
                "Parameter 'config_file' is empty or the file does not exist!"
            )
        with open(cfg_path, "r") as f:
            yaml_cfg = yaml.safe_load(f)

        self.use_depth = self.get_parameter("use_depth").value
        self.bridge = CvBridge()
        self.camera_dict = {}
        self.exposure = {}  # serial -> (exposure_us, iso)
        available_cams = dai.Device.getAllAvailableDevices()
        available_ids = [c.mxid for c in available_cams]
        entries = dict(yaml_cfg or {})
        camera_id = self.get_parameter("camera_id").value.strip()
        if camera_id:
            # keeps the settings of the camera's entry, or of the only entry of a one-camera config
            template = entries.get(camera_id) or (next(iter(entries.values())) if len(entries) == 1 else {})
            entries = {camera_id: template}
        for serial_str, values in entries.items():
            if serial_str not in available_ids:
                self.get_logger().warn(f"oakd camera {serial_str} not available (connected: {available_ids})")
                continue
            self._add_camera(serial_str, values)
        if not self.camera_dict and not camera_id and len(entries) == 1 and len(available_ids) == 1:
            # a one-camera setup with another camera than the config's: use the one that is connected
            serial_str = available_ids[0]
            self.get_logger().warn(f"using the only connected oakd camera {serial_str} instead of "
                                   f"{next(iter(entries))} (parameter camera_id chooses one)")
            self._add_camera(serial_str, next(iter(entries.values())))
        if not self.camera_dict:
            self.get_logger().error(f"no oakd camera of {cfg_path} connected; connected: {available_ids} "
                                    "(parameter camera_id chooses one)")

        self.init_cameras()

    def _add_camera(self, serial_str, values):
        values = values or {}
        self.camera_dict[values.get("name", f"cam_{serial_str}")] = serial_str
        exposure_us = self.get_parameter("exposure_us").value or values.get("exposure_us", 0)
        iso = self.get_parameter("iso").value or values.get("iso", 0)
        self.exposure[serial_str] = (int(exposure_us), int(iso))

    def init_cameras(self):

        for camera_name, camera_id in self.camera_dict.items():
            self.camera_dict[camera_name] = {
                "color_stamp": None,
                "driver": OakDDriver(
                    self.recv_oakd_images,
                    device_mxid=camera_id,
                    camera_name=camera_name,
                    use_depth=self.use_depth,
                    exposure_us=self.exposure[camera_id][0],
                    iso=self.exposure[camera_id][1],
                ),
                "rgb_output_pub": self.create_publisher(
                    Image, f"/oakd_{camera_name}/color",  
                    QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, durability=DurabilityPolicy.VOLATILE)
                ),

                "depth_output_pub": self.create_publisher(
                    Image, f"/oakd_{camera_name}/depth", 
                    QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, durability=DurabilityPolicy.VOLATILE)
                ) if self.use_depth and camera_name != "wrist_view" else None,

                "intrinsics_pub": self.create_publisher(
                    CameraIntrinsics, f"/oakd_{camera_name}/intrinsics", 
                    QoSProfile(depth=2, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, durability=DurabilityPolicy.TRANSIENT_LOCAL)
                ),
            }

    def recv_oakd_images(self, color, depth, color_stamp, camera_name):
        if color is None:
            return
        try:
            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = "world"
            # "passthrough" and then setting the encoding skips cv_bridge's encoding check, which fails when the
            # OpenCV of Python (pip) and the OpenCV cv_bridge was built with (conda) are different versions
            output_img_rgb = self.bridge.cv2_to_imgmsg(color, "passthrough", header=header)
            output_img_rgb.encoding = "bgr8"
            self.camera_dict[camera_name]["rgb_output_pub"].publish(
                output_img_rgb
            )
            # print(f"Published image for {camera_name}")
            self.camera_dict[camera_name]["color_stamp"] = color_stamp
        except CvBridgeError as e:
            self.get_logger().error(f"{camera_name} Error publishing color image: {e}")
        
        if depth is None:
            return
        try:
            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = "world"
            # same as for the color image above
            output_img_depth = self.bridge.cv2_to_imgmsg(depth, "passthrough", header=header)
            output_img_depth.encoding = "mono16"
            self.camera_dict[camera_name]["depth_output_pub"].publish(
                output_img_depth
            )
        except CvBridgeError as e:
            self.get_logger().error(f"{camera_name} Error publishing depth image: {e}")
        

    def publish_cam_info(self):
        for camera_name in self.camera_dict.keys():
            if self.camera_dict[camera_name]["color_stamp"] is not None:
                current_time = dai.Clock.now()
                time_diff = current_time - self.camera_dict[camera_name]["color_stamp"]
                if time_diff.seconds > 1.0:
                    self.get_logger().error(f"{camera_name} latency larger than 1s")

                intrinsic_matrix = np.array(self.camera_dict[camera_name]["driver"].intrinsics)
                msg = CameraIntrinsics(
                    fx = intrinsic_matrix[0,0],
                    fy = intrinsic_matrix[1,1],
                    cx = intrinsic_matrix[0,2],
                    cy = intrinsic_matrix[1,2],
                    disto = self.camera_dict[camera_name]["driver"].distortion_coeff
                )
                self.camera_dict[camera_name]["intrinsics_pub"].publish(msg)


def main():
    rclpy.init()
    print("Starting oakd_publisher")
    oakd_publisher = OakDPublisher()
    import threading

    spin_thread = threading.Thread(target=rclpy.spin, args=(oakd_publisher,))
    while rclpy.ok():
        oakd_publisher.publish_cam_info()
        time.sleep(1)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
