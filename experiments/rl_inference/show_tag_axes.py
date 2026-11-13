#!/usr/bin/env python3
"""Live view of the camera with every tag36h11 AprilTag it sees: ID and axes (x red, y green, z blue, into the tag).
Use it to read hand_to_tag_yaw of cube_spin_v2.launch.py from the direction of the red x axis.

Needs the camera node (ros2 run ingress oakd_node.py ...). Press q or Esc to quit.

    python3 show_tag_axes.py [--tag-size 0.08] [--camera oakd_bottom_view]
"""
import argparse

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from ingress.msg import CameraIntrinsics

try:
    from dt_apriltags import Detector
except ImportError:
    from pupil_apriltags import Detector


class Viewer(Node):
    def __init__(self, camera):
        super().__init__("show_tag_axes")
        self.K, self.img = None, None
        self.create_subscription(CameraIntrinsics, f"/{camera}/intrinsics", self.on_k,
                                 QoSProfile(depth=2, reliability=ReliabilityPolicy.RELIABLE,
                                            durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                            history=HistoryPolicy.KEEP_LAST))
        self.create_subscription(Image, f"/{camera}/color", self.on_img,
                                 QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT,
                                            history=HistoryPolicy.KEEP_LAST))

    def on_k(self, m):
        self.K = m

    def on_img(self, m):
        self.img = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3).copy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag-size", type=float, default=0.08)
    p.add_argument("--camera", default="oakd_bottom_view")
    args = p.parse_args()

    rclpy.init()
    node = Viewer(args.camera)
    detector = Detector(families="tag36h11")
    print("Waiting for images and intrinsics ...")
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            if node.img is None or node.K is None:
                continue
            img, k = node.img, node.K
            node.img = None
            K = np.array([[k.fx, 0, k.cx], [0, k.fy, k.cy], [0, 0, 1]])
            dets = detector.detect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), estimate_tag_pose=True,
                                   camera_params=(k.fx, k.fy, k.cx, k.cy), tag_size=args.tag_size)
            for d in dets:
                rvec, _ = cv2.Rodrigues(d.pose_R)
                L = args.tag_size
                pts, _ = cv2.projectPoints(np.float32([[0, 0, 0], [L, 0, 0], [0, L, 0], [0, 0, L]]),
                                           rvec, d.pose_t, K, None)
                pts = pts.reshape(-1, 2).astype(int)
                for i, color in zip((1, 2, 3), ((0, 0, 255), (0, 255, 0), (255, 0, 0))):
                    cv2.arrowedLine(img, tuple(pts[0]), tuple(pts[i]), color, 3, tipLength=0.2)
                cv2.putText(img, f"id {d.tag_id}  {np.linalg.norm(d.pose_t):.2f} m", tuple(pts[0] + [10, -10]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            if not dets:
                cv2.putText(img, "no tag detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.imshow("AprilTag axes (x red, y green, z blue) - q to quit", img)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
