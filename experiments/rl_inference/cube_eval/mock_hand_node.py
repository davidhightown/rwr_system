#!/usr/bin/env python3
"""A stand-in for hand_control_node.py without a hand, for eval_policies.py --mock: the same topics and the torque
service. The joints follow the commands with a lag, and the motors heat while the torque is on and they move."""

import sys

import numpy as np
import rclpy
import rclpy.executors
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Int32MultiArray
from std_srvs.srv import SetBool

N = 17


class MockHand(Node):
    def __init__(self, heat_rate):
        super().__init__("hand_controller_node")
        self.q = np.zeros(N)
        self.target = np.zeros(N)
        self.torque = True
        self.temps = np.full(N, 30.0)
        self.heat_rate = heat_rate  # deg C/s at full motion
        self.count = 0

        def pub(kind, name):
            return self.create_publisher(kind, f"/right/hand/{name}", 10)

        self.pos_pub = pub(Float32MultiArray, "joint_position_from_motor")
        self.vel_pub = pub(Float32MultiArray, "joint_velocity_from_motor")
        self.cur_pub = pub(Float32MultiArray, "motor_current")
        self.temp_pub = pub(Float32MultiArray, "motor_temperature")
        self.volt_pub = pub(Float32MultiArray, "motor_voltage")
        self.status_pub = pub(Int32MultiArray, "motor_status")
        self.create_subscription(Float32MultiArray, "/right/hand/policy_output", self.command, 10)
        self.create_service(SetBool, "/right/hand/set_torque_enabled", self.set_torque)
        self.create_timer(0.01, self.step)

    def command(self, msg):
        self.target = np.array(msg.data, dtype=float)

    def set_torque(self, request, response):
        self.torque = bool(request.data)
        response.success, response.message = True, "ok"
        return response

    def step(self):
        velocity = (self.target - self.q) / 0.06 if self.torque else np.zeros(N)
        self.q += velocity * 0.01
        effort = np.minimum(1.0, np.abs(velocity) / 2.0) if self.torque else np.zeros(N)
        self.temps += (self.heat_rate * (0.3 * self.torque + effort) - 0.01 * (self.temps - 25.0)) * 0.01
        self.pos_pub.publish(Float32MultiArray(data=self.q.astype(np.float32)))
        self.vel_pub.publish(Float32MultiArray(data=velocity.astype(np.float32)))
        self.cur_pub.publish(Float32MultiArray(data=(100 + 600 * effort).astype(np.float32)))
        self.count += 1
        if self.count % 100 == 0:
            self.temp_pub.publish(Float32MultiArray(data=np.floor(self.temps).astype(np.float32)))
            self.volt_pub.publish(Float32MultiArray(data=np.full(N, 7.4, dtype=np.float32)))
            self.status_pub.publish(Int32MultiArray(data=[0] * N))


def main():
    heat_rate = float(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else 0.1
    rclpy.init()
    node = MockHand(heat_rate)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass


if __name__ == "__main__":
    main()
