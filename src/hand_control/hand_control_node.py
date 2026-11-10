#!/usr/bin/env python3
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
import numpy as np
from std_msgs.msg import Float32MultiArray, Int32MultiArray
from std_srvs.srv import SetBool
from rcl_interfaces.msg import ParameterDescriptor
import os
import signal
from ament_index_python.packages import get_package_share_directory


class HandControllerNode(Node):
    def __init__(self, debug=False):
        super().__init__("hand_controller_node")

        # Determine which hand controller to use
        self.declare_parameter("hand_controller/hand_name", "orca_v2")
        self.declare_parameter("hand_controller/hand_chirality", "right")
        self.declare_parameter("hand_controller/calibrate", False)
        self.declare_parameter('hand_controller/compliant_test_mode', False)

        self.hand_name = self.get_parameter("hand_controller/hand_name").value
        self.chirality = self.get_parameter("hand_controller/hand_chirality").value
        self.calibrate = self.get_parameter("hand_controller/calibrate").value
        self.compliant_test_mode = self.get_parameter('hand_controller/compliant_test_mode').value
        # true: switch the torque off when the node stops (Ctrl+C, SIGTERM), so the hand does not keep holding a pose
        self.declare_parameter('hand_controller/torque_off_on_exit', False)
        self.torque_off_on_exit = self.get_parameter('hand_controller/torque_off_on_exit').value
        self.get_logger().info(f"Hand Controller Node Started with hand: {self.hand_name}")
        self.get_logger().info(f"Hand Controller Node Started with chirality: {self.chirality}")

        # Franka gripper
        if self.hand_name == "franka_hand":
            from franka_gripper.hand_controller import HandController as FrankaGripperHandController
            self._hc = FrankaGripperHandController(hand_chirality = "right" if self.get_parameter("hand_controller/hand_chirality").value == "right" else "left")

        # Orca v2 hand, driven through the orca_core package
        elif self.hand_name == "orca_v2":
            self.declare_parameter('hand_controller/hand_cfg_name', 'orcahand_v2')
            hand_cfg_name = self.get_parameter('hand_controller/hand_cfg_name').value

            config_dir_path = os.path.join(
                get_package_share_directory("hand_control"),
                "orca",
                "configs",
                hand_cfg_name
            )
            # publisher for sensor values (in radians)
            self.joint_angle_pub = self.create_publisher(Float32MultiArray, f'/{self.chirality}/hand/joint_position_from_motor', 10)
            self.joint_velocity_pub = self.create_publisher(Float32MultiArray, f'/{self.chirality}/hand/joint_velocity_from_motor', 10)
            # present motor currents (mA, joint_ids order), read together with the positions
            hand_topic = f'/{self.chirality}/hand'
            self.motor_current_pub = self.create_publisher(Float32MultiArray, f'{hand_topic}/motor_current', 10)
            # once per second: supply voltage (V) and the servo status bytes seen since the last message (bits:
            # 0x01 input voltage, 0x02 angle sensor, 0x04 overheating, 0x08 overcurrent, 0x20 overload)
            self.motor_voltage_pub = self.create_publisher(Float32MultiArray, f'{hand_topic}/motor_voltage', 10)
            self.motor_error_pub = self.create_publisher(Int32MultiArray, f'{hand_topic}/motor_errors', 10)
            self._status_warned = False  # "no voltage and status bytes" is logged once

            # optional wrist angle (degrees) of the start-up pose instead of the config's neutral_position
            self.declare_parameter('hand_controller/neutral_wrist_deg', 'none',
                                   ParameterDescriptor(dynamic_typing=True))
            neutral_wrist = self.get_parameter('hand_controller/neutral_wrist_deg').value
            neutral_overrides = None
            if neutral_wrist not in (None, '', 'none'):
                neutral_overrides = {'wrist': float(neutral_wrist)}
                self.get_logger().info(f"Start-up pose: neutral_position with the wrist at {float(neutral_wrist):.1f} deg")

            from faive_system.src.hand_control.orca.hand_controller_v2 import OrcaHandV2
            # orca_core requires the file name config.yaml; calibration.yaml is stored next to it
            self._hc = OrcaHandV2(os.path.join(config_dir_path, self.chirality, "config.yaml"))
            self._hc.init_joints(calibrate=self.calibrate, initialize=True, compliant_test_mode=self.compliant_test_mode,
                                 neutral_overrides=neutral_overrides)
            self._orca_timer = self.create_timer(
                0.01, # seconds
                self.orca_v2_control_cb,  # callback
            )
            # motor temperatures (°C, joint_ids order) once per second, e.g. for evaluate_real_policies.py
            self.temperature_pub = self.create_publisher(Float32MultiArray, f'/{self.chirality}/hand/motor_temperature', 10)
            self._temperature_timer = self.create_timer(1.0, self.orca_v2_temperature_cb)

        else:
            raise NotImplementedError("Hand name not recognized")

        self.joint_angle_sub = self.create_subscription(
            Float32MultiArray, f"/{self.chirality}/hand/policy_output", self.joint_angle_cb, 0
        )
        
        # Add emergency stop service, kills itself when value is True
        self.estop_service = self.create_service(
            SetBool, f"/{self.chirality}/hand/estop", self.faive_estop_cb
        )

        # Service to enable/disable torque on the underlying hand controller.
        self.torque_service = self.create_service(
            SetBool,
            f"/{self.chirality}/hand/set_torque_enabled",
            self.set_torque_enabled_cb,
        )

    def joint_angle_cb(self, msg):
        num_motors = self._hc.num_motors
        if len(msg.data) != num_motors:
            self.get_logger().error(f"Expected {num_motors} joint angles, got {len(msg.data)}")
            return
        joint_angles = np.array(msg.data)
        joint_angles = np.rad2deg(joint_angles)
        if self.hand_name == "orca_v2":
            self._hc.set_joint_pos_array(joint_angles)
        else:
            self._hc.command_joint_angles(joint_angles)


    def orca_v2_control_cb(self):
        """
        publish joint angles and velocities from the orca_core hand to ROS
        """
        q_deg, qd_deg, current_ma = self._hc.read_joint_state()
        self.joint_angle_pub.publish(Float32MultiArray(data=np.deg2rad(q_deg).astype(np.float32)))
        self.joint_velocity_pub.publish(Float32MultiArray(data=np.deg2rad(qd_deg).astype(np.float32)))
        self.motor_current_pub.publish(Float32MultiArray(data=current_ma.astype(np.float32)))
        self._hc.collect_status()

    def orca_v2_temperature_cb(self):
        """
        publish the motor temperatures (°C, joint_ids order)
        """
        try:
            self.temperature_pub.publish(Float32MultiArray(data=self._hc.read_motor_temperature().tolist()))
        except Exception as e:
            self.get_logger().warn(f"Could not read the motor temperatures: {e}", throttle_duration_sec=10.0)
        status = self._hc.take_status()
        if status is not None:
            voltage, errors = status
            self.motor_voltage_pub.publish(Float32MultiArray(data=voltage.astype(np.float32).tolist()))
            self.motor_error_pub.publish(Int32MultiArray(data=[int(e) for e in errors]))
        elif self._hc.status_available is False and not self._status_warned:
            self._status_warned = True
            self.get_logger().warn(f"No motor voltage and status bytes: {self._hc.status_problem}")


    def set_torque_enabled_cb(self, request, response):
        """
        Enable or disable torque on the underlying hand controller via a SetBool service.
        """
        try:
            if request.data:
                self._hc.enable_torque()
                self.get_logger().info("Torque enabled via service")
                response.success = True
                response.message = "Torque enabled"
            else:
                self._hc.disable_torque()
                self.get_logger().info("Torque disabled via service")
                response.success = True
                response.message = "Torque disabled"
        except Exception as e:
            self.get_logger().error(f"Failed to set torque: {e}")
            response.success = False
            response.message = str(e)
        return response

    def faive_estop_cb(self, request, response):
        if request.data:
            self.get_logger().warn("Emergency stop received, shutting down")
            if self.hand_name == "orca_v2":
                # stop reading and commanding before the bus goes away. Without this, an e-stop crashes the node
                if not self._orca_timer.is_canceled():
                    self._orca_timer.cancel()
                    self._temperature_timer.cancel()
                    self.destroy_subscription(self.joint_angle_sub)
                    self._hc.disconnect()
            else:
                self._hc.disable_torque()
                self._hc.disconnect_from_dynamixels()

            response.success = True
            response.message = "Emergency stop executed"
            return response
        else:
            response.success = False
            response.message = "contents not True, Emergency stop not executed"
            return response

    def release_hand(self):
        """On exit with torque_off_on_exit: stop commanding, switch the torque off and close the motor bus."""
        if not (self.torque_off_on_exit and self.hand_name == "orca_v2") or self._orca_timer.is_canceled():
            return  # not wanted, or the e-stop already disconnected the hand
        self._orca_timer.cancel()
        self._temperature_timer.cancel()
        try:
            self._hc.disable_torque()
            print("[hand_controller_node] torque off on exit", flush=True)
        except Exception as e:
            print(f"[hand_controller_node] could not switch the torque off on exit: {e}", flush=True)
        try:
            self._hc.disconnect()
        except Exception:
            pass


def main(args=None):
    # Ctrl+C must not raise KeyboardInterrupt inside a motor bus transaction: the bus then stays "in use" and the
    # torque can no longer be switched off. Without a Python handler, rclpy's own handler shuts the node down after
    # the current callback (spin raises ExternalShutdownException).
    signal.signal(signal.SIGINT, lambda *_: None)
    rclpy.init(args=args)
    node = HandControllerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.release_hand()
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == "__main__":
    main()
