"""ORCA v2 hand controller. Drives the hand through the orca_core package,
exposing the interface hand_control_node.py expects."""
import numpy as np
import yaml
from orca_core import load_hand

from faive_system.src.hand_control.orca import servo_profile


FEETECH_PRESENT_VOLTAGE = 62  # 0.1 V per unit; register 63 is the temperature in deg C


class OrcaHandV2:
    def __init__(self, config_path: str):
        # An explicit config_path skips orca_core's hardware autodetection.
        # calibration.yaml is read from and saved next to config_path.
        self._hand = load_hand(config_path=config_path)
        with open(config_path) as stream:
            self._config_yaml = yaml.safe_load(stream)
        servo_profile.acceleration_raw(self._config_yaml)  # reject a bad value before the hand moves
        config = self._hand.config
        self.joint_ids = list(config.joint_ids)
        self.num_motors = len(self.joint_ids)
        self.last_current_ma = None
        joint_index = {joint: i for i, joint in enumerate(self.joint_ids)}
        # get_motor_state() arrays are in motor_ids order; these map them to joint_ids order.
        self._joint_order = np.empty(self.num_motors, dtype=np.intp)
        self._joint_signs = np.ones(self.num_motors)
        for motor_idx, motor_id in enumerate(config.motor_ids):
            joint = config.motor_to_joint_dict[motor_id]
            self._joint_order[joint_index[joint]] = motor_idx
            if config.joint_inversion_dict.get(joint, False):
                self._joint_signs[joint_index[joint]] = -1.0

    def init_joints(self, calibrate: bool = False, initialize: bool = True, compliant_test_mode: bool = False):
        ok, message = self._hand.connect(interactive=False)
        if not ok:
            raise RuntimeError(f"Could not connect to the ORCA hand: {message}")
        # Calibrates automatically when no calibration exists; moves to the config's neutral pose.
        self._hand.init_joints(force_calibrate=calibrate, move_to_neutral=initialize)
        # The same servo acceleration ramp as the chirp recordings the sim was fit to (init_joints writes orca_core's own).
        self.servo_profile = servo_profile.apply(self._hand, self._config_yaml)
        if compliant_test_mode:
            self.disable_torque()

    def set_joint_pos_array(self, joint_pos_deg):
        """Command joint positions in joint_ids order, in degrees (clamped to the ROMs by orca_core)."""
        self._hand.set_joint_positions(np.asarray(joint_pos_deg, dtype=np.float64))

    def read_joint_state(self):
        """One bus read -> (positions [deg], velocities [deg/s]) in joint_ids order.

        The motor currents of the same read (mA, joint_ids order) are kept in ``last_current_ma``."""
        state = self._hand.get_motor_state()
        self.last_current_ma = np.asarray(state.current, dtype=np.float64)[self._joint_order]
        joint_pos = self._hand._motor_to_joint_pos(state.position)
        pos = np.array([np.nan if joint_pos[j] is None else joint_pos[j] for j in self.joint_ids])
        ratios = self._hand.calibration.joint_to_motor_ratios_dict  # motor rad per joint deg
        motor_ids = self._hand.config.motor_ids
        motor_ratios = np.array([ratios[motor_ids[i]] or np.nan for i in self._joint_order])
        vel = self._joint_signs * np.asarray(state.velocity)[self._joint_order] / motor_ratios
        return pos, vel

    def read_health(self):
        """One bus read -> (temperatures [deg C], supply voltages [V], servo status bytes) in joint_ids order.

        Feetech servos answer all three in one sync read of registers 62-63 (the status byte comes with every
        reply: 0x01 input voltage, 0x02 angle sensor, 0x04 overheating, 0x08 overcurrent, 0x20 overload). That
        read uses orca_core's private sync-read helper; if it is missing (another orca_core version or motor
        family), only the temperatures are read, through the public API, and the other two are None.
        """
        client = self._hand.motor_client
        try:
            with client._bus_lock:
                reader = client._sync_read(FEETECH_PRESENT_VOLTAGE, 2)
                if reader.txRxPacket() != 0:  # COMM_SUCCESS
                    client._flush_input_buffer()
                    raise IOError("health sync read failed")
                temp = np.full(self.num_motors, np.nan)
                volt = np.full(self.num_motors, np.nan)
                status = np.zeros(self.num_motors, dtype=np.int32)
                for joint_idx, motor_idx in enumerate(self._joint_order):
                    motor_id = self._hand.config.motor_ids[motor_idx]
                    available, error = reader.isAvailable(motor_id, FEETECH_PRESENT_VOLTAGE, 2)
                    if not available:
                        continue
                    volt[joint_idx] = reader.getData(motor_id, FEETECH_PRESENT_VOLTAGE, 1) * 0.1
                    temp[joint_idx] = reader.getData(motor_id, FEETECH_PRESENT_VOLTAGE + 1, 1)
                    status[joint_idx] = int(error)
            return temp, volt, status
        except (AttributeError, IOError):
            temp = np.asarray(self._hand.get_motor_temp(), dtype=np.float64)[self._joint_order]
            return temp, None, None

    def enable_torque(self):
        self._hand.enable_torque()

    def disable_torque(self):
        self._hand.disable_torque()

    def disconnect(self):
        self._hand.disconnect()
