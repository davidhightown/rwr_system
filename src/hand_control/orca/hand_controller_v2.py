"""ORCA v2 hand controller. Drives the hand through the orca_core package,
exposing the interface hand_control_node.py expects."""
import dataclasses

import numpy as np
import yaml
from orca_core import OrcaJointPositions, load_hand
from orca_core.hardware_hand import NUM_STEPS

from faive_system.src.hand_control.orca import servo_profile


FEETECH_PRESENT_VOLTAGE = 62  # 0.1 V per unit; register 63 is the temperature in deg C


class OrcaHandV2:
    def __init__(self, config_path: str, acceleration_raw=None, max_current_ma=None, calibration_current_ma=None):
        # An explicit config_path skips orca_core's hardware autodetection.
        # calibration.yaml is read from and saved next to config_path.
        # acceleration_raw: the servo ramp to write instead of the config's servo_acceleration_raw (None: the config's)
        # max_current_ma: every motor's goal-current limit (mA) instead of the config's max_current (None: the
        # config's); written after orca_core's init_joints, which writes the config's (and calibrates with its own)
        self._hand = load_hand(config_path=config_path)
        with open(config_path) as stream:
            self._config_yaml = yaml.safe_load(stream)
        if acceleration_raw is not None:
            self._config_yaml[servo_profile.CONFIG_KEY] = acceleration_raw
        servo_profile.acceleration_raw(self._config_yaml)  # reject a bad value before the hand moves
        if max_current_ma is not None and (isinstance(max_current_ma, bool) or not 0 < float(max_current_ma) <= 5000):
            raise ValueError(f"max_current_ma must be a number of mA from 1 to 5000, got {max_current_ma!r}")
        # A hand with safety_max_current_ma in its config.yaml (the baby hand) never gets more than that
        ceiling = self._config_yaml.get("safety_max_current_ma")
        if max_current_ma is not None and ceiling is not None and int(max_current_ma) > int(ceiling):
            print(f"max_current_ma {int(max_current_ma)} is above this hand's safety_max_current_ma: using {int(ceiling)}")
            max_current_ma = int(ceiling)
        self.max_current_ma = None if max_current_ma is None else int(max_current_ma)
        # calibration_current_ma: the current every joint is driven onto its hardstops with in this run's calibration,
        # instead of the config's calibration_current (None: the config's); limited like max_current_ma
        if calibration_current_ma is not None:
            if isinstance(calibration_current_ma, bool) or not 0 < float(calibration_current_ma) <= 5000:
                raise ValueError(f"calibration_current_ma must be a number of mA from 1 to 5000, got {calibration_current_ma!r}")
            if ceiling is not None and int(calibration_current_ma) > int(ceiling):
                print(f"calibration_current_ma {int(calibration_current_ma)} is above this hand's safety_max_current_ma: "
                      f"using {int(ceiling)}")
                calibration_current_ma = int(ceiling)
        self.calibration_current_ma = None if calibration_current_ma is None else int(calibration_current_ma)
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
        if self.calibration_current_ma is None:
            # Calibrates automatically when no calibration exists; moves to the config's neutral pose.
            self._hand.init_joints(force_calibrate=calibrate, move_to_neutral=initialize)
        else:
            # This run's calibration current on every joint (the wrist too, unless config.yaml sets its own).
            # orca_core needs max_current >= calibration_current and a retry current above it (the retry only runs
            # for joints with a joint_motor_travel baseline): both raised for the calibration, then the config's
            # values are back before the hand moves to neutral.
            config = self._hand.config
            current = self.calibration_current_ma
            retry = config.calibration_retry_current
            self._hand.config = dataclasses.replace(
                config, calibration_current=current, max_current=max(int(config.max_current), current),
                calibration_retry_current=None if retry is None else max(int(retry), current + 100),
                **({} if self._config_yaml.get("wrist_calibration_current") is not None
                   else dict(wrist_calibration_current=current)))
            try:
                self._hand.init_joints(force_calibrate=calibrate, move_to_neutral=False)
            finally:
                self._hand.config = config
            self._hand.set_max_current(config.max_current)
            if initialize:  # as orca_core's init_joints(move_to_neutral=True)
                self._hand.set_joint_positions(OrcaJointPositions.from_dict(config.neutral_position),
                                               num_steps=NUM_STEPS)
                self._hand._settle_neutral_move()
        # The same servo acceleration ramp as the chirp recordings the sim was fit to (init_joints writes orca_core's own).
        self.servo_profile = servo_profile.apply(self._hand, self._config_yaml)
        if self.max_current_ma is not None:  # clamped by orca_core to each motor's hardware ceiling
            self._hand.set_max_current(self.max_current_ma)
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
