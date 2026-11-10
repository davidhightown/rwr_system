"""ORCA v2 hand controller. Drives the hand through the orca_core package,
exposing the interface hand_control_node.py expects."""
import numpy as np
from orca_core import load_hand

# Feetech: read_position_velocity_current() sync-reads registers 56..70 (position, speed, load, voltage, temperature,
# moving, current), and every motor's reply carries its status byte. orca_core keeps only position, speed and current;
# the voltage and the status byte are taken from the same packet, without any extra bus traffic.
FEETECH_STATE_BLOCK = (56, 15)
FEETECH_PRESENT_POSITION, FEETECH_PRESENT_VOLTAGE = 56, 62  # voltage in 0.1 V


class OrcaHandV2:
    def __init__(self, config_path: str):
        # An explicit config_path skips orca_core's hardware autodetection.
        # calibration.yaml is read from and saved next to config_path.
        self._hand = load_hand(config_path=config_path)
        config = self._hand.config
        self.joint_ids = list(config.joint_ids)
        self.num_motors = len(self.joint_ids)
        joint_index = {joint: i for i, joint in enumerate(self.joint_ids)}
        # get_motor_state() arrays are in motor_ids order; these map them to joint_ids order.
        self._joint_order = np.empty(self.num_motors, dtype=np.intp)
        self._joint_signs = np.ones(self.num_motors)
        for motor_idx, motor_id in enumerate(config.motor_ids):
            joint = config.motor_to_joint_dict[motor_id]
            self._joint_order[joint_index[joint]] = motor_idx
            if config.joint_inversion_dict.get(joint, False):
                self._joint_signs[joint_index[joint]] = -1.0
        self._status_errors = np.zeros(self.num_motors, dtype=np.int64)  # motor_ids order, OR since take_status()
        self._status_voltage = np.full(self.num_motors, np.nan)
        self.status_available = None  # None: not tried yet, False: not a Feetech hand (see status_problem)
        self.status_problem = ""

    def init_joints(self, calibrate: bool = False, initialize: bool = True, compliant_test_mode: bool = False,
                    neutral_overrides: dict | None = None):
        ok, message = self._hand.connect(interactive=False)
        if not ok:
            raise RuntimeError(f"Could not connect to the ORCA hand: {message}")
        # Calibrates automatically when no calibration exists; moves to the config's neutral pose,
        # with the joint angles in neutral_overrides (degrees) replaced if given.
        if neutral_overrides:
            from orca_core.constants import NUM_STEPS
            from orca_core.joint_position import OrcaJointPositions

            self._hand.init_joints(force_calibrate=calibrate, move_to_neutral=False)
            if initialize:
                pose = dict(self._hand.config.neutral_position)
                pose.update(neutral_overrides)
                self._hand.set_joint_positions(OrcaJointPositions.from_dict(pose), num_steps=NUM_STEPS)
        else:
            self._hand.init_joints(force_calibrate=calibrate, move_to_neutral=initialize)
        if compliant_test_mode:
            self.disable_torque()

    def set_joint_pos_array(self, joint_pos_deg):
        """Command joint positions in joint_ids order, in degrees (clamped to the ROMs by orca_core)."""
        self._hand.set_joint_positions(np.asarray(joint_pos_deg, dtype=np.float64))

    def read_joint_state(self):
        """One bus read -> (positions [deg], velocities [deg/s], currents [mA]) in joint_ids order.

        The current comes with the same bus transaction; its sign follows the joint direction like the velocity."""
        state = self._hand.get_motor_state()
        joint_pos = self._hand._motor_to_joint_pos(state.position)
        pos = np.array([np.nan if joint_pos[j] is None else joint_pos[j] for j in self.joint_ids])
        ratios = self._hand.calibration.joint_to_motor_ratios_dict  # motor rad per joint deg
        motor_ids = self._hand.config.motor_ids
        motor_ratios = np.array([ratios[motor_ids[i]] or np.nan for i in self._joint_order])
        vel = self._joint_signs * np.asarray(state.velocity)[self._joint_order] / motor_ratios
        cur = self._joint_signs * np.asarray(state.current, dtype=np.float64)[self._joint_order]
        return pos, vel, cur

    def collect_status(self):
        """After read_joint_state(): the status byte (OR-ed until take_status()) and the supply voltage of every motor,
        from the packet that was just received. Feetech only; switches itself off for other motors."""
        if self.status_available is False:
            return
        try:
            reader = self._hand._motor_client._sync_reads.get(FEETECH_STATE_BLOCK)
            if reader is None:
                raise LookupError("no Feetech state sync read (another motor family?)")
            for i, motor_id in enumerate(self._hand.config.motor_ids):
                available, error = reader.isAvailable(motor_id, FEETECH_PRESENT_POSITION, 2)
                if available:
                    self._status_errors[i] |= int(error)
                    self._status_voltage[i] = 0.1 * reader.getData(motor_id, FEETECH_PRESENT_VOLTAGE, 1)
            self.status_available = True
        except Exception as e:
            self.status_available, self.status_problem = False, f"{type(e).__name__}: {e}"

    def take_status(self):
        """-> (supply voltages [V], status bytes OR-ed since the last call) in joint_ids order, or None."""
        if not self.status_available:
            return None
        voltage = self._status_voltage[self._joint_order].copy()
        errors = self._status_errors[self._joint_order].copy()
        self._status_errors[:] = 0
        return voltage, errors

    def read_motor_temperature(self):
        """One bus read -> motor temperatures [°C] in joint_ids order."""
        return np.asarray(self._hand.get_motor_temp(), dtype=np.float32)[self._joint_order]

    def enable_torque(self):
        self._hand.enable_torque()

    def disable_torque(self):
        self._hand.disable_torque()

    def disconnect(self):
        self._hand.disconnect()
