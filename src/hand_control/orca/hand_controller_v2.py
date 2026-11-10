"""ORCA v2 hand controller. Drives the hand through the orca_core package,
exposing the interface hand_control_node.py expects."""
import numpy as np
from orca_core import load_hand


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
        """One bus read -> (positions [deg], velocities [deg/s]) in joint_ids order."""
        state = self._hand.get_motor_state()
        joint_pos = self._hand._motor_to_joint_pos(state.position)
        pos = np.array([np.nan if joint_pos[j] is None else joint_pos[j] for j in self.joint_ids])
        ratios = self._hand.calibration.joint_to_motor_ratios_dict  # motor rad per joint deg
        motor_ids = self._hand.config.motor_ids
        motor_ratios = np.array([ratios[motor_ids[i]] or np.nan for i in self._joint_order])
        vel = self._joint_signs * np.asarray(state.velocity)[self._joint_order] / motor_ratios
        return pos, vel

    def read_motor_temperature(self):
        """One bus read -> motor temperatures [°C] in joint_ids order."""
        return np.asarray(self._hand.get_motor_temp(), dtype=np.float32)[self._joint_order]

    def enable_torque(self):
        self._hand.enable_torque()

    def disable_torque(self):
        self._hand.disable_torque()

    def disconnect(self):
        self._hand.disconnect()
