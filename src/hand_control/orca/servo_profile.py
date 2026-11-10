"""The servo acceleration ramp of the ORCA v2 hand, applied explicitly at every bring-up.

The Feetech servos do not jump to a goal position: they run their own trajectory toward it, with an
acceleration limit in the ACC register (8.7 deg/s^2 of motor shaft per unit, 0 = no limit). At 20 Hz this
ramp is the dominant actuator dynamics: about 24 rad/s^2 at the finger joints with the orca_core default of
150. The register lives in RAM and orca_core writes its own default (150) inside ``init_joints``. This module
writes the value from the hand's ``config.yaml`` after ``init_joints`` and reads it back, so the chirp
recorder and the deployment controller always run the same, recorded ramp, whatever an orca_core update
changes. The simulation models the same ramp (rwr_faive_lab ``ORCA_V2_SERVO_ACCEL``): if you change
``servo_acceleration_raw``, record new chirps and refit before you deploy.

The goal-current limit (``max_current``, 900 mA by default) and the goal speed (orca_core's 1500, above
what the motors reach) are left to orca_core and only recorded.

No ROS and no NumPy, so the standalone chirp environment can import it.
"""

CONFIG_KEY = "servo_acceleration_raw"
DEFAULT_ACCELERATION_RAW = 150  # orca_core 0.5.2 DEFAULT_ACC for Feetech
MAX_ACCELERATION_RAW = 254
# Feetech HLS: 8.7 deg/s^2 per unit (orca_core HLS.ACC_SCALE_RAD_S2)
ACC_SCALE_RAD_S2 = 0.15184364492350666


def acceleration_raw(config):
    """The ACC register value from a config.yaml dict; ``DEFAULT_ACCELERATION_RAW`` if the key is absent."""
    value = config.get(CONFIG_KEY, DEFAULT_ACCELERATION_RAW)
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= MAX_ACCELERATION_RAW:
        raise ValueError(f"{CONFIG_KEY} must be an integer from 0 (no ramp) to {MAX_ACCELERATION_RAW}, "
                         f"got {value!r}")
    return value


def describe(config):
    """Plan entry: the ramp this config asks for."""
    raw = acceleration_raw(config)
    return {
        CONFIG_KEY: raw,
        "motor_acceleration_rad_s2": raw * ACC_SCALE_RAD_S2 if raw else None,  # None: no ramp
    }


def apply(hand, config):
    """Write the configured ramp to every motor, read all profiles back and return them for the record.

    Call after ``hand.init_joints()``, which overwrites the register. Raises if a motor does not report the
    value that was written, so a run never records data under a ramp other than the one in its metadata.
    """
    from orca_core.hardware.motor_client import ServoProfile

    raw = acceleration_raw(config)
    motor_ids = list(hand.config.motor_ids)
    hand.set_servo_profile({motor: ServoProfile(acceleration_rad_s2=raw * ACC_SCALE_RAD_S2)
                            for motor in motor_ids})
    readback = hand.get_servo_profile()
    record, wrong = {}, []
    for motor in motor_ids:
        profile = readback.get(motor)
        if profile is None:
            wrong.append(motor)
            record[str(motor)] = None
            continue
        read_raw = round(profile.acceleration_rad_s2 / ACC_SCALE_RAD_S2)
        if read_raw != raw:
            wrong.append(motor)
        record[str(motor)] = {"acceleration_raw": read_raw, "velocity_rad_s": profile.velocity_rad_s}
    if wrong:
        raise RuntimeError(f"Servo acceleration {raw} did not read back on motors {wrong}: {record}")
    return {**describe(config), "motors": record}
