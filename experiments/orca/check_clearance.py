"""Check on the hand that the other fingers really move out of the way of each abduction sweep.

For each joint, the hand ramps into the pose chirp.py uses during that joint's sweep (the other fingers at their
neighbour_pose, the tested joint at its sweep center), holds it, reads every joint back, and prints commanded
vs measured for the joints that should have moved. No chirp, about 4 s per joint. Takes the same flags as
chirp.py (--config, --joints, --abd-gap-deg, --abd-clear-all, ...); by default it checks the four finger
abduction joints.

    python experiments/orca/check_clearance.py --config <config.yaml>            # plan only
    python experiments/orca/check_clearance.py --config <config.yaml> --execute  # move the hand
"""

import math
import sys
import time

import yaml

import chirp

ABD_JOINTS = [f"{finger}_abd" for finger in chirp.FINGERS]
TOLERANCE_DEG = 3.0  # a neighbour further than this from its target did not get there


def make_parser():
    parser = chirp.make_parser()
    parser.description = __doc__
    parser.set_defaults(joints=ABD_JOINTS)
    parser.add_argument("--hold", type=float, default=2.0, help="Seconds to hold each pose before reading it.")
    return parser


def read_joints(hand):
    return hand._motor_to_joint_pos(hand.get_motor_state().position)


def check_joint(hand, plan, joint, ramp_steps, step_size, hold, *, sleep=time.sleep):
    """Ramp into joint's sweep pose, hold, and return rows (name, neutral, target, measured, ok)."""
    neutral = plan["config"]["neutral_position"]
    targets = dict(plan["neighbour_pose"].get(joint, {}))
    targets[joint] = plan["centers_deg"][joint]
    command = dict(neutral)
    command.update(targets)
    hand.set_joint_positions(command, num_steps=ramp_steps, step_size=step_size)
    sleep(hold)
    measured = read_joints(hand)
    rows = []
    for name, target in targets.items():
        value = measured.get(name)
        ok = value is not None and math.isfinite(value) and abs(value - target) <= TOLERANCE_DEG
        rows.append((name, neutral[name], target, value, ok))
    hand.set_joint_positions(dict(neutral), num_steps=ramp_steps, step_size=step_size)
    sleep(hold / 2)
    return rows


def print_rows(joint, rows):
    print(f"\n{joint} sweep pose")
    print(f"  {'joint':<12}{'neutral':>9}{'target':>9}{'measured':>10}")
    for name, neutral, target, value, ok in rows:
        shown = f"{value:10.1f}" if value is not None else f"{'none':>10}"
        note = "" if ok else ("  <-- stayed near neutral" if value is not None and abs(value - neutral) <= TOLERANCE_DEG
                              else "  <-- did not reach target")
        print(f"  {name:<12}{neutral:9.1f}{target:9.1f}{shown}{note}")


def main(argv=None, *, factory=None, sleep=time.sleep):
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        config = yaml.safe_load(args.config.read_text())
        plan = chirp.make_plan(args, config)
    except (OSError, ValueError, TypeError, KeyError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    for joint in args.joints:
        print(f"{joint}: center {plan['centers_deg'][joint]:g}, others {plan['neighbour_pose'].get(joint) or 'at neutral'}")
    if not args.execute:
        print("Preview only. Add --execute to connect and move the calibrated hand.")
        return 0

    if factory is None:
        from orca_core import load_hand
        factory = load_hand
    hand = factory(config_path=str(args.config.resolve()))
    if not hand.calibrated or not hand.is_calibrated():
        print("A complete calibration.yaml is required beside config.yaml.", file=sys.stderr)
        return 1
    ok, message = hand.connect(interactive=False)
    if not ok:
        print(f"Could not connect: {message}", file=sys.stderr)
        return 1
    failed = []
    try:
        hand.init_joints(force_calibrate=False, move_to_neutral=True)
        chirp.servo_profile.apply(hand, plan["config"])
        ramp_steps, step_size = max(2, math.ceil(args.ctrl_freq)), 1.0 / args.ctrl_freq
        for joint in args.joints:
            rows = check_joint(hand, plan, joint, ramp_steps, step_size, args.hold, sleep=sleep)
            print_rows(joint, rows)
            failed += [f"{joint}: {name}" for name, *_, ok in rows if not ok]
    except KeyboardInterrupt:
        print("Stopped.", file=sys.stderr)
        return 130
    finally:
        try:
            hand.disable_torque()
        finally:
            hand.disconnect()
    print(f"\nAll joints reached their targets within {TOLERANCE_DEG:g} deg." if not failed
          else f"\nOff target by more than {TOLERANCE_DEG:g} deg: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
