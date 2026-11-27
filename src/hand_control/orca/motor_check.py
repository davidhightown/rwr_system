"""
Check the motor -> joint map of an ORCA hand, one motor at a time, before the first calibration.

For each motor the script names the joint that should move and turns only that motor once (+, i.e. motor
angle increasing; if that is blocked, the other way), holds it there and asks with a single key what you
see. Then the motor goes back and the next one starts. All other motors stay without torque.

Your answers go to ~/orca_motor_check/<hand>_<time>.yaml, with the sign each one implies for
joint_to_motor_map. Positive joint angles (ORCA convention): fingers and thumb bend (close); sideways
(abduction) towards the pinky side (the neutral pose fans the index towards the thumb, the pinky outwards).

Needs no calibration. Stop the hand controller node first: both need the hand's serial port.

    python motor_check.py --hand baby                 # all motors except the wrist
    python motor_check.py --hand baby --motors 15 4   # only these
    python motor_check.py --hand baby --wrist         # the wrist too (the whole hand swings)
"""
import argparse
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from orca_core import load_hand

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hands  # noqa: E402

# Baby hand tendons per motor, from the team's Tendon Mapping table: (code, top spool, bottom spool)
BABY_TENDONS = {
    1: ("wrist", "-", "-"),
    2: ("I-DP", "23 I-DP-ext", "22 I-DP-flex"),
    3: ("I-PP", "24 I-PP-flex", "21 I-PP-ext"),
    4: ("T-AP", "17 T-AP-L", "20 T-AP-R"),
    5: ("T-DP", "18 T-DP-flex", "19 T-DP-ext"),
    6: ("M-AP", "25 M-AP-R", "26 M-AP-L"),
    7: ("R-AP", "32 R-AP-L", "27 R-AP-R"),
    8: ("R-DP", "31 R-DP-ext", "28 R-DP-flex"),
    9: ("R-PP", "30 R-PP-flex", "29 R-PP-ext"),
    10: ("P-PP", "4 P-PP-ext", "3 P-PP-flex"),
    11: ("P-DP", "5 P-DP-flex", "2 P-DP-ext"),
    12: ("P-AP", "6 P-AP-R", "1 P-AP-L"),
    13: ("M-PP", "7 M-PP-flex", "8 M-PP-ext"),
    14: ("M-DP", "10 M-DP-flex", "9 M-DP-ext"),
    15: ("I-AP", "11 I-AP-L", "12 I-AP-R"),
    16: ("T-PP", "16 T-PP-flex", "13 T-PP-ext"),
    17: ("T-TP", "15 T-TP-flex", "14 T-TP-ext"),
}

FINGER_NAMES = {"thumb": "Thumb", "index": "Index finger", "middle": "Middle finger", "ring": "Ring finger",
                "pinky": "Pinky"}
PART_NAMES = {"abd": "sideways", "mcp": "knuckle (base joint)", "pip": "middle joint", "dip": "tip joint",
              "cmc": "base (rotation towards the palm)", "wrist": "tilt"}

# The question per kind of joint: (key, answer text, joint direction: +1 = positive joint angle, None = unknown)
QUESTIONS = {
    "bend": [("c", "bent / closed", 1), ("o", "straightened / opened", -1)],
    "side": [("p", "towards the PINKY side", 1), ("t", "towards the THUMB side", -1)],
    "thumb": [("c", "closed: towards the palm / the fingers", None), ("o", "opened: away from the palm", None)],
    "wrist": [("c", "tilted towards the palm", None), ("o", "tilted towards the back of the hand", None)],
}
OTHER = [("n", "nothing moved", None), ("w", "a DIFFERENT joint moved", None), ("r", "repeat the move", None),
         ("s", "skip this motor", None), ("q", "quit", None)]
BG = {"ok": "\033[42;30m", "bad": "\033[41;97m", "info": "\033[44;97m"}
BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"


def kind_of(joint):
    finger, part = joint.split("_")[0], joint.split("_")[-1]
    if joint == "wrist":
        return "wrist"
    if finger == "thumb" and part in ("abd", "cmc"):
        return "thumb"
    return "side" if part == "abd" else "bend"


def joint_label(joint):
    if joint == "wrist":
        return "Wrist tilt"
    finger, part = joint.split("_")[0], joint.split("_")[-1]
    return f"{FINGER_NAMES.get(finger, finger)} {PART_NAMES.get(part, part)}"


def getkey(allowed):
    """One key press without Enter (falls back to a line when stdin is not a terminal)."""
    if not sys.stdin.isatty():
        while True:
            line = sys.stdin.readline()
            if not line:
                raise KeyboardInterrupt
            if line.strip().lower()[:1] in allowed:
                return line.strip().lower()[:1]
    import termios
    import tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            ch = sys.stdin.read(1).lower()
            if ch == "\x03":
                raise KeyboardInterrupt
            if ch in allowed:
                return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def move(hand, motor, start, target, speed_deg_s):
    """Half-cosine move of one motor from start to target (rad); returns where it got (rad) and the peak mA."""
    index = hand.config.motor_ids.index(motor)
    duration = max(abs(math.degrees(target - start)) / speed_deg_s, 0.2)
    peak, pos = 0.0, start
    t0 = time.time()
    while True:
        s = min((time.time() - t0) / duration, 1.0)
        hand.write_motor_pos([motor], np.array([start + (target - start) * (1 - math.cos(math.pi * s)) / 2]))
        time.sleep(0.02)
        state = hand.get_motor_state()
        pos = float(state.position[index])
        if state.current is not None:
            peak = max(peak, abs(float(state.current[index])))
        if s >= 1.0:
            break
    time.sleep(0.15)  # let it settle against the tendons
    return float(hand.get_motor_state().position[index]), peak


def swing_out(hand, motor, amplitude, speed):
    """Move + (or - when + is blocked) and hold there. Returns (start, direction used, reached deg, peak mA)."""
    start = float(hand.get_motor_pos(as_dict=True)[motor])
    hand.write_motor_pos([motor], np.array([start]))
    hand.enable_torque([motor])
    for direction in (1, -1):
        pos, peak = move(hand, motor, start, start + direction * amplitude, speed)
        reached = math.degrees(pos - start)
        if abs(reached) >= 0.4 * math.degrees(amplitude):
            return start, direction, reached, peak
        move(hand, motor, pos, start, speed)
        if direction > 0:
            print(f"   blocked this way ({reached:+.0f} deg, {peak:.0f} mA). {BOLD}Watch again{RESET}: "
                  "it now moves the other way...")
            time.sleep(1.0)
    return start, -1, reached, peak


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    hands.add_hand_argument(p, default="baby")
    p.add_argument("--motors", type=int, nargs="+", help="Only these motor IDs (default: all but the wrist).")
    p.add_argument("--wrist", action="store_true", help="Also move the wrist motor.")
    p.add_argument("--amplitude", type=float, default=45.0, help="Motor-shaft move in degrees (default 45; max 120).")
    p.add_argument("--speed", type=float, default=40.0, help="Motor-shaft speed in deg/s (default 40; max 90).")
    p.add_argument("--current", type=int, default=600,
                   help="Goal-current limit of the moving motor in mA (default 600; clamped to the hand's "
                        "safety_max_current_ma).")
    args = p.parse_args()
    if not 0 < args.amplitude <= 120:
        p.error("--amplitude: from 0 to 120 deg")
    if not 0 < args.speed <= 90:
        p.error("--speed: from 0 to 90 deg/s")

    config_path = hands.config_path(args.hand)
    config_yaml = yaml.safe_load(config_path.read_text())
    current = hands.clamp_current(args.current, config_yaml, "--current")
    hand = load_hand(config_path=str(config_path))
    motor_to_joint = hand.config.motor_to_joint_dict
    wrist_motor = abs(hand.config.joint_to_motor_map["wrist"])
    motors = args.motors or [m for m in hand.config.motor_ids if m != wrist_motor]
    if args.wrist and wrist_motor not in motors:
        motors = [wrist_motor] + list(motors)
    unknown = [m for m in motors if m not in motor_to_joint]
    if unknown:
        p.error(f"motors not in {config_path}: {unknown}")
    tendons = BABY_TENDONS if args.hand == "baby" else {}

    ok, message = hand.connect(interactive=False)
    print(message)
    if not ok:
        sys.exit(1)
    out_dir = Path.home() / "orca_motor_check"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"{args.hand}_{datetime.now():%Y%m%d_%H%M%S}.yaml"
    results = {"hand": args.hand, "config": str(config_path), "amplitude_deg": args.amplitude,
               "current_ma": current, "motors": {}}
    amplitude = math.radians(args.amplitude)
    try:
        hand.disable_torque()
        # The mode switch turns torque on for every motor (orca_core): first make every goal the present
        # position under the low current, so nothing jumps, then turn the torque off again.
        hand.set_max_current(current)
        hand.write_motor_pos(list(hand.config.motor_ids), np.asarray(hand.get_motor_pos(), dtype=float))
        hand.set_control_mode("current_based_position")
        hand.disable_torque()
        hand.set_max_current(current)
        print(f"\n{BOLD}Hand {args.hand}{RESET}: {len(motors)} motors, {args.amplitude:g} deg of motor shaft at "
              f"{args.speed:g} deg/s, {current} mA. Answer with one key (no Enter); Ctrl+C stops.\n")

        i = 0
        while i < len(motors):
            motor = motors[i]
            joint = motor_to_joint[motor]
            kind = kind_of(joint)
            code, top, bottom = tendons.get(motor, ("", "", ""))
            config_sign = -1 if hand.config.joint_inversion_dict.get(joint) else 1
            print(f"{BG['info']} {i + 1}/{len(motors)}  motor {motor} {RESET}  Watch: "
                  f"{BOLD}{joint_label(joint)}{RESET}  {DIM}({joint}{', ' + code + ': top ' + top + ', bottom ' + bottom if code else ''}){RESET}")
            time.sleep(1.0)  # time to look at the joint before it moves
            start, direction, reached, peak = swing_out(hand, motor, amplitude, args.speed)
            print(f"   {DIM}(motor {reached:+.0f} deg, peak {peak:.0f} mA; it holds here until you answer){RESET}")
            print(f"   {BOLD}The move you just saw:{RESET} which way did it go?")
            options = QUESTIONS[kind] + OTHER
            for key, text, _ in options:
                print(f"     {BOLD}{key}{RESET}  {text}")
            key = getkey([k for k, _, _ in options])
            hand_pos = float(hand.get_motor_pos(as_dict=True)[motor])
            move(hand, motor, hand_pos, start, args.speed)
            hand.disable_torque([motor])
            if key == "q":
                break
            if key == "r":
                continue
            if key == "s":
                print("   skipped\n")
                i += 1
                continue

            moved, answer, joint_dir = joint, dict((k, t) for k, t, _ in options)[key], None
            if key == "n":
                moved = "nothing"
            elif key == "w":
                print("   Which joint moved? Type it (e.g. 'ring knuckle', 'middle sideways') and Enter:")
                moved = input("   > ").strip() or "unknown"
                print(f"   and which way? {', '.join(f'{k} = {t}' for k, t, _ in QUESTIONS['bend'] + QUESTIONS['side'])}")
                answer = dict((k, t) for k, t, _ in QUESTIONS["bend"] + QUESTIONS["side"]).get(
                    getkey(["c", "o", "p", "t"]))
            else:
                joint_dir = dict((k, d) for k, _, d in options)[key]

            # positive motor rotation -> joint direction: the sign joint_to_motor_map needs
            needed = None if joint_dir is None else joint_dir * direction
            results["motors"][motor] = {
                "expected_joint": joint, "tendons": code or None, "moved": moved, "answer": answer,
                "motor_direction_tested": "+" if direction > 0 else "-", "motor_moved_deg": round(reached, 1),
                "peak_current_ma": round(peak), "config_sign": config_sign, "sign_needed": needed}
            if moved == "nothing":
                print(f"   {BG['bad']} NOTHING MOVED {RESET} motor {motor} (slack or detached tendon? try --current 700)\n")
            elif moved != joint:
                print(f"   {BG['bad']} WRONG JOINT {RESET} motor {motor}: {moved}, not {joint}\n")
            elif needed is None:
                print(f"   {BG['ok']} joint ok {RESET} (noted: {answer})\n")
            elif needed == config_sign:
                print(f"   {BG['ok']} joint and direction ok {RESET}\n")
            else:
                print(f"   {BG['bad']} INVERTED {RESET} joint ok, but the map needs {joint}: "
                      f"{'-' if needed < 0 else ''}{motor}\n")
            out_file.write_text(yaml.safe_dump(results, sort_keys=False))
            i += 1
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        try:
            hand.disable_torque()
        finally:
            hand.disconnect()
        if results["motors"]:
            out_file.write_text(yaml.safe_dump(results, sort_keys=False))
            print(f"\n{BOLD}Summary{RESET} (saved to {out_file})")
            fixes = []
            for motor, r in results["motors"].items():
                if r["moved"] == "nothing":
                    status = "NOTHING MOVED"
                elif r["moved"] != r["expected_joint"]:
                    status = f"WRONG: {r['moved']}"
                elif r["sign_needed"] is None:
                    status = f"ok ({r['answer']})"
                elif r["sign_needed"] == r["config_sign"]:
                    status = "ok"
                else:
                    status = "INVERTED"
                    fixes.append(f"  {r['expected_joint']}: {'-' if r['sign_needed'] < 0 else ''}{motor}")
                print(f"  motor {motor:2d}  {r['expected_joint']:<11} {status}")
            if fixes:
                print("Sign fixes for joint_to_motor_map:\n" + "\n".join(fixes))


if __name__ == "__main__":
    main()
