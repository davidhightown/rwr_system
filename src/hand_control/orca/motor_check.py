"""
Check the motor -> joint map (which motor moves which joint, and in which direction) of an ORCA hand,
one motor at a time. Run it after tensioning and before the first calibration; it needs no calibration.

For each motor the script names the joint to watch and moves only that motor, twice:
MOVE 1 (motor angle increasing) and back, then MOVE 2 (the opposite way) and back. It then asks about one
of the moves, MOVE 1 unless that one was blocked, and you answer with a single key. All other motors stay
without torque.

Your answers are saved to ~/orca_motor_check/<hand>_<time>.yaml. At the end, the script lists every motor
that moved a wrong joint and the sign fixes for joint_to_motor_map in config.yaml. Directions (ORCA
convention, positive joint angle): bending joints bend, sideways joints move towards the pinky side. The
thumb base, thumb sideways and the wrist are only recorded, since their positive direction is not checked.

Stop the hand controller node first: both need the hand's serial port.

    python motor_check.py --hand baby                 # every motor except the wrist
    python motor_check.py --hand baby --motors 15 4   # only these motors
    python motor_check.py --hand baby --wrist         # the wrist too (the whole hand swings)
"""
import argparse
import csv
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


FINGER_NAMES = {"thumb": "thumb", "index": "index finger", "middle": "middle finger", "ring": "ring finger",
                "pinky": "pinky"}
PART_NAMES = {"abd": "sideways", "mcp": "knuckle (base joint)", "pip": "middle joint", "dip": "tip joint",
              "cmc": "base"}
# The two answers per joint: (text for key 1, text for key 2, joint direction of answer 1: +1 positive, None unknown)
BEND = ("bent / closed", "straightened / opened", 1)
SIDEWAYS = {  # named by the neighbouring finger; positive = towards the pinky side of the hand
    "index": ("towards the MIDDLE finger", "away from the middle finger, towards the thumb", 1),
    "middle": ("towards the RING finger", "towards the INDEX finger", 1),
    "ring": ("towards the PINKY", "towards the MIDDLE finger", 1),
    "pinky": ("away from the ring finger (outwards)", "towards the RING finger", 1),
}
THUMB_BASE = ("towards the palm / the index finger", "away from the palm", None)
WRIST = ("tilted towards the palm", "tilted towards the back of the hand", None)
OTHER = [("n", "nothing moved / not sure"), ("w", "a DIFFERENT joint moved"), ("r", "show both moves again"),
         ("s", "skip this motor"), ("q", "quit (the answers so far are saved)")]
BLOCKED_FRACTION = 0.4  # a move that gets less than this share of --amplitude counts as blocked
BG = {"ok": "\033[42;30m", "bad": "\033[41;97m", "info": "\033[44;97m"}
BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"


def load_tendons(hand_name):
    """Motor -> (code, top-spool tendon, bottom-spool tendon) from the hand's tendon_mapping.csv, if it has one."""
    path = hands.CONFIGS_DIR / hands.cfg_name(hand_name) / "tendon_mapping" / "tendon_mapping.csv"
    if not path.exists():
        return {}
    spools = {}
    for row in csv.DictReader(path.open()):
        spools.setdefault(int(row["motor_id"]), {})[row["spool"]] = f"{row['nr']} {row['function']}"
    return {motor: (s["top"].split()[1].rsplit("-", 1)[0], s["top"], s["bottom"]) for motor, s in spools.items()}


def answers_for(joint):
    finger, part = joint.split("_")[0], joint.split("_")[-1]
    if joint == "wrist":
        return WRIST
    if finger == "thumb" and part in ("abd", "cmc"):
        return THUMB_BASE
    if part == "abd":
        return SIDEWAYS[finger]
    return BEND


def joint_label(joint):
    if joint == "wrist":
        return "the wrist"
    finger, part = joint.split("_")[0], joint.split("_")[-1]
    return f"the {FINGER_NAMES.get(finger, finger)} {PART_NAMES.get(part, part)}"


def getkey(allowed):
    """One key press without Enter (a line when stdin is not a terminal)."""
    if not sys.stdin.isatty():
        while True:
            line = sys.stdin.readline()
            if not line:
                raise KeyboardInterrupt
            key = line.strip().lower()[:1] or "\n"  # an empty line is Enter
            if key in allowed:
                return key
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
    peak = 0.0
    t0 = time.time()
    while True:
        s = min((time.time() - t0) / duration, 1.0)
        hand.write_motor_pos([motor], np.array([start + (target - start) * (1 - math.cos(math.pi * s)) / 2]))
        time.sleep(0.02)
        state = hand.get_motor_state()
        if state.current is not None:
            peak = max(peak, abs(float(state.current[index])))
        if s >= 1.0:
            break
    time.sleep(0.15)  # let it settle against the tendons
    return float(hand.get_motor_state().position[index]), peak


def show_moves(hand, motor, amplitude, speed):
    """MOVE 1 (+) and back, MOVE 2 (-) and back; returns the motor degrees each move reached."""
    start = float(hand.get_motor_pos(as_dict=True)[motor])
    hand.write_motor_pos([motor], np.array([start]))
    hand.enable_torque([motor])
    reached = {}
    try:
        for number, direction in ((1, 1), (2, -1)):
            print(f"   {BOLD}MOVE {number}{RESET}{' (the opposite way)' if number == 2 else ''} ...", end="", flush=True)
            pos, peak = move(hand, motor, start, start + direction * amplitude, speed)
            reached[number] = math.degrees(pos - start)
            time.sleep(0.6)  # hold, so the end position is easy to see
            back, _ = move(hand, motor, pos, start, speed)
            blocked = abs(reached[number]) < BLOCKED_FRACTION * math.degrees(amplitude)
            print(f" {DIM}motor {reached[number]:+.0f} deg, {peak:.0f} mA{RESET}"
                  + (f"  {BOLD}blocked{RESET}" if blocked else ""))
            time.sleep(0.8)
    finally:
        hand.disable_torque([motor])
    return reached


def intro(args, current, count):
    print(f"""
{BOLD}Motor check, {args.hand} hand{RESET}: {count} motors, one at a time.

For each motor:
  1. The script names the joint to watch. Look at it.
  2. The motor moves twice: {BOLD}MOVE 1{RESET} and back, then {BOLD}MOVE 2{RESET} (the opposite way) and back.
  3. It asks about one of the two moves, normally MOVE 1, and says which one.
  4. Answer with one key (no Enter): {BOLD}1{RESET} or {BOLD}2{RESET} for the direction it shows you,
     {BOLD}n{RESET} nothing moved / not sure, {BOLD}w{RESET} a different joint moved, {BOLD}r{RESET} show both moves again,
     {BOLD}s{RESET} skip, {BOLD}q{RESET} quit.

Only the motor being checked has torque ({current} mA, {args.amplitude:g} deg of motor shaft each way); the rest of
the hand is limp. If you are unsure, press r rather than guessing. Ctrl+C stops at any time.
""")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    hands.add_hand_argument(p, default="baby")
    p.add_argument("--motors", type=int, nargs="+", help="Only these motor IDs (default: all but the wrist).")
    p.add_argument("--wrist", action="store_true", help="Also check the wrist motor.")
    p.add_argument("--amplitude", type=float, default=45.0,
                   help="Motor-shaft move each way in degrees (default 45; max 120).")
    p.add_argument("--speed", type=float, default=40.0, help="Motor-shaft speed in deg/s (default 40; max 90).")
    p.add_argument("--current", type=int, default=700,
                   help="Current limit of the moving motor in mA (default 700, what moved the tightened baby hand; "
                        "limited to the hand's safety_max_current_ma).")
    args = p.parse_args()
    if not 0 < args.amplitude <= 120:
        p.error("--amplitude: from 0 to 120 deg")
    if not 0 < args.speed <= 90:
        p.error("--speed: from 0 to 90 deg/s")
    if args.current <= 0:
        p.error("--current: a current in mA above 0")

    config_path = hands.config_path(args.hand)
    current = hands.clamp_current(args.current, yaml.safe_load(config_path.read_text()), "--current")
    hand = load_hand(config_path=str(config_path))
    motor_to_joint = hand.config.motor_to_joint_dict
    wrist_motor = abs(hand.config.joint_to_motor_map["wrist"])
    motors = args.motors or [m for m in hand.config.motor_ids if m != wrist_motor]
    if args.wrist and wrist_motor not in motors:
        motors = [wrist_motor] + list(motors)
    unknown = [m for m in motors if m not in motor_to_joint]
    if unknown:
        p.error(f"motors not in {config_path}: {unknown}")
    tendons = load_tendons(args.hand)

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
        intro(args, current, len(motors))
        print(f"Press {BOLD}Enter{RESET} to start.")
        getkey(["\n", "\r", " "])

        i = 0
        while i < len(motors):
            motor = motors[i]
            joint = motor_to_joint[motor]
            answer1, answer2, plus_dir = answers_for(joint)
            code, top, bottom = tendons.get(motor, ("", "", ""))
            config_sign = -1 if hand.config.joint_inversion_dict.get(joint) else 1
            print(f"\n{BG['info']} {i + 1}/{len(motors)}  motor {motor} {RESET}  Watch {BOLD}{joint_label(joint)}{RESET}"
                  f"  {DIM}({joint}" + (f"; tendons {code}: top {top}, bottom {bottom}" if code else "") + f"){RESET}")
            time.sleep(1.0)  # time to look at the joint
            reached = show_moves(hand, motor, amplitude, args.speed)

            # ask about MOVE 1 unless it was blocked and MOVE 2 was not; warn when both were small
            limit = BLOCKED_FRACTION * args.amplitude
            asked = 2 if abs(reached[1]) < limit <= abs(reached[2]) else 1
            if asked == 2:
                print(f"   MOVE 1 was blocked, so describe {BOLD}MOVE 2{RESET}.")
            elif abs(reached[2]) < limit and abs(reached[1]) < limit:
                print(f"   {BOLD}Both moves were small{RESET} (the joint may sit near an end, or the tendons are tight): "
                      "answer only if you saw it clearly, otherwise r (again) or n.")
            print(f"   In {BOLD}MOVE {asked}{RESET}, {joint_label(joint)} went:")
            print(f"     {BOLD}1{RESET}  {answer1}\n     {BOLD}2{RESET}  {answer2}")
            print("     " + "   ".join(f"{BOLD}{k}{RESET} {t}" for k, t in OTHER))
            key = getkey(["1", "2"] + [k for k, _ in OTHER])
            if key == "q":
                break
            if key == "r":
                continue
            if key == "s":
                print("   skipped")
                i += 1
                continue

            moved, answer, joint_dir = joint, None, None
            if key == "n":
                moved = "nothing"
            elif key == "w":
                print("   Which joint moved? Type it (e.g. 'ring knuckle', 'middle sideways') and press Enter:")
                moved = input("   > ").strip() or "unknown"
            else:
                answer = answer1 if key == "1" else answer2
                joint_dir = None if plus_dir is None else (plus_dir if key == "1" else -plus_dir)
            motor_dir = 1 if asked == 1 else -1  # MOVE 1 turned the motor +, MOVE 2 -
            needed = None if joint_dir is None else joint_dir * motor_dir  # the sign joint_to_motor_map needs
            results["motors"][motor] = {
                "expected_joint": joint, "tendons": code or None, "moved": moved, "answer": answer,
                "move_described": asked, "motor_moved_deg": {f"move_{n}": round(d, 1) for n, d in reached.items()},
                "config_sign": config_sign, "sign_needed": needed}
            if moved == "nothing":
                print(f"   {BG['bad']} NOTHING MOVED {RESET} slack or detached tendon? Try again with a higher --current.")
            elif moved != joint:
                print(f"   {BG['bad']} WRONG JOINT {RESET} motor {motor} moved {moved}, not {joint}")
            elif needed is None:
                print(f"   {BG['ok']} right joint {RESET} (direction noted: {answer})")
            elif needed == config_sign:
                print(f"   {BG['ok']} right joint, right direction {RESET}")
            else:
                print(f"   {BG['bad']} INVERTED {RESET} right joint, but config.yaml needs {joint}: "
                      f"{'-' if needed < 0 else ''}{motor}")
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
                    status = f"WRONG JOINT: {r['moved']}"
                elif r["sign_needed"] is None:
                    status = f"right joint ({r['answer']})"
                elif r["sign_needed"] == r["config_sign"]:
                    status = "ok"
                else:
                    status = "INVERTED"
                    fixes.append(f"  {r['expected_joint']}: {'-' if r['sign_needed'] < 0 else ''}{motor}")
                print(f"  motor {motor:2d}  {r['expected_joint']:<11} {status}")
            if fixes:
                print("Sign fixes for joint_to_motor_map in config.yaml:\n" + "\n".join(fixes))


if __name__ == "__main__":
    main()
