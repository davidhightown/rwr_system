"""
Tension the tendons of an ORCA hand: the motors wind the tendons taut and then
hold them under current while you adjust them. Press Ctrl+C when done.
Stop the hand controller node first: both need the hand's serial port.

The motors wind with the config's calibration_current and hold with its max_current (orca_core), so the tendons
are wound taut under the load calibration drives the joints into their end stops with. --wind-current and
--hold-current override them for one run.

    python tensioning_v2.py               # the full-size ORCA v2 hand
    python tensioning_v2.py --hand baby   # the baby hand
"""
import argparse
import dataclasses
import sys
from pathlib import Path

from orca_core import HandConfigValidationError, load_hand

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hands  # noqa: E402


def print_progress(event: dict):
    phase = event.get("phase") if event.get("event") == "phase" else None
    if phase == "winding":
        print("Winding tendons taut...")
    elif phase == "holding":
        print("Holding motors for manual tensioning - press Ctrl+C when done.")


def positive_ma(text):
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"a current in mA above 0, not {text}")
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    hands.add_hand_argument(parser)
    parser.add_argument("--config_dir", type=Path,
                        help="Directory containing the per-side config folders of the hand (default: the --hand "
                             "folder in configs/, e.g. configs/orcahand_v2). Takes precedence over --hand.")
    parser.add_argument("--left_hand", action="store_true", help="Use the left hand configuration (default is right hand).")
    parser.add_argument("--wind-current", type=positive_ma,
                        help="Winding current in mA (default: the config's calibration_current).")
    parser.add_argument("--hold-current", type=positive_ma,
                        help="Holding current in mA (default: the config's max_current).")
    args = parser.parse_args()

    config_dir = args.config_dir or hands.CONFIGS_DIR / hands.cfg_name(args.hand)
    config_path = config_dir / ("left" if args.left_hand else "right") / "config.yaml"
    # run_tension (orca_core) winds with calibration_current and holds with max_current
    currents = {"calibration_current": args.wind_current, "max_current": args.hold_current}

    hand = load_hand(config_path=str(config_path))
    try:
        hand.config = dataclasses.replace(hand.config, **{k: v for k, v in currents.items() if v is not None})
    except HandConfigValidationError as exc:  # e.g. a winding current above the holding current
        parser.error(f"{exc} (winding = calibration_current, holding = max_current)")

    ok, message = hand.connect(interactive=False)  # also resolves "default" currents to the motor family's
    print(message)
    if ok:
        print(f"{config_path}: winding at {hand.config.calibration_current} mA, "
              f"holding at {hand.config.max_current} mA")
        try:
            hand.tension(progress_callback=print_progress)
        except KeyboardInterrupt:
            print("\nTensioning stopped.")
        finally:
            hand.disconnect()
