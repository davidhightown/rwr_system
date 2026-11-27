"""The ORCA hands this repository drives and where their configs are.

--hand v2    the full-size ORCA v2 hand (configs/orcahand_v2)
--hand baby  the scaled-down ORCA v2 of the task-x baby hand (configs/orcahand_baby): its own tendon routing,
             motor map and lower currents

Scripts take --hand and get the config folder name (the hand controller node's hand_controller/hand_cfg_name)
or the config.yaml path from here.
"""
import argparse
from pathlib import Path

import yaml

CONFIGS_DIR = Path(__file__).resolve().parent / "configs"
HANDS = {"v2": "orcahand_v2", "baby": "orcahand_baby"}
DEFAULT_HAND = "v2"
# config.yaml key (read by rwr_system, not orca_core): the highest goal current (mA) any script may set on this
# hand. Overrides above it (e.g. a demo yaml's max_current) are clamped to it.
SAFETY_KEY = "safety_max_current_ma"


def cfg_name(hand: str) -> str:
    """The config folder name of a hand: --hand baby -> orcahand_baby."""
    if hand not in HANDS:
        raise ValueError(f"unknown hand {hand!r}; one of {', '.join(HANDS)}")
    return HANDS[hand]


def config_path(hand: str, side: str = "right", configs_dir: Path = CONFIGS_DIR) -> Path:
    """The config.yaml of a hand in the source tree (or in configs_dir, e.g. the installed share folder)."""
    return Path(configs_dir) / cfg_name(hand) / side / "config.yaml"


def safety_max_current(config: dict):
    """The hand's current ceiling in mA from its config.yaml contents, or None when it has none."""
    value = config.get(SAFETY_KEY)
    return None if value is None else int(value)


def clamp_current(max_current_ma, config: dict, what: str = "max_current"):
    """max_current_ma (mA or None: the config's) limited to the hand's safety ceiling, with a warning."""
    ceiling = safety_max_current(config)
    if ceiling is None or max_current_ma is None or int(max_current_ma) <= ceiling:
        return max_current_ma
    print(f"\033[93m{what} {int(max_current_ma)} mA is above this hand's {SAFETY_KEY} ({ceiling} mA): "
          f"using {ceiling} mA\033[0m")
    return ceiling


def load_config(hand: str, side: str = "right") -> dict:
    return yaml.safe_load(config_path(hand, side).read_text())


def add_hand_argument(parser: argparse.ArgumentParser, default: str = DEFAULT_HAND):
    parser.add_argument("--hand", choices=sorted(HANDS), default=default,
                        help=f"Which ORCA hand: v2 (full size) or baby (scaled down; own motor map and lower "
                             f"currents). Default {default}.")
