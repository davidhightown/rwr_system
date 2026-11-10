"""Offline tests for check_clearance.py; never open serial ports or move a physical hand."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/orca"))
import check_clearance  # noqa: E402


class Hand:
    """Joints follow the last command, except the ones in ``stuck`` ({joint: angle}), which stay put."""
    calibrated = True

    def __init__(self, stuck=None):
        self.config = type("Config", (), {"motor_ids": list(range(1, 18))})
        self.stuck = dict(stuck or {})
        self.position = dict(self.stuck)
        self.events = []

    def is_calibrated(self):
        return True

    def connect(self, **kwargs):
        return True, "fake"

    def init_joints(self, **kwargs):
        pass

    def set_servo_profile(self, profiles):
        self.profiles = profiles

    def get_servo_profile(self):
        return self.profiles

    def set_joint_positions(self, command, **kwargs):
        for joint, value in command.items():
            if joint not in self.stuck:
                self.position[joint] = value

    def get_motor_state(self):
        return type("State", (), {"position": dict(self.position)})

    def _motor_to_joint_pos(self, position):
        return position

    def disable_torque(self):
        self.events.append("disable")

    def disconnect(self):
        self.events.append("disconnect")


def run(hand, *extra):
    return check_clearance.main(["--execute", "--hold", "0", *extra],
                                factory=lambda **kwargs: hand, sleep=lambda s: None)


def test_preview_lists_the_poses_without_a_hand(capsys):
    assert check_clearance.main([]) == 0
    output = capsys.readouterr().out
    assert "ring_abd: center" in output and "Preview only" in output


def test_neighbours_that_follow_pass():
    hand = Hand()
    assert run(hand) == 0
    assert hand.events == ["disable", "disconnect"]


def test_a_neighbour_stuck_at_neutral_is_reported(capsys):
    hand = Hand(stuck={"middle_abd": -4.0})  # neutral
    assert run(hand, "--joints", "ring_abd") == 1
    output = capsys.readouterr().out
    assert "middle_abd" in output and "stayed near neutral" in output
