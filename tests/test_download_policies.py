"""Which W&B artifact download_policies.py takes for each model (without W&B)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/rl_inference"))

from download_policies import model_of, pick  # noqa: E402


def test_model_names():
    assert model_of("policy-2026-11-11_12-51-32_walrus") == ("walrus", (1, "2026-11-11_12-51-32"))
    assert model_of("policy-2026-11-10_13-03-14_cygnet_hist")[0] == "cygnet_hist"
    assert model_of("capybara-policy") == ("capybara", (0, ""))
    assert model_of("run-58bn38a1-history") == (None, None)


def test_pick_prefers_run_folder_then_newest():
    collections = ["horse-policy", "policy-2026-11-06_11-45-24_horse", "falcon-policy",
                   "policy-2026-11-03_16-04-05_frog", "policy-2026-11-04_05-21-15_frog", "something-else"]
    assert pick(collections) == {"falcon": "falcon-policy", "frog": "policy-2026-11-04_05-21-15_frog",
                                 "horse": "policy-2026-11-06_11-45-24_horse"}
