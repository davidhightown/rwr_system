"""Which W&B artifact download_policies.py takes for each model (without W&B)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/rl_inference"))

from download_policies import model_of, pick, training_summary  # noqa: E402


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


def test_training_summary_of_a_wandb_run_config():
    observations = dict(policy=dict(joint_pos=dict(func="a", history_length=5), joint_vel=dict(func="b", history_length=0),
                                    history_length=None),
                        critic=dict(object_quat=dict(func="c")))
    config = dict(env_cfg=dict(actions=dict(joint_pos=dict(alpha=0.5)), observations=observations,
                               commands=dict(continuous_rotation=dict(axis_probs=[0.1, 0.1, 0.8],
                                                                      angular_vel_range=[1, 1, 1.5], debug_vis=False))),
                  runner_cfg=dict(obs_groups=dict(policy=["policy"], critic=["critic"])))
    assert training_summary(config) == dict(
        action_alpha=0.5, actor_obs=["policy"], history=dict(joint_pos=5),
        commands=dict(continuous_rotation=dict(axis_probs=[0.1, 0.1, 0.8], angular_vel_range=[1, 1, 1.5])))
    config["runner_cfg"]["obs_groups"]["policy"] = ["critic"]  # a teacher
    assert training_summary(config)["actor_obs"] == ["critic"] and training_summary(config)["history"] == {}
    assert training_summary({}) == {}
