#!/usr/bin/env python3
"""Download the exported policies on W&B that are not yet in the models folders.

    python experiments/rl_inference/download_policies.py                 # both projects
    python experiments/rl_inference/download_policies.py --dry-run       # only list what is new
    python experiments/rl_inference/download_policies.py --only proprio  # one project

Training logs every exported policy as a W&B artifact of type "model": in rwr-skin/orca_proprio (proprioceptive task,
saved to models_proprio/) and rwr-skin/orca_extero (saved to models_extero/). The artifact is "policy-<run folder>"
(e.g. policy-2026-11-11_12-51-32_walrus), or "<name>-policy" for older runs; when a model has both, the first is taken.

A model is saved as <models folder>/<name>_policy.pt, the name eval_policies.py and W&B use for it, with
<name>_policy.json next to it (the artifact, its version and its metadata: training run, checkpoint, commit). A model
whose <name>_policy.pt is already there is left alone, so files copied in by hand are not overwritten.

Needs wandb and a login (`wandb login`).
"""

import argparse
import json
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from cube_eval.wandb_results import ENTITY, retry  # noqa: E402

PROJECTS = {"proprio": ("orca_proprio", "models_proprio"), "extero": ("orca_extero", "models_extero")}
RUN_FOLDER = re.compile(r"^policy-(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_(.+)$")
OLD_STYLE = re.compile(r"^(.+)-policy$")


def model_of(collection):
    """(model name, rank) of an artifact collection; a "policy-<run folder>" collection ranks above an old-style one,
    a newer run folder above an older one. (None, None) if the name is neither."""
    m = RUN_FOLDER.match(collection)
    if m:
        return m.group(2), (1, m.group(1))
    m = OLD_STYLE.match(collection)
    if m:
        return m.group(1), (0, "")
    return None, None


def pick(collections):
    """{model name: collection}, one collection per model, by name."""
    best = {}
    for c in collections:
        name, rank = model_of(c)
        if name and (name not in best or rank > best[name][1]):
            best[name] = (c, rank)
    return {name: c for name, (c, _) in sorted(best.items())}


def collections(api, path):
    """Names of the "model" artifact collections in a project ([] if it has none)."""
    types = retry(lambda: [t.name for t in api.artifact_types(project=path)])
    if "model" not in types:
        return []
    return retry(lambda: [c.name for c in api.artifact_collections(path, "model")])


def download(api, path, collection, target):
    """The artifact's policy.pt to target, its description to target's .json; returns the artifact's version."""
    artifact = retry(lambda: api.artifact(f"{path}/{collection}:latest", type="model"))
    with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
        file = retry(lambda: artifact.get_entry("policy.pt").download(root=tmp))
        info = dict(artifact=f"{path}/{collection}:{artifact.version}", aliases=artifact.aliases,
                    digest=artifact.digest, created=artifact.created_at,
                    downloaded=datetime.now(timezone.utc).isoformat(timespec="seconds"), metadata=artifact.metadata)
        target.with_suffix(".json").write_text(json.dumps(info, indent=1, default=str))
        os.replace(file, target)
    return artifact.version


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=list(PROJECTS), help="Only this project.")
    ap.add_argument("--dry-run", action="store_true", help="Only list the new policies, download nothing.")
    ap.add_argument("--entity", default=ENTITY)
    args = ap.parse_args(argv)
    try:
        import wandb
        api = wandb.Api(timeout=60)
        api.viewer  # noqa: B018 (fails here without a login)
    except Exception as e:
        raise SystemExit(f"W&B is not reachable ({type(e).__name__}: {e}).\nInstall it and log in once: "
                         "pip install wandb && wandb login")
    failed = 0
    for key in [args.only] if args.only else list(PROJECTS):
        project, folder = PROJECTS[key]
        path, models_dir = f"{args.entity}/{project}", REPO / folder
        try:
            models = pick(collections(api, path))
        except Exception as e:
            print(f"{path}: could not list the policies ({type(e).__name__}: {e})")
            failed += 1
            continue
        new = {name: c for name, c in models.items() if not (models_dir / f"{name}_policy.pt").exists()}
        print(f"{path}: {len(models)} policies, {len(models) - len(new)} already in {folder}/, {len(new)} new")
        for name, collection in new.items():
            target = models_dir / f"{name}_policy.pt"
            if args.dry_run:
                print(f"  new         {target.name:30s} {collection}")
                continue
            models_dir.mkdir(exist_ok=True)
            try:
                version = download(api, path, collection, target)
                print(f"  downloaded  {target.name:30s} {collection}:{version}")
            except Exception as e:
                print(f"  FAILED      {target.name:30s} {collection} ({type(e).__name__}: {e})")
                failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
