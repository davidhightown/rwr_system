#!/usr/bin/env bash
# eval_policies.py for the policies with the cube pose (models_extero/), with the camera calibration and the
# hand-to-tag pose of our setup filled in. Every other option of eval_policies.py works as usual and is passed on;
# a later --hand-to-tag, --calibration or --speed overrides the values below.
#
#   experiments/rl_inference/eval_extero.sh extero_meerkat         # evaluate models_extero/extero_meerkat_policy.pt
#   experiments/rl_inference/eval_extero.sh --camera-only          # only the camera: prints the cube pose the policy
#                                                                  # would get; cube in the palm ~ (0, 0, -0.05)
#   experiments/rl_inference/eval_extero.sh extero_meerkat --wandb # and upload the run (without --wandb: no upload)
#
# Give a model name: without one, eval_policies.py switches to --auto, which uploads to W&B.
#
# Calibration: $RWR_CALIBRATION, default ~/camera_extrinsics.npy (apriltag_calibration.py --tag-id 9 --tag-size 0.08).
# Redo it whenever the camera or the tag moved.
# Hand to tag (m, deg): measured 2026-11-12 (sketch: tag 400 mm from the stand, mount 262 - 28 sin 30 mm high, forearm
# 30 deg down), yaw checked with show_tag_axes.py, y corrected to -0.04 from the cube in the palm in RViz.
# Speed: 1.5 rad/s, in the training range of both extero models (meerkat 1.5-2.2, chameleon up to 1.5) and the speed
# their parents ibex and crow ran at on the hand.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
CALIBRATION="${RWR_CALIBRATION:-$HOME/camera_extrinsics.npy}"
if [[ ! -f "$CALIBRATION" ]]; then
    echo "Camera calibration not found: $CALIBRATION (run apriltag_calibration.py, or set RWR_CALIBRATION)" >&2
    exit 1
fi
exec python3 "$HERE/eval_policies.py" \
    --models-dir "$REPO/models_extero" \
    --calibration "$CALIBRATION" \
    --hand-to-tag 0.0 -0.04 0.423 60 0 0 \
    --speed 1.5 \
    "$@"
