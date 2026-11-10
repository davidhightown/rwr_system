#!/bin/bash
# Run the real-hand evaluation with one command, from any folder. All arguments go to evaluate_real_policies.py:
#   run_real_evaluation.sh --models 06 20
# The ROS workspace is the folder that contains src/rwr_system (set ROS_WS to use another one).
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="${ROS_WS:-$(cd "$HERE/../../../.." && pwd)}"
PIXI="$(command -v pixi || echo "$HOME/.pixi/bin/pixi")"
if [ ! -x "$PIXI" ]; then
    echo "pixi not found: install it (https://pixi.sh) or add it to the PATH" >&2
    exit 1
fi
cd "$WS"
exec "$PIXI" run python "$HERE/evaluate_real_policies.py" "$@"
