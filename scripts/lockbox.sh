#!/bin/sh
# Lock-box protocol (docs/evaluation_lockbox_plan.md): thin wrapper around `python -m pvdetect.lockbox`.
#
#   sh scripts/lockbox.sh seal    --dir PRIVATE --energy E --labels L
#   sh scripts/lockbox.sh develop --dir PRIVATE NAME [--unlabelled U] [-- TRAIN_FLAGS ...]
#   sh scripts/lockbox.sh compare --dir PRIVATE A B
#   sh scripts/lockbox.sh freeze  --dir PRIVATE NAME
#   sh scripts/lockbox.sh open    --dir PRIVATE
#   sh scripts/lockbox.sh final   --dir PRIVATE --energy E --labels L
#   sh scripts/lockbox.sh status  --dir PRIVATE
#
# Environment: PYTHON interpreter (default: .venv/bin/python if present, else python3).
# Run by the data holder only; PRIVATE must be outside this repository. Output is AIL-derived on real data.
set -eu

REPO=$(cd "$(dirname "$0")/.." && pwd -P)
PYTHON=${PYTHON:-}
if [ -z "$PYTHON" ]; then
  if [ -x "$REPO/.venv/bin/python" ]; then PYTHON=$REPO/.venv/bin/python; else PYTHON=python3; fi
fi
PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
PYTHONUNBUFFERED=1
export PYTHONPATH PYTHONUNBUFFERED
exec "$PYTHON" -m pvdetect.lockbox "$@"
