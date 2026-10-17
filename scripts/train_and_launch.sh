#!/bin/sh
# Train the final model on the labelled data, then open the local GUI with it.
#
#   sh scripts/train_and_launch.sh DEV_ENERGY_CSV DEV_LABELS_CSV [UNLABELLED_CSV]
#
# Environment:
#   PRIVATE_DIR  where the model and the CV log go (default: ~/pvdetect_private/final); must be outside this repo
#   PYTHON       interpreter (default: .venv/bin/python if present, else python3)
#   CV=0         skip nested CV (faster; only the final model is fitted)
#   NO_GUI=1     train only, do not start the GUI
#   PORT=8766    GUI port (default 8765)
#   TRAIN_FLAGS  extra train.py flags, word-split (e.g. TRAIN_FLAGS=--no-e1e2)
#
# Run by the data holder only. The CV lines are AIL-derived: keep them with the team, never in an agent session.
set -eu

if [ $# -lt 2 ] || [ $# -gt 3 ]; then
  echo "usage: sh scripts/train_and_launch.sh DEV_ENERGY_CSV DEV_LABELS_CSV [UNLABELLED_CSV]" >&2
  exit 2
fi

REPO=$(cd "$(dirname "$0")/.." && pwd -P)
cd "$REPO"
PYTHON=${PYTHON:-}
if [ -z "$PYTHON" ]; then
  if [ -x .venv/bin/python ]; then PYTHON=.venv/bin/python; else PYTHON=python3; fi
fi
PRIVATE_DIR=${PRIVATE_DIR:-$HOME/pvdetect_private/final}
[ -d "$PRIVATE_DIR" ] && CREATED=0 || CREATED=1
mkdir -p "$PRIVATE_DIR"
PRIVATE_DIR=$(cd "$PRIVATE_DIR" && pwd -P)

case "$PRIVATE_DIR/" in
  "$REPO"/*)
    [ "$CREATED" = "1" ] && rmdir "$PRIVATE_DIR"
    echo "refusing: PRIVATE_DIR is inside the repository; choose a folder outside it" >&2
    exit 1
    ;;
esac
chmod 700 "$PRIVATE_DIR"

for f in "$1" "$2" ${3:+"$3"}; do
  [ -f "$f" ] || { echo "input file not found (argument $f)" >&2; exit 1; }
done

LABELS=$2
set -- --timeseries "$1" --labels "$2" ${3:+--unlabelled "$3"}
[ "${CV:-1}" = "0" ] || set -- "$@" --cv --oof-out "$PRIVATE_DIR/oof.csv"

echo "training; model and log go to the private folder"
rm -f "$PRIVATE_DIR/model.json" "$PRIVATE_DIR/oof.csv"
# shellcheck disable=SC2086  # TRAIN_FLAGS is a word-split flag list
PYTHONUNBUFFERED=1 "$PYTHON" train.py "$@" ${TRAIN_FLAGS:-} --out "$PRIVATE_DIR/model.json" | tee "$PRIVATE_DIR/train_log.txt"
[ -f "$PRIVATE_DIR/model.json" ] || { echo "training failed; see the messages above" >&2; exit 1; }

rm -f "$PRIVATE_DIR/diagnosis.json"
if [ -f "$PRIVATE_DIR/oof.csv" ]; then
  echo
  echo "error analysis on out-of-fold predictions (each customer predicted by a model that never saw it):"
  "$PYTHON" -m pvdetect.diagnose "$PRIVATE_DIR/oof.csv" "$LABELS" --out-dir "$PRIVATE_DIR" | tee "$PRIVATE_DIR/diagnosis.txt"
fi

[ "${NO_GUI:-0}" = "1" ] && exit 0
exec "$PYTHON" gui.py --model "$PRIVATE_DIR/model.json" --labels "$LABELS" ${PORT:+--port "$PORT"}
