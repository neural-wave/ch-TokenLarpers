#!/bin/sh
# Export a clean submission tree for the NW_26 repository.
#
#   sh scripts/export_submission.sh COMMIT DEST [MODEL_DIR]
#
# 1. Extracts `git archive COMMIT` into DEST (created if needed): no history.
# 2. With MODEL_DIR: copies MODEL_DIR/model.json and exactly the files it
#    references (check_release.model_files, standard library only, so the default
#    python3 works; it mirrors pvdetect.model.referenced_files) into
#    DEST/artifacts/.  Without MODEL_DIR no model file is copied
#    (organizer-approved private channel instead).
# 3. Runs scripts/check_release.py on DEST with an allow list generated from
#    model.json; any failure aborts with a non-zero exit.
set -eu

usage() {
    echo "usage: sh scripts/export_submission.sh COMMIT DEST [MODEL_DIR]" >&2
    exit 2
}
[ $# -ge 2 ] && [ $# -le 3 ] || usage
COMMIT=$1
DEST=$2
MODEL_DIR=${3:-}
PY=${PYTHON:-python3}
case $PY in  # normalise a relative interpreter path so it survives directory changes
    */*) PY=$(cd "$(dirname "$PY")" && pwd)/$(basename "$PY") ;;
esac
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

abort() {
    echo "export: $1" >&2
    echo "export: aborted; do not commit or publish $DEST" >&2
    exit 1
}

REPO=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel) || abort "scripts/ is not inside a git checkout"
git -C "$REPO" rev-parse --quiet --verify "$COMMIT^{commit}" >/dev/null || abort "unknown commit: $COMMIT"
mkdir -p "$DEST"
DEST=$(cd "$DEST" && pwd)
TAR=$(mktemp "${TMPDIR:-/tmp}/export_submission.XXXXXX") || abort "could not create a temporary archive"
trap 'rm -f "$TAR"' EXIT
# A file, not a pipe: plain sh has no pipefail, so a failing git archive would go unnoticed.
git -C "$REPO" archive --format=tar -o "$TAR" "$COMMIT" || abort "git archive of $COMMIT failed"
tar -x -C "$DEST" -f "$TAR" || abort "extracting the archive of $COMMIT failed"

set --
if [ -n "$MODEL_DIR" ]; then
    [ -f "$MODEL_DIR/model.json" ] || abort "no model.json in $MODEL_DIR"
    MODEL_DIR=$(cd "$MODEL_DIR" && pwd)
    REFS=$("$PY" -c '
import json, sys
sys.path.insert(0, sys.argv[2])
from check_release import model_files
with open(sys.argv[1], encoding="utf-8") as handle:
    model = json.load(handle)
for name in model_files(model):
    print(name)
' "$MODEL_DIR/model.json" "$SCRIPT_DIR") || abort "could not list the files referenced by model.json"
    mkdir -p "$DEST/artifacts"
    cp "$MODEL_DIR/model.json" "$DEST/artifacts/model.json"
    set -- --allow artifacts/model.json
    while IFS= read -r ref; do
        [ -n "$ref" ] || continue
        [ -f "$MODEL_DIR/$ref" ] || abort "model.json references a missing file: $ref"
        mkdir -p "$(dirname "$DEST/artifacts/$ref")"
        cp "$MODEL_DIR/$ref" "$DEST/artifacts/$ref"
        set -- "$@" --allow "artifacts/$ref"
    done <<EOF
$REFS
EOF
fi

"$PY" "$SCRIPT_DIR/check_release.py" "$DEST" "$@" || abort "release check failed (see the paths above)"
echo "export: $DEST passed the release check; human review and organizer approval are still required"
