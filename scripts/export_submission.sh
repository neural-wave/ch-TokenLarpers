#!/bin/sh
# Export a clean submission tree for the NW_26 repository.
#
#   sh scripts/export_submission.sh COMMIT DEST [MODEL_DIR]
#
# 1. Extracts `git archive COMMIT` into DEST (created if needed): no history.
# 2. With MODEL_DIR: copies MODEL_DIR/model.json and exactly the files it
#    references (pvdetect.model.referenced_files, evaluated with ${PYTHON:-python3}
#    against the exported tree) into DEST/artifacts/.  Without MODEL_DIR no model
#    file is copied (organizer-approved private channel instead).
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
case $PY in  # the referenced-file listing runs inside DEST, so a relative interpreter path must be absolute
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
    REFS=$(cd "$DEST" && "$PY" -c '
import json, sys
from pathlib import PurePosixPath
from pvdetect.model import referenced_files
with open(sys.argv[1], encoding="utf-8") as handle:
    model = json.load(handle)
for name in referenced_files(model):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\n" in name or "\\" in name:
        sys.exit("unsafe referenced file name in model.json")
    print(path.as_posix())
' "$MODEL_DIR/model.json") || abort "could not list the files referenced by model.json"
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
