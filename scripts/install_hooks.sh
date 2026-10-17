#!/bin/sh
# Install the data-hygiene pre-commit hook into this clone (or worktree).
# The check is copied into the hook, not referenced, so it keeps working on
# branches that do not carry scripts/check_hygiene.py. Re-run after editing it.
set -eu
root=$(git rev-parse --show-toplevel)
cd "$root"
hook=$(git rev-parse --git-path hooks/pre-commit)
case "$hook" in
  /*) ;;
  *) hook="$root/$hook" ;;
esac
mkdir -p "$(dirname "$hook")"
cp scripts/check_hygiene.py "$hook"
chmod +x "$hook"
echo "installed $hook (copy of scripts/check_hygiene.py)"
