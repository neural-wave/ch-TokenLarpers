#!/usr/bin/env python3
"""Audit an exported submission tree before it is published.

The NW_26 submission repository has no pre-commit hook, so this check walks
every file under DEST on the filesystem (not via git) and exits 1 when

* a file that is not on the ``--allow`` list is a data or model dump, or lives
  under an artifacts/data directory (``check_hygiene.is_data_path``);
* any file, allowed ones included, contains a real customer identifier
  (``check_hygiene.ID_PATTERN``: the letters ``POD`` in any case, starting a
  word and immediately followed by digits);
* an ``--allow`` path does not exist.

``--allow`` paths are exact files relative to DEST in POSIX form (no
wildcards, no directories); the export script generates them from
``model.json``.  ``.git`` directories are skipped: they hold a checkout's
metadata, not the published tree.  Only paths and reasons are printed, never
file content.  Passing is necessary, not sufficient: the export still needs
human review and organizer approval.  Standard library only.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_hygiene import has_customer_identifier, is_data_path  # noqa: E402


def tree_files(dest: Path) -> list[str]:
    """Every file under dest as a POSIX path relative to it, skipping .git directories."""
    files = []
    for root, dirs, names in os.walk(dest):
        dirs[:] = sorted(d for d in dirs if d != ".git")
        rel_root = Path(root).relative_to(dest)
        for name in sorted(names):
            files.append((rel_root / name).as_posix())
    return files


def check(dest: Path, allowed: set[str]) -> list[tuple[str, str]]:
    """Return (reason, path) for every problem, missing allowed files first."""
    offending = [("missing allowed file", p) for p in sorted(allowed) if not (dest / p).is_file()]
    for path in tree_files(dest):
        if path not in allowed and is_data_path(path):
            offending.append(("data file", path))
        with open(dest / path, "rb") as handle:
            if has_customer_identifier(handle.read()):
                offending.append(("customer identifier", path))
    return offending


def _allow_path(value: str) -> str:
    p = PurePosixPath(value)
    if p.is_absolute() or ".." in p.parts or "\\" in value or any(c in value for c in "*?["):
        raise argparse.ArgumentTypeError(f"--allow takes an exact relative POSIX file path: {value}")
    return p.as_posix()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dest", type=Path, help="exported tree to audit")
    parser.add_argument("--allow", action="append", default=[], type=_allow_path, metavar="PATH",
                        help="exact file (relative to DEST) that may be a data path; repeatable")
    args = parser.parse_args(argv)
    if not args.dest.is_dir():
        print(f"release: not a directory: {args.dest}")
        return 1

    offending = check(args.dest, set(args.allow))
    if not offending:
        return 0
    print(f"release: {len(offending)} problem(s) in {args.dest}:")
    for reason, path in offending:
        print(f"  {reason}: {path}")
    print("do not publish this tree; see scripts/check_release.py --help for the rules")
    return 1


if __name__ == "__main__":
    sys.exit(main())
