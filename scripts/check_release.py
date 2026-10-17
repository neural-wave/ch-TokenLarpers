#!/usr/bin/env python3
"""Audit an exported submission tree before it is published.

The NW_26 submission repository has no pre-commit hook, so this check walks
every file under DEST on the filesystem (not via git) and exits 1 when

* a file that is not on the ``--allow`` list is a data or model dump, or lives
  under an artifacts/data directory (``check_hygiene.is_data_path``);
* any file, allowed ones included, contains a real customer identifier
  (``check_hygiene.ID_PATTERN``: the letters ``POD`` in any case, starting a
  word and immediately followed by digits);
* an ``--allow`` path does not exist;
* an entry is a symlink (it would escape the audit: links are not followed,
  so the check fails closed instead);
* a path itself contains a real customer identifier (review plots, for
  example, are named after customers); such paths are printed redacted.

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
from check_hygiene import PUBLISHED_ENCODERS, has_customer_identifier, is_data_path, is_published_encoder  # noqa: E402


REDACTED = "<path containing a customer identifier>"


def model_files(model: dict) -> list[str]:
    """Sibling files a model.json references, as safe relative POSIX paths.

    Standard library only, so the export runs with any interpreter.  Must mirror
    ``pvdetect.model.referenced_files``, including the optional encoder pair.
    Raises ``ValueError`` for absolute, parent-relative or otherwise unsafe names.
    """
    names = []
    for section in ("classifier", "capacity"):
        name = (model.get(section) or {}).get("catboost_file")
        if not name:
            continue
        names.append(name)
    embedding = model.get("embedding")
    if embedding is not None:
        refs = embedding.get("encoder_files")
        if not isinstance(refs, dict) or set(refs) != {"weights", "metadata"}:
            raise ValueError("embedding encoder_files must name weights and metadata")
        names.extend(refs[key] for key in ("weights", "metadata"))
    for name in names:
        path = PurePosixPath(name)
        if (path.is_absolute() or ".." in path.parts or "\\" in name
                or any(c in name for c in "\r\n\x00") or not name
                or path.as_posix() != name):
            raise ValueError("unsafe referenced file name in model.json")
    if len(names) != len(set(names)):
        raise ValueError("duplicate model artifact reference")
    return names


def _id_in_path(path: str) -> bool:
    return has_customer_identifier(path.encode("utf-8", "surrogateescape"))


def tree_entries(dest: Path) -> tuple[list[str], list[str]]:
    """(files, symlinks) under dest as POSIX paths relative to it, skipping .git directories.

    Symlinks (to files or directories, dangling or not) are reported, never followed or opened.
    """
    files, links = [], []
    for root, dirs, names in os.walk(dest):
        rel_root = Path(root).relative_to(dest)
        keep = []
        for d in sorted(dirs):
            if d == ".git":
                continue
            if (Path(root) / d).is_symlink():
                links.append((rel_root / d).as_posix())
            else:
                keep.append(d)
        dirs[:] = keep
        for name in sorted(names):
            rel = (rel_root / name).as_posix()
            (links if (Path(root) / name).is_symlink() else files).append(rel)
    return files, links


def check(dest: Path, allowed: set[str]) -> list[tuple[str, str]]:
    """Return (reason, path) for every problem, missing allowed files first; identifier paths redacted."""
    files, links = tree_entries(dest)
    offending = [("missing allowed file", p) for p in sorted(allowed) if not (dest / p).is_file()]
    offending += [("symlink", p) for p in links]
    for path in files:
        if _id_in_path(path):
            offending.append(("customer identifier in path", path))
        with open(dest / path, "rb") as handle:
            content = handle.read()
        if path in PUBLISHED_ENCODERS:
            if not is_published_encoder(path, content):
                offending.append(("published encoder hash mismatch", path))
        elif path not in allowed and is_data_path(path):
            offending.append(("data file", path))
        if has_customer_identifier(content):
            offending.append(("customer identifier", path))
    return [(reason, REDACTED if _id_in_path(path) else path) for reason, path in offending]


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
