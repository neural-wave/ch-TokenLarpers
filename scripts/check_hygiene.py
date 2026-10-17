#!/usr/bin/env python3
"""Refuse to commit organizer data or real customer identifiers.

Datasets and unapproved derived artifacts must not enter git history. The
user explicitly approved the exact checksum-pinned encoder and CNN files below.
The check inspects files staged for
commit (the default, used by the pre-commit hook), the working-tree copy of
every tracked file (``--all``, used by CI and one-off audits) or explicit
paths, and exits 1 when any of them

* is a data or model dump (``.csv``, ``.parquet``, ``.cbm``, ``.pt`` and
  similar) or lives under an artifacts/data directory, or
* contains a real customer identifier: the letters ``POD`` in any case,
  starting a word and immediately followed by digits, whatever comes after
  the digits.  Synthetic fixtures use a different scheme, for example
  ``PODSYN001``, which starts with ``POD`` but is not matched.

Only the offending paths are printed, never the matching content.
Standard library only, so the hook works in any interpreter.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import sys
from pathlib import PurePosixPath

ID_PATTERN = re.compile(rb"(?i)\bPOD\d+")
DATA_SUFFIXES = frozenset({
    ".csv", ".parquet", ".cbm", ".pt", ".pth", ".ckpt", ".npy", ".npz", ".pkl", ".pickle",
})
DATA_DIRS = frozenset({"artifacts", "artifacts_private", "data_private"})
DATA_DIR_PREFIXES = ("AIL_data_for_neural_wave",)

# Explicit publication approval: only these bytes at these exact paths.
PUBLISHED_ENCODERS = {
    "pretrained/small/encoder.pt": "fd006059193939a00d039f19168e2b80cb3d3b602df17268f23f5445d1952b3e",
    "pretrained/small/encoder.json": "5c5d159219b1055b8417e73db7129f7a22b739d33d0f60a72375b0db7ff35d95",
    "pretrained/medium/encoder.pt": "e8bbcc4273ad7f3af72ccdd93093a8560a23729a8e86af0a7d8745ae266b2857",
    "pretrained/medium/encoder.json": "5cd05732b466dae1f28683081612583ff71ee97f9c76ad574da4dbde2146df03",
    "pretrained/pv_presence/fraunhofer.pt": "55c68001718cf5a5932c9af5d99bd8e466229fc9ca1cc719dd3e44c26b34d08b",
    "pretrained/pv_presence/model.json": "c5d845e08f687a98dc8a652829e0cd773787dd920a7cbf446ef4b7a851566e73",
}


def is_published_encoder(path: str, content: bytes) -> bool:
    expected = PUBLISHED_ENCODERS.get(path)
    return expected is not None and hashlib.sha256(content).hexdigest() == expected


def _git(*args: str) -> bytes:
    return subprocess.run(["git", *args], check=True, capture_output=True).stdout


def staged_paths() -> list[str]:
    out = _git("diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z")
    return [p.decode("utf-8", "surrogateescape") for p in out.split(b"\0") if p]


def tracked_paths() -> list[str]:
    out = _git("ls-files", "-z")
    return [p.decode("utf-8", "surrogateescape") for p in out.split(b"\0") if p]


def staged_content(path: str) -> bytes:
    """Content of the path as it sits in the index (what the commit would contain)."""
    return _git("show", f":{path}")


def is_data_path(path: str) -> bool:
    parts = PurePosixPath(path).parts
    if PurePosixPath(path).suffix.lower() in DATA_SUFFIXES:
        return True
    if any(part in DATA_DIRS for part in parts[:-1]):
        return True
    return any(part.startswith(DATA_DIR_PREFIXES) for part in parts)


def has_customer_identifier(content: bytes) -> bool:
    return ID_PATTERN.search(content) is not None


def check(paths: list[str], reader) -> list[tuple[str, str]]:
    """Return (reason, path) for every offending path, in path order."""
    offending: list[tuple[str, str]] = []
    for path in paths:
        if path in PUBLISHED_ENCODERS:
            content = reader(path)
            if not is_published_encoder(path, content):
                offending.append(("published encoder hash mismatch", path))
                continue
        elif is_data_path(path):
            offending.append(("data file", path))
            continue
        else:
            content = reader(path)
        if has_customer_identifier(content):
            offending.append(("customer identifier", path))
    return offending


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*", help="explicit working-tree files to check instead of the index")
    parser.add_argument("--all", action="store_true", help="check every tracked file instead of the staged ones")
    args = parser.parse_args(argv)

    if args.paths:
        paths = list(args.paths)

        def reader(path: str) -> bytes:
            with open(path, "rb") as handle:
                return handle.read()
    elif args.all:
        paths = [p for p in tracked_paths() if os.path.isfile(p)]

        def reader(path: str) -> bytes:  # the working-tree copy of every tracked file
            with open(path, "rb") as handle:
                return handle.read()
    else:
        paths, reader = staged_paths(), staged_content

    offending = check(paths, reader)
    if not offending:
        return 0
    print(f"hygiene: refusing {len(offending)} file(s) that would publish organizer data:")
    for reason, path in offending:
        print(f"  {reason}: {path}")
    print("remove the data or identifiers (synthetic ids such as PODSYN001 are fine), then retry;")
    print("see scripts/check_hygiene.py --help for the rules")
    return 1


if __name__ == "__main__":
    sys.exit(main())
