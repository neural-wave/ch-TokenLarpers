#!/usr/bin/env python3
"""Check the repository-contained encoder pairs with fabricated CPU inputs only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=("small", "medium"), default=["small", "medium"])
    args = parser.parse_args(argv)
    import numpy as np
    import pandas as pd
    import torch
    from neuralwave_experiment.checkpoint import load_encoder_artifact
    from neuralwave_experiment.windows import WindowConfig, build_windows

    torch.set_num_threads(1)
    catalog = json.loads((REPO / "configs/pretrained_encoders.json").read_text())
    if catalog.get("schema_version") != 1:
        raise ValueError("unsupported pretrained catalog schema")
    for size in dict.fromkeys(args.models):
        entry = catalog["models"][size]
        if entry["directory"] != f"pretrained/{size}":
            raise ValueError("unexpected pretrained model directory")
        directory = REPO / entry["directory"]
        for filename, key in (("encoder.pt", "weights_sha256"), ("encoder.json", "metadata_sha256")):
            path = directory / filename
            if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != entry[key]:
                raise ValueError(f"{size}: {filename} integrity mismatch")
        model, metadata = load_encoder_artifact(directory, device="cpu")
        config = metadata["encoder_config"]
        length = int(config["context_length"])
        epochs = 1672531200 + np.arange(length, dtype=np.int64) * 900
        frame = pd.DataFrame({
            "customer_id": "PODSYNMODEL",
            "timestamp_epoch": epochs,
            "energy_kwh": 1.0 + 0.3 * np.sin(np.arange(length) * 2 * np.pi / 96),
            "value_status": "observed",
        })
        batch = build_windows(frame, WindowConfig(context_length=length, stride=96))
        values = torch.as_tensor(batch.values)
        observed = torch.as_tensor(batch.observed)
        calendar = torch.as_tensor(batch.calendar)
        if model.training or any(p.requires_grad for p in model.parameters()):
            raise ValueError(f"{size}: encoder must load frozen in evaluation mode")
        with torch.no_grad():
            first = model(values, observed, calendar)
            second = model(values, observed, calendar)
        if first.shape != (1, int(config["d_model"])) or not torch.isfinite(first).all():
            raise ValueError(f"{size}: invalid synthetic forward output")
        if not torch.equal(first, second):
            raise ValueError(f"{size}: frozen inference is not deterministic")
        print(f"{size}: checksums, frozen CPU load and deterministic synthetic forward passed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
