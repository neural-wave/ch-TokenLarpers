from __future__ import annotations

import numpy as np
import pandas as pd

from pvdetect.daily import daily_signal
from pvdetect.io import LoadedTimeseries, day_matrix
from pvdetect.synthetic import CustomerSpec, generate_customer


def as_loaded(df: pd.DataFrame) -> LoadedTimeseries:
    """Convert generator output to the loader's frame without a CSV round trip."""
    frame = pd.DataFrame(
        {
            "customer_id": df["customer_id"].astype(str),
            "epoch_s": df["timestamp_epoch"].astype(np.int64),
            "energy_kwh": df["energy_consumption_kwh"].astype(float),
        }
    ).drop_duplicates(["customer_id", "epoch_s"]).sort_values(["customer_id", "epoch_s"]).reset_index(drop=True)
    ids = sorted(frame["customer_id"].unique().tolist())
    return LoadedTimeseries(frame=frame, customer_ids=ids, report={})


def customer_daily(spec: CustomerSpec, seed: int, variant: str = "A") -> pd.DataFrame:
    frame = as_loaded(generate_customer(spec, seed, variant)).frame
    return daily_signal(*day_matrix(frame))
