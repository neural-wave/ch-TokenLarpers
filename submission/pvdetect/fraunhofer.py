"""PyTorch port of Fraunhofer Neubert et al. (Energies 15(13):4922, 2022).

Combines a 1D CNN over 168-hour weekly load profiles with an MLP over
a 100-bin Gaussian Kernel Density Estimation (KDE) feature distribution.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

KDE_GRID = np.linspace(-5.0, 15.0, 100, dtype=np.float32)
KDE_H = 0.25
PERSISTENCE_WEEKS = 4


def fast_kde_batch(X: np.ndarray, grid: np.ndarray = KDE_GRID, h: float = KDE_H) -> np.ndarray:
    """Vectorized Gaussian KDE of 1D series on fixed grid.
    
    X: shape (N, L), grid: shape (G,) -> returns (N, G).
    """
    diff = (X[:, :, None] - grid[None, None, :]) / h
    weights = np.exp(-0.5 * (diff ** 2)) / (math.sqrt(2 * math.pi) * h)
    return weights.mean(axis=1).astype(np.float32)


@dataclass
class WindowDataset:
    customer_ids: np.ndarray  # (N,) string
    week_start_dates: np.ndarray  # (N,) datetime64[D]
    series: np.ndarray  # (N, 1, 168) float32
    kdes: np.ndarray  # (N, 100) float32
    labels: np.ndarray  # (N,) float32 (0.0 or 1.0)


def extract_customer_windows(
    customer_id: str,
    dates: np.ndarray,
    M: np.ndarray,
    pv_installed: bool,
    install_date: pd.Timestamp | None = None,
    min_coverage: float = 0.80,
) -> list[dict]:
    """Extract 168-hour weekly windows from day matrix M (n_days, 96)."""
    n_days = len(dates)
    n_weeks = n_days // 7
    windows = []
    
    # Normalize each customer's hourly profile once; both branches consume
    # this same scale-free signal.  A fixed additive offset would make the
    # representation depend on the customer's absolute consumption.
    quarters = M.reshape(n_days, 24, 4)
    daily_hourly = np.sum(quarters, axis=-1)
    valid_hourly = daily_hourly[np.isfinite(daily_hourly)]
    cust_median = float(np.median(valid_hourly)) if len(valid_hourly) > 0 else 0.0
    scale = abs(cust_median)
    if scale < 1e-6:
        scale = 1.0

    # An undated positive has unknown window labels, not negative windows.
    if pv_installed and pd.isna(install_date):
        return windows
    for w in range(n_weeks):
        w_dates = dates[w * 7 : (w + 1) * 7]
        w_M = M[w * 7 : (w + 1) * 7]  # (7, 96)
        
        # Reshape to hourly sums: (7, 24, 4) -> sum over 15-min intervals
        hourly = np.nansum(w_M.reshape(7, 24, 4), axis=-1).flatten()  # (168,)
        valid_mask = np.isfinite(w_M.reshape(7, 24, 4)).all(axis=-1).flatten()
        coverage = valid_mask.mean()
        
        if coverage < min_coverage:
            continue
            
        # Impute missing hours with median of valid hours in this week
        if not valid_mask.all():
            fill_val = np.median(hourly[valid_mask]) if valid_mask.any() else cust_median
            hourly[~valid_mask] = fill_val

        start_date = pd.Timestamp(w_dates[0])
        end_date = pd.Timestamp(w_dates[-1])

        # Window labelling
        if not pv_installed or pd.isna(install_date):
            label = 0.0
        else:
            t_inst = pd.Timestamp(install_date)
            if end_date < t_inst:
                label = 0.0
            elif start_date >= t_inst:
                label = 1.0
            else:
                # Straddling window -> drop
                continue

        # Normalized series and KDE
        norm_hourly = (hourly / scale).astype(np.float32)
        windows.append({
            "customer_id": str(customer_id),
            "week_start": start_date,
            "norm_hourly": norm_hourly,
            "label": label,
        })
    return windows


def build_window_dataset(
    customers_data: dict,
    labels_df: pd.DataFrame | None = None,
    ids: Sequence[str] | None = None,
) -> WindowDataset:
    """Build WindowDataset for the given customer IDs (labels_df is optional for inference)."""
    target_ids = list(ids) if ids is not None else list(customers_data.keys())
    lab = labels_df.set_index("customer_id") if labels_df is not None else None
    all_windows = []

    for cid in target_ids:
        if cid not in customers_data:
            continue
        cdata = customers_data[cid]
        if cdata is None or len(cdata.dates) == 0:
            continue
        if lab is not None:
            if cid not in lab.index:
                raise ValueError("CNN training customer is missing its label")
            row = lab.loc[cid]
            pv_inst = bool(row["pv_installed"])
            t_inst = row.get("pv_installation_date", None)
        else:
            pv_inst, t_inst = False, None
        wins = extract_customer_windows(cid, cdata.dates, cdata.M, pv_inst, t_inst)
        all_windows.extend(wins)
    if not all_windows:
        return WindowDataset(
            customer_ids=np.array([], dtype=str),
            week_start_dates=np.array([], dtype="datetime64[D]"),
            series=np.zeros((0, 1, 168), dtype=np.float32),
            kdes=np.zeros((0, 100), dtype=np.float32),
            labels=np.zeros((0,), dtype=np.float32),
        )

    norm_series = np.stack([w["norm_hourly"] for w in all_windows])  # (N, 168)
    # KDE and CNN must see identical per-customer-normalized hourly inputs.
    kdes = fast_kde_batch(norm_series, KDE_GRID, KDE_H)  # (N, 100)

    return WindowDataset(
        customer_ids=np.array([w["customer_id"] for w in all_windows]),
        week_start_dates=np.array([w["week_start"] for w in all_windows]),
        series=norm_series[:, None, :],  # (N, 1, 168)
        kdes=kdes,
        labels=np.array([w["label"] for w in all_windows], dtype=np.float32),
    )


class MLPBranch(nn.Module):
    """4-layer LeakyReLU MLP on 100-bin KDE input."""
    def __init__(self, in_features: int = 100, hidden: int = 400):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.LeakyReLU(0.05),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.05),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.05),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.05),
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.net(x)
        logits = self.head(feat).squeeze(-1)
        return logits, feat


class CNNBranch(nn.Module):
    """4-layer 1D CNN over 168-hour weekly profiles."""
    def __init__(self, in_channels: int = 1, hidden: int = 400):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 10, hidden),
            nn.LeakyReLU(0.05),
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.fc(self.conv(x))
        logits = self.head(feat).squeeze(-1)
        return logits, feat


class CombinedClassifier(nn.Module):
    """Combined head on frozen MLP and CNN features."""
    def __init__(self, mlp: MLPBranch, cnn: CNNBranch, hidden: int = 200):
        super().__init__()
        self.mlp = mlp
        self.cnn = cnn
        self.head = nn.Sequential(
            nn.Linear(800, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, kde: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            _, f_mlp = self.mlp(kde)
            _, f_cnn = self.cnn(seq)
        f_comb = torch.cat([f_mlp, f_cnn], dim=-1)
        return self.head(f_comb).squeeze(-1)


def _training_weights(customer_ids: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, float]:
    """Equal customer mass, with class balance measured under that same mass."""
    _, owner, counts = np.unique(customer_ids, return_inverse=True, return_counts=True)
    weights = len(labels) / (len(counts) * counts[owner])
    n_pos = float(np.dot(weights, labels))
    n_neg = float(np.dot(weights, 1.0 - labels))
    pos_weight = max(0.1, n_neg / n_pos) if n_pos > 0 else 1.0
    return weights.astype(np.float32), pos_weight


def train_fraunhofer(
    train_ds: WindowDataset,
    epochs: tuple[int, int, int] = (40, 30, 20),
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str | None = None,
    seed: int = 0,
) -> CombinedClassifier:
    """Train the Neubert model in 3 stages: MLP, CNN, then Combined Head."""
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(seed)
    np.random.seed(seed)

    n = len(train_ds.labels)
    if n == 0:
        raise ValueError("CNN training requires at least one labelled, sufficiently covered week")
    weights, positive_weight = _training_weights(train_ds.customer_ids, train_ds.labels)
    pos_weight = torch.tensor([positive_weight], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    x_kde = torch.from_numpy(train_ds.kdes).to(device)
    x_seq = torch.from_numpy(train_ds.series).to(device)
    y = torch.from_numpy(train_ds.labels).to(device)

    sample_weights = torch.from_numpy(weights).to(device)
    dataset = TensorDataset(x_kde, x_seq, y, sample_weights)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    mlp = MLPBranch().to(device)
    cnn = CNNBranch().to(device)

    # Stage 1: Train MLP branch
    opt_mlp = torch.optim.Adam(mlp.parameters(), lr=lr)
    mlp.train()
    for _ in range(epochs[0]):
        for b_kde, _, b_y, b_weight in loader:
            opt_mlp.zero_grad()
            logits, _ = mlp(b_kde)
            loss = (loss_fn(logits, b_y) * b_weight).mean()
            loss.backward()
            opt_mlp.step()

    # Stage 2: Train CNN branch
    opt_cnn = torch.optim.Adam(cnn.parameters(), lr=lr)
    cnn.train()
    for _ in range(epochs[1]):
        for _, b_seq, b_y, b_weight in loader:
            opt_cnn.zero_grad()
            logits, _ = cnn(b_seq)
            loss = (loss_fn(logits, b_y) * b_weight).mean()
            loss.backward()
            opt_cnn.step()

    # Freeze branches
    mlp.eval()
    cnn.eval()
    for p in mlp.parameters():
        p.requires_grad = False
    for p in cnn.parameters():
        p.requires_grad = False

    # Stage 3: Train Combined Head
    model = CombinedClassifier(mlp, cnn).to(device)
    opt_head = torch.optim.Adam(model.head.parameters(), lr=lr)
    model.head.train()
    for _ in range(epochs[2]):
        for b_kde, b_seq, b_y, b_weight in loader:
            opt_head.zero_grad()
            logits = model(b_kde, b_seq)
            loss = (loss_fn(logits, b_y) * b_weight).mean()
            loss.backward()
            opt_head.step()

    model.eval()
    return model


def predict_fraunhofer(
    model: CombinedClassifier,
    test_ds: WindowDataset,
    batch_size: int = 128,
    device: str | torch.device | None = None,
) -> np.ndarray:
    """Predict window probabilities."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    if len(test_ds.labels) == 0:
        return np.array([], dtype=np.float32)

    x_kde = torch.from_numpy(test_ds.kdes).to(device)
    x_seq = torch.from_numpy(test_ds.series).to(device)
    dataset = TensorDataset(x_kde, x_seq)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    probs = []
    with torch.no_grad():
        for b_kde, b_seq in loader:
            logits = model(b_kde, b_seq)
            p = torch.sigmoid(logits).cpu().numpy()
            probs.append(p)
    return np.concatenate(probs).astype(np.float32)


def _persistent_week_score(probabilities: np.ndarray, dates: np.ndarray) -> float:
    """Strongest four-calendar-week run, limited by its weakest weekly score.

    Missing/rejected weeks break a run. Histories with fewer than four
    consecutive usable weeks have no supported persistent evidence.
    """
    if len(probabilities) < PERSISTENCE_WEEKS:
        return 0.0
    order = np.argsort(dates)
    dates = np.asarray(dates)[order].astype("datetime64[D]")
    probabilities = np.asarray(probabilities)[order]
    best = 0.0
    for start in range(len(dates) - PERSISTENCE_WEEKS + 1):
        stop = start + PERSISTENCE_WEEKS
        if np.all(np.diff(dates[start:stop]) == np.timedelta64(7, "D")):
            best = max(best, float(probabilities[start:stop].min()))
    return best


def fraunhofer_customer_scores(
    model: CombinedClassifier,
    table,
    ids: Sequence[str],
    device: str | torch.device | None = None,
) -> np.ndarray:
    """Score persistent PV evidence without dilution by earlier non-PV history."""
    if device is None:
        device = next(model.parameters()).device
    ds = build_window_dataset(table.customers, labels_df=None, ids=ids)
    if len(ds.labels) == 0:
        return np.zeros(len(ids), dtype=np.float32)
    probs = predict_fraunhofer(model, ds, device=device)
    scores = []
    for cid in ids:
        mask = ds.customer_ids == cid
        if mask.any():
            scores.append(_persistent_week_score(probs[mask], ds.week_start_dates[mask]))
        else:
            scores.append(0.0)
    return np.array(scores, dtype=np.float32)


def inner_customer_scores(table, labels: pd.DataFrame, ids: Sequence[str], splits) -> np.ndarray:
    """Customer-level OOF scores on the procedure's common inner splits."""
    scores = pd.Series(index=list(ids), dtype=float)
    seen = set()
    for train_ids, valid_ids in splits:
        if set(train_ids) & set(valid_ids):
            raise ValueError("CNN inner split separates windows of the same customer")
        if not set(train_ids) <= set(ids) or not set(valid_ids) <= set(ids) or seen & set(valid_ids):
            raise ValueError("CNN inner splits must partition the fit's customers")
        net = train_fraunhofer(build_window_dataset(table.customers, labels, train_ids))
        scores.loc[valid_ids] = fraunhofer_customer_scores(net, table, valid_ids)
        seen.update(valid_ids)
    if seen != set(ids):
        raise ValueError("CNN inner splits must validate every training customer exactly once")
    return scores.to_numpy(dtype=float)
