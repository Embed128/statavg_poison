"""Dataset loading, preprocessing, and non-IID federated partitioning.

Supports:
  1. Official TON_IoT Linux Memory CSV (`Train_test_linux_memory.csv` -> `dataset/dataset.csv`)
  2. Generic tabular IDS CSVs (e.g. TON_IoT Network, CICIoT2023)
  3. High-fidelity synthetic TON_IoT-like dataset for offline smoke testing
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


_UNIT_RE = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)\s*([kKmMgGtT]?)\s*$")
_UNIT_MULT = {"": 1.0, "k": 1e3, "m": 1e6, "g": 1e9, "t": 1e12}


def _parse_numeric_series(s: pd.Series) -> pd.Series:
    """Convert numeric or unit-suffixed strings (e.g. '12K', '3.4M') to float."""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(np.float64)

    def _conv(val):
        if pd.isna(val):
            return np.nan
        txt = str(val).strip()
        m = _UNIT_RE.match(txt)
        if m:
            return float(m.group(1)) * _UNIT_MULT[m.group(2).lower()]
        try:
            return float(txt)
        except ValueError:
            return np.nan

    parsed = s.map(_conv)
    # If almost all entries became NaN, treat as categorical via factorize
    if parsed.isna().mean() > 0.5:
        codes, _ = pd.factorize(s.astype(str))
        return pd.Series(codes.astype(np.float64), index=s.index)
    return parsed


@dataclass
class FederatedDataset:
    client_splits: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
    # Each entry: (x_train, y_train, x_val, y_val) in RAW (unnormalized) feature space
    x_test: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    class_names: List[str]
    num_features: int
    num_classes: int


def generate_synthetic_ton_iot(
    n_samples: int = 30000,
    seed: int = 42,
    save_path: str | None = None,
) -> pd.DataFrame:
    """Generate a synthetic TON_IoT Linux Memory-like CSV (8 features, 6 classes).

    Class imbalance mirrors TON_IoT Linux Memory (dominant normal + 5 attack types),
    with multi-scale features (e.g., ts, MINFLT, MAJFLT, VSTEXT, VSIZE, RSIZE, VGROW, RGROW)
    and class-conditional distributions.
    """
    rng = np.random.default_rng(seed)
    class_names = ["normal", "ddos", "dos", "injection", "mitm", "password"]
    class_probs = np.array([0.46, 0.14, 0.13, 0.11, 0.06, 0.10])
    counts = np.round(class_probs * n_samples).astype(int)
    counts[0] += n_samples - counts.sum()

    # Feature scales mimic real Linux memory telemetry (spanning 1e1 to 1e6)
    scales = np.array([1e5, 5e3, 2e2, 4e4, 8e5, 3e5, 1.5e4, 6e3], dtype=np.float64)
    base_centers = np.array([
        [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],   # normal
        [1.4,  2.1, -0.5,  1.2,  1.8,  1.5,  1.9,  1.2],   # ddos
        [-1.2, 1.7,  1.8, -0.8,  2.2, -0.6,  1.5, -1.1],   # dos
        [0.8, -1.4,  2.0,  1.9, -1.0,  1.6, -1.2,  1.8],   # injection
        [-1.8,-0.9, -1.5,  2.3, -1.3,  2.1,  0.7, -1.9],   # mitm
        [1.7, -0.8,  1.4, -1.6,  0.9, -1.7, -1.6,  1.6],   # password
    ], dtype=np.float64)

    rows = []
    labels = []
    for c_idx, (cname, cnt) in enumerate(zip(class_names, counts)):
        # Mixture of 2 sub-clusters per class for nonlinear MLP decision boundary
        half = cnt // 2
        rem = cnt - half
        z1 = rng.normal(loc=base_centers[c_idx], scale=1.15, size=(half, 8))
        z2 = rng.normal(loc=-0.55 * base_centers[c_idx] + 0.8 * np.roll(base_centers[c_idx], 1), scale=1.15, size=(rem, 8))
        z = np.vstack([z1, z2])
        # Add subtle cross-feature coupling
        z[:, 1] += 0.35 * np.sin(z[:, 0])
        z[:, 4] += 0.30 * z[:, 2] * z[:, 3] / 3.0
        x = z * scales[None, :] + np.array([1.55e9, 1e4, 5e2, 8e4, 2e6, 7e5, 2e4, 1e4])[None, :]
        rows.append(x)
        labels.extend([cname] * cnt)

    X = np.vstack(rows)
    y = np.array(labels)
    perm = rng.permutation(len(y))
    X, y = X[perm], y[perm]

    cols = ["ts", "MINFLT", "MAJFLT", "VSTEXT", "VSIZE", "RSIZE", "VGROW", "RGROW"]
    df = pd.DataFrame(X, columns=cols)
    df.insert(1, "PID", rng.integers(100, 32000, size=len(y)))
    df.insert(2, "CMD", "proc")
    df["label"] = (y != "normal").astype(int)
    df["type"] = y

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        df.to_csv(save_path, index=False)
    return df


def load_tabular_ids_csv(
    csv_path: str,
    max_rows: int | None = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Load TON_IoT Linux Memory CSV or generic tabular IDS CSV."""
    df = pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(axis=0, how="any")

    if max_rows is not None and len(df) > max_rows:
        target_col = "type" if "type" in df.columns else ("label" if "label" in df.columns else df.columns[-1])
        # Stratified subsample so minority classes remain represented
        vc = df[target_col].value_counts()
        valid_classes = vc[vc >= 20].index
        df = df[df[target_col].isin(valid_classes)]
        df, _ = train_test_split(
            df, train_size=max_rows, stratify=df[target_col], random_state=seed
        )

    # Identify target column (official StatAvg uses 'type')
    if "type" in df.columns:
        target_col = "type"
    elif "Attack_type" in df.columns:
        target_col = "Attack_type"
    elif "label" in df.columns:
        target_col = "label"
    else:
        target_col = df.columns[-1]

    # Filter out ultra-rare classes (< 20 samples) that cannot be split across clients
    vc = df[target_col].value_counts()
    keep_cls = vc[vc >= 20].index
    df = df[df[target_col].isin(keep_cls)].copy()

    y_raw = df[target_col].astype(str).to_numpy()
    enc = LabelEncoder()
    y = enc.fit_transform(y_raw).astype(np.int64)
    class_names = [str(c) for c in enc.classes_]

    # Drop official non-feature identifier columns if present
    drop_cols = {target_col, "PID", "CMD", "label", "Label", "ts_str", "src_ip", "dst_ip"}
    feat_cols = [c for c in df.columns if c not in drop_cols]

    X_cols = []
    used_names = []
    for c in feat_cols:
        parsed = _parse_numeric_series(df[c])
        if parsed.isna().any():
            parsed = parsed.fillna(parsed.median())
        arr = parsed.to_numpy(dtype=np.float64)
        # Drop constant zero-variance columns
        if np.std(arr) > 1e-12:
            X_cols.append(arr)
            used_names.append(c)

    X = np.column_stack(X_cols).astype(np.float64)
    return X, y, used_names, class_names


def partition_dirichlet(
    X: np.ndarray,
    y: np.ndarray,
    num_clients: int,
    alpha: float = 0.5,
    min_samples_per_client: int = 80,
    seed: int = 42,
) -> List[np.ndarray]:
    """Dirichlet(alpha) non-IID label partitioning across clients."""
    rng = np.random.default_rng(seed)
    num_classes = int(np.max(y)) + 1

    for _attempt in range(50):
        client_indices: List[List[int]] = [[] for _ in range(num_clients)]
        for c in range(num_classes):
            idx_c = np.where(y == c)[0]
            rng.shuffle(idx_c)
            proportions = rng.dirichlet(np.full(num_clients, alpha))
            # Split points
            cuts = (np.cumsum(proportions)[:-1] * len(idx_c)).astype(int)
            chunks = np.split(idx_c, cuts)
            for k in range(num_clients):
                client_indices[k].extend(chunks[k].tolist())
        if min(len(ci) for ci in client_indices) >= min_samples_per_client:
            return [np.array(ci, dtype=np.int64) for ci in client_indices]

    # Fallback: stratified equal split
    perm = rng.permutation(len(y))
    return [arr.astype(np.int64) for arr in np.array_split(perm, num_clients)]


def prepare_federated_dataset(
    csv_path: str,
    num_clients: int = 10,
    partition_mode: str = "dirichlet",
    dirichlet_alpha: float = 0.5,
    test_ratio: float = 0.2,
    val_ratio: float = 0.2,
    max_rows: int | None = None,
    seed: int = 42,
) -> FederatedDataset:
    """Load CSV, hold out global stratified test set, and partition clients."""
    X, y, feat_names, class_names = load_tabular_ids_csv(
        csv_path, max_rows=max_rows, seed=seed
    )

    X_train_all, X_test, y_train_all, y_test = train_test_split(
        X, y, test_size=test_ratio, stratify=y, random_state=seed
    )

    if partition_mode == "dirichlet":
        part_indices = partition_dirichlet(
            X_train_all, y_train_all, num_clients, alpha=dirichlet_alpha, seed=seed
        )
    elif partition_mode == "iid":
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(y_train_all))
        part_indices = [arr.astype(np.int64) for arr in np.array_split(perm, num_clients)]
    else:
        raise ValueError(f"Unknown partition_mode: {partition_mode}")

    client_splits = []
    for k, idx in enumerate(part_indices):
        xk, yk = X_train_all[idx], y_train_all[idx]
        # Safe local train/val split (stratify when every class has >=2 samples)
        _, counts = np.unique(yk, return_counts=True)
        strat = yk if (len(counts) > 1 and counts.min() >= 2) else None
        x_tr, x_va, y_tr, y_va = train_test_split(
            xk, yk, test_size=val_ratio, stratify=strat, random_state=seed + k
        )
        client_splits.append((x_tr, y_tr, x_va, y_va))

    return FederatedDataset(
        client_splits=client_splits,
        x_test=X_test,
        y_test=y_test,
        feature_names=feat_names,
        class_names=class_names,
        num_features=X.shape[1],
        num_classes=len(class_names),
    )
