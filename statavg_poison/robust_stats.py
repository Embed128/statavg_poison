"""Robust statistics aggregation defenses for StatAvg.

Implements:
  - vanilla: original StatAvg weighted mean + pooled variance (IEEE TNSM 2025)
  - n_clip: clip reported sample counts at n_clip_mult * median(n), then vanilla
  - median: coordinate-wise median for mean and variance
  - trimmed: coordinate-wise trimmed mean (removes top/bottom trim_ratio)
  - mad_filter: MAD-based outlier client rejection on (mean, log-var) + n_clip + exact pooled aggregation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np


VALID_DEFENSES = (
    "vanilla",
    "n_clip",
    "median",
    "trimmed",
    "mad_filter",
)


@dataclass
class DefenseConfig:
    defense_type: str = "vanilla"
    trim_ratio: float = 0.2          # fraction trimmed from each tail for trimmed mean
    mad_threshold: float = 3.5       # modified Z-score threshold for MAD outlier rejection
    n_clip_mult: float = 2.0         # cap n_k at n_clip_mult * median(n)
    min_keep_frac: float = 0.5       # safeguard: always keep at least 50% of clients
    mad_feature_quantile: float = 0.85  # quantile across feature Z-scores (robust to targeted attacks)


@dataclass
class AggregationResult:
    mean_global: np.ndarray
    var_global: np.ndarray
    flagged_clients: List[int] = field(default_factory=list)
    client_scores: List[float] = field(default_factory=list)
    effective_weights: List[float] = field(default_factory=list)


def _pooled_stats(
    means: np.ndarray, vars_: np.ndarray, weights: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Exact StatAvg weighted mean and parallel pooled variance."""
    w = np.asarray(weights, dtype=np.float64)
    w_sum = np.sum(w)
    if w_sum <= 0:
        w = np.ones_like(w)
        w_sum = np.sum(w)
    w_norm = w[:, None] / w_sum
    mean_g = np.sum(w_norm * means, axis=0)
    var_g = np.sum(w_norm * (vars_ + (means - mean_g[None, :]) ** 2), axis=0)
    var_g = np.maximum(var_g, 1e-8)
    return mean_g, var_g


def _clip_counts(counts: np.ndarray, mult: float) -> np.ndarray:
    med = float(np.median(counts))
    cap = max(1.0, mult * med)
    return np.minimum(counts.astype(np.float64), cap)


def _mad_scores(matrix: np.ndarray) -> np.ndarray:
    """Compute modified Z-scores per (client, feature) using median & MAD.

    matrix: shape (K, D)
    returns: shape (K, D) non-negative outlier scores
    """
    med = np.median(matrix, axis=0, keepdims=True)
    abs_dev = np.abs(matrix - med)
    mad = np.median(abs_dev, axis=0, keepdims=True)
    # Consistency constant for normal distribution: 1.4826 * MAD
    # Fallback when MAD == 0 (e.g. >50% identical values): use IQR/1.349 or tiny floor
    iqr = np.percentile(matrix, 75, axis=0, keepdims=True) - np.percentile(
        matrix, 25, axis=0, keepdims=True
    )
    scale = np.where(mad > 1e-9, 1.4826 * mad, np.where(iqr > 1e-9, iqr / 1.349, 1e-6))
    return abs_dev / scale


def aggregate_statistics(
    means: Sequence[np.ndarray],
    vars_: Sequence[np.ndarray],
    num_examples: Sequence[int],
    cfg: DefenseConfig,
    client_ids: Sequence[int] | None = None,
) -> AggregationResult:
    """Aggregate client statistics under the chosen defense strategy."""
    if cfg.defense_type not in VALID_DEFENSES:
        raise ValueError(f"Unknown defense_type: {cfg.defense_type}")

    M = np.asarray(means, dtype=np.float64)        # (K, D)
    V = np.maximum(np.asarray(vars_, dtype=np.float64), 1e-10)  # (K, D)
    N = np.asarray(num_examples, dtype=np.float64)  # (K,)
    K, D = M.shape
    cids = list(range(K)) if client_ids is None else list(client_ids)

    if cfg.defense_type == "vanilla":
        mg, vg = _pooled_stats(M, V, N)
        return AggregationResult(
            mean_global=mg,
            var_global=vg,
            flagged_clients=[],
            client_scores=[0.0] * K,
            effective_weights=(N / np.sum(N)).tolist(),
        )

    if cfg.defense_type == "n_clip":
        w = _clip_counts(N, cfg.n_clip_mult)
        mg, vg = _pooled_stats(M, V, w)
        return AggregationResult(
            mean_global=mg,
            var_global=vg,
            flagged_clients=[],
            client_scores=[0.0] * K,
            effective_weights=(w / np.sum(w)).tolist(),
        )

    if cfg.defense_type == "median":
        mg = np.median(M, axis=0)
        # Robust pooled variance estimate: median within-client var + median between-client sq dev
        vg = np.median(V, axis=0) + np.median((M - mg[None, :]) ** 2, axis=0)
        vg = np.maximum(vg, 1e-8)
        return AggregationResult(
            mean_global=mg,
            var_global=vg,
            flagged_clients=[],
            client_scores=[0.0] * K,
            effective_weights=[1.0 / K] * K,
        )

    if cfg.defense_type == "trimmed":
        k_trim = int(np.floor(K * cfg.trim_ratio))
        if 2 * k_trim >= K:
            k_trim = max(0, (K - 1) // 2)
        mg = np.zeros(D, dtype=np.float64)
        vg = np.zeros(D, dtype=np.float64)
        w_clipped = _clip_counts(N, cfg.n_clip_mult)
        for d in range(D):
            order_m = np.argsort(M[:, d])
            keep_m = order_m[k_trim : K - k_trim] if k_trim > 0 else order_m
            mg[d] = np.average(M[keep_m, d], weights=w_clipped[keep_m])

            # Trim total second-moment contribution v_k + (m_k - mg)^2
            comp_v = V[:, d] + (M[:, d] - mg[d]) ** 2
            order_v = np.argsort(comp_v)
            keep_v = order_v[k_trim : K - k_trim] if k_trim > 0 else order_v
            vg[d] = np.average(comp_v[keep_v], weights=w_clipped[keep_v])
        vg = np.maximum(vg, 1e-8)
        return AggregationResult(
            mean_global=mg,
            var_global=vg,
            flagged_clients=[],
            client_scores=[0.0] * K,
            effective_weights=(w_clipped / np.sum(w_clipped)).tolist(),
        )

    # mad_filter (recommended defense):
    # Evaluate client-level anomaly score on both means and log-variances,
    # plus detect sample-count anomalies, then apply exact pooled aggregation on clean clients.
    z_mean = _mad_scores(M)                        # (K, D)
    z_logv = _mad_scores(np.log(V + 1e-10))        # (K, D)
    z_combined = np.maximum(z_mean, z_logv)        # (K, D)

    # Use high-quantile feature score so both global and targeted-feature attacks get caught
    client_scores = np.quantile(z_combined, cfg.mad_feature_quantile, axis=1)

    # Flag clients exceeding threshold while enforcing min_keep_frac
    min_keep = max(1, int(np.ceil(K * cfg.min_keep_frac)))
    order = np.argsort(client_scores)  # ascending (most normal first)
    keep_mask = client_scores <= cfg.mad_threshold
    if int(np.sum(keep_mask)) < min_keep:
        keep_mask = np.zeros(K, dtype=bool)
        keep_mask[order[:min_keep]] = True

    flagged = [cids[i] for i in range(K) if not keep_mask[i]]
    kept_idx = np.where(keep_mask)[0]

    # Apply sample-count clipping on kept clients as secondary safeguard against n_inflate
    w_kept = _clip_counts(N[kept_idx], cfg.n_clip_mult)
    mg, vg = _pooled_stats(M[kept_idx], V[kept_idx], w_kept)

    eff_w = np.zeros(K, dtype=np.float64)
    eff_w[kept_idx] = w_kept / np.sum(w_kept)

    return AggregationResult(
        mean_global=mg,
        var_global=vg,
        flagged_clients=flagged,
        client_scores=client_scores.tolist(),
        effective_weights=eff_w.tolist(),
    )


def evaluate_detection(
    true_malicious: Sequence[int],
    flagged_clients: Sequence[int],
    total_clients: int,
) -> Dict[str, float]:
    """Compute precision, recall, F1, and FPR for attacker detection."""
    true_set = set(true_malicious)
    pred_set = set(flagged_clients)
    all_set = set(range(total_clients))

    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    tn = len((all_set - true_set) - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if len(true_set) == 0 else 0.0)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "det_tp": float(tp),
        "det_fp": float(fp),
        "det_fn": float(fn),
        "det_precision": precision,
        "det_recall": recall,
        "det_f1": f1,
        "det_fpr": fpr,
    }
