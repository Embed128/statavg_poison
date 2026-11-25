"""Statistics poisoning attacks against StatAvg Round-1 normalization exchange.

Implements four core statistics poisoning strategies plus sample-count inflation
and targeted high-variance feature manipulation:
  - none: honest client
  - mean_shift: shift reported local mean by alpha * local_std
  - var_inflate: inflate reported local variance by beta (squashes normalized inputs -> 0)
  - var_shrink: shrink reported local variance by 1/beta (explodes normalized inputs)
  - hijack: combined mean_shift + var_shrink (strongest untargeted disruption)
  - targeted_shift: perturb only top-k highest-variance features (stealthier)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


VALID_ATTACKS = (
    "none",
    "mean_shift",
    "var_inflate",
    "var_shrink",
    "hijack",
    "targeted_shift",
)


@dataclass
class AttackConfig:
    attack_type: str = "none"
    mean_alpha: float = 20.0      # multiplier of local std for mean shift
    var_beta: float = 100.0       # multiplier / divisor for variance inflation/shrink
    n_inflate: float = 1.0        # sample count inflation factor (e.g., 5.0 = 5x fake n)
    target_frac: float = 0.25     # fraction of features targeted for targeted_shift
    sign: float = 1.0             # shift direction (+1 or -1)


def poison_statistics(
    mean: np.ndarray,
    var: np.ndarray,
    num_examples: int,
    cfg: AttackConfig,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Return poisoned (mean, var, num_examples) for a malicious client.

    Parameters
    ----------
    mean : np.ndarray, shape (D,)
        Honest local feature means.
    var : np.ndarray, shape (D,)
        Honest local feature variances (non-negative).
    num_examples : int
        Honest local training sample count.
    cfg : AttackConfig
        Poisoning configuration.
    """
    if cfg.attack_type not in VALID_ATTACKS:
        raise ValueError(f"Unknown attack_type: {cfg.attack_type}")

    p_mean = np.asarray(mean, dtype=np.float64).copy()
    p_var = np.maximum(np.asarray(var, dtype=np.float64).copy(), 1e-8)
    p_n = int(max(1, round(num_examples * max(cfg.n_inflate, 1.0))))

    if cfg.attack_type == "none":
        return p_mean, p_var, int(num_examples)

    std = np.sqrt(p_var)

    if cfg.attack_type == "mean_shift":
        p_mean = p_mean + cfg.sign * cfg.mean_alpha * std
    elif cfg.attack_type == "var_inflate":
        p_var = p_var * (cfg.var_beta ** 2)
    elif cfg.attack_type == "var_shrink":
        p_var = np.maximum(p_var / (cfg.var_beta ** 2), 1e-10)
    elif cfg.attack_type == "hijack":
        # Shift mean strongly AND collapse reported variance (with optional n_inflate)
        p_mean = p_mean + cfg.sign * cfg.mean_alpha * std
        p_var = np.maximum(p_var / (cfg.var_beta ** 2), 1e-10)
    elif cfg.attack_type == "targeted_shift":
        d = p_mean.shape[0]
        k = max(1, int(round(d * cfg.target_frac)))
        top_idx = np.argsort(p_var)[-k:]
        p_mean[top_idx] = p_mean[top_idx] + cfg.sign * cfg.mean_alpha * std[top_idx]
        p_var[top_idx] = p_var[top_idx] * (cfg.var_beta ** 2)

    return p_mean, p_var, p_n
