"""Federated Learning simulator for FedAvg, StatAvg, Statistics Poisoning, and Defenses.

Key capabilities:
  - Exact FedAvg (local per-client standardization) vs StatAvg (Round-1 global statistics)
  - Supports 4+1 statistics poisoning attacks + sample-count inflation (n_inflate)
  - Supports 5 statistics aggregation defenses (vanilla, n_clip, median, trimmed, mad_filter)
  - Supports model-level Byzantine defense baseline (Multi-Krum on model weights) to
    empirically prove that model-level defenses fail to stop Round-1 statistics poisoning.
  - Reports global test Accuracy, Macro-F1, Weighted-F1, Per-Class Recall,
    normalization estimation error, and MAD attacker detection metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch.utils.data import DataLoader, TensorDataset

from .dataset import FederatedDataset
from .model import StatAvgMLP
from .poison import AttackConfig, poison_statistics
from .robust_stats import DefenseConfig, aggregate_statistics, evaluate_detection


@dataclass
class RunConfig:
    strategy: str = "statavg"         # "fedavg" or "statavg"
    weight_agg: str = "fedavg"        # "fedavg" or "krum" (model-level Byzantine baseline)
    num_rounds: int = 40
    local_epochs: int = 2
    batch_size: int = 512
    learning_rate: float = 0.002
    malicious_frac: float = 0.0       # fraction of malicious clients (0.0, 0.1, 0.2, 0.3)
    attack_type: str = "none"
    mean_alpha: float = 20.0
    var_beta: float = 100.0
    n_inflate: float = 1.0
    defense_type: str = "vanilla"
    trim_ratio: float = 0.2
    mad_threshold: float = 3.5
    n_clip_mult: float = 2.0
    use_class_weights: bool = True    # lightweight alternative to SMOTE for imbalanced IDS
    seed: int = 42
    eval_every: int = 1


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _get_weights(model: nn.Module) -> List[np.ndarray]:
    return [p.detach().cpu().numpy().copy() for p in model.parameters()]


def _set_weights(model: nn.Module, weights: Sequence[np.ndarray]) -> None:
    with torch.no_grad():
        for p, w in zip(model.parameters(), weights):
            p.copy_(torch.from_numpy(w).to(p.dtype))


def _fedavg_weights(
    client_weights: List[List[np.ndarray]], client_ns: List[int]
) -> List[np.ndarray]:
    total = float(sum(client_ns))
    agg = [np.zeros_like(w, dtype=np.float64) for w in client_weights[0]]
    for w_list, n in zip(client_weights, client_ns):
        factor = n / total
        for i, w in enumerate(w_list):
            agg[i] += factor * w
    return [a.astype(np.float32) for a in agg]


def _multikrum_weights(
    client_weights: List[List[np.ndarray]], f_est: int
) -> List[np.ndarray]:
    """Model-level Multi-Krum aggregation on flattened client parameter vectors."""
    K = len(client_weights)
    f = min(max(1, f_est), max(1, (K - 3) // 2))
    m_keep = max(1, K - f)
    vecs = np.stack([np.concatenate([w.ravel() for w in wl]) for wl in client_weights])
    # Pairwise squared Euclidean distances
    diff = vecs[:, None, :] - vecs[None, :, :]
    dist_sq = np.sum(diff ** 2, axis=-1)
    n_neighbors = max(1, K - f - 2)
    scores = np.zeros(K, dtype=np.float64)
    for i in range(K):
        sorted_d = np.sort(dist_sq[i])
        scores[i] = np.sum(sorted_d[1 : 1 + n_neighbors])
    selected = np.argsort(scores)[:m_keep]
    agg = [np.zeros_like(w, dtype=np.float64) for w in client_weights[0]]
    for idx in selected:
        for i, w in enumerate(client_weights[idx]):
            agg[i] += w / m_keep
    return [a.astype(np.float32) for a in agg]


def _normalize(x: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    std = np.sqrt(np.maximum(var, 1e-10))
    z = (x - mean[None, :]) / std[None, :]
    # Prevent numerical NaN overflow under extreme var_shrink while preserving saturation impact
    return np.clip(z, -1e4, 1e4).astype(np.float32)


def _train_local(
    model: nn.Module,
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    class_weights: Optional[torch.Tensor],
    seed: int,
) -> List[np.ndarray]:
    model.train()
    gen = torch.Generator()
    gen.manual_seed(seed)
    ds = TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr.astype(np.int64)))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=gen)
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.99, 0.999), eps=1e-8)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    for _ in range(epochs):
        for bx, by in loader:
            opt.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                opt.step()
    return _get_weights(model)


@torch.no_grad()
def _evaluate_model(
    model: nn.Module,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    num_classes: int,
) -> Dict[str, Any]:
    model.eval()
    bx = torch.from_numpy(x_eval)
    logits = model(bx)
    preds = torch.argmax(logits, dim=1).cpu().numpy()
    acc = float(accuracy_score(y_eval, preds))
    macro_f1 = float(f1_score(y_eval, preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_eval, preds, average="weighted", zero_division=0))
    per_class_rec = recall_score(
        y_eval, preds, labels=list(range(num_classes)), average=None, zero_division=0
    ).tolist()
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class_recall": per_class_rec,
    }


def run_federated_experiment(
    fed_data: FederatedDataset,
    cfg: RunConfig,
) -> Dict[str, Any]:
    """Run a single federated learning experiment end-to-end."""
    _set_seed(cfg.seed)
    K = len(fed_data.client_splits)
    D = fed_data.num_features
    C = fed_data.num_classes

    # Determine malicious clients deterministically for this seed
    num_mal = int(round(K * cfg.malicious_frac))
    if cfg.attack_type == "none" or cfg.strategy == "fedavg":
        num_mal = 0
    rng = np.random.default_rng(cfg.seed + 999)
    mal_clients = sorted(rng.choice(K, size=num_mal, replace=False).tolist()) if num_mal > 0 else []
    mal_set = set(mal_clients)

    # Compute honest local statistics for all clients (Round 1)
    honest_means, honest_vars, honest_ns = [], [], []
    for x_tr, _, _, _ in fed_data.client_splits:
        honest_means.append(np.mean(x_tr, axis=0))
        honest_vars.append(np.maximum(np.var(x_tr, axis=0), 1e-8))
        honest_ns.append(len(x_tr))

    # Ground-truth clean global statistics (for measuring normalization distortion)
    clean_res = aggregate_statistics(
        honest_means, honest_vars, honest_ns, DefenseConfig(defense_type="vanilla")
    )
    true_mean_g, true_var_g = clean_res.mean_global, clean_res.var_global
    true_std_g = np.sqrt(np.maximum(true_var_g, 1e-8))

    # Build client-side normalized datasets
    flagged_clients: List[int] = []
    det_metrics = evaluate_detection(mal_clients, [], K)
    mean_rel_err = 0.0
    var_log_err = 0.0

    if cfg.strategy == "fedavg":
        # Each client normalizes using ONLY its own local statistics
        client_norm_train = [
            (_normalize(x_tr, m, v), y_tr)
            for (x_tr, y_tr, _, _), m, v in zip(fed_data.client_splits, honest_means, honest_vars)
        ]
        # Evaluate global test set using average local stats (standard FedAvg inference)
        x_test_norm = _normalize(fed_data.x_test, true_mean_g, true_var_g)
    elif cfg.strategy == "statavg":
        # Round 1: Malicious clients submit poisoned statistics
        attack_cfg = AttackConfig(
            attack_type=cfg.attack_type,
            mean_alpha=cfg.mean_alpha,
            var_beta=cfg.var_beta,
            n_inflate=cfg.n_inflate,
        )
        rep_means, rep_vars, rep_ns = [], [], []
        for k in range(K):
            if k in mal_set:
                pm, pv, pn = poison_statistics(
                    honest_means[k], honest_vars[k], honest_ns[k], attack_cfg
                )
                rep_means.append(pm)
                rep_vars.append(pv)
                rep_ns.append(pn)
            else:
                rep_means.append(honest_means[k])
                rep_vars.append(honest_vars[k])
                rep_ns.append(honest_ns[k])

        # Server aggregates statistics using chosen defense
        def_cfg = DefenseConfig(
            defense_type=cfg.defense_type,
            trim_ratio=cfg.trim_ratio,
            mad_threshold=cfg.mad_threshold,
            n_clip_mult=cfg.n_clip_mult,
        )
        agg_res = aggregate_statistics(rep_means, rep_vars, rep_ns, def_cfg)
        mg, vg = agg_res.mean_global, agg_res.var_global
        flagged_clients = agg_res.flagged_clients
        det_metrics = evaluate_detection(mal_clients, flagged_clients, K)

        # Measure distortion in standardized units
        mean_rel_err = float(np.mean(np.abs(mg - true_mean_g) / true_std_g))
        var_log_err = float(
            np.mean(np.abs(np.log10(np.maximum(vg, 1e-10)) - np.log10(np.maximum(true_var_g, 1e-10))))
        )

        # All clients normalize with the broadcast global statistics (mg, vg)
        client_norm_train = [
            (_normalize(x_tr, mg, vg), y_tr)
            for (x_tr, y_tr, _, _) in fed_data.client_splits
        ]
        # Global test set is also normalized using the deployed global statistics (mg, vg)
        x_test_norm = _normalize(fed_data.x_test, mg, vg)
    else:
        raise ValueError(f"Unknown strategy: {cfg.strategy}")

    # Optional mild inverse-frequency class weights per client (avoids minority collapse)
    client_class_weights: List[Optional[torch.Tensor]] = []
    for _, y_tr in client_norm_train:
        if cfg.use_class_weights:
            counts = np.bincount(y_tr, minlength=C).astype(np.float64)
            w = np.where(counts > 0, len(y_tr) / (C * np.maximum(counts, 1.0)), 0.0)
            w = np.clip(w, 0.25, 4.0)
            client_class_weights.append(torch.tensor(w, dtype=torch.float32))
        else:
            client_class_weights.append(None)

    # Federated training loop (Rounds 1..num_rounds)
    global_model = StatAvgMLP(D, C)
    global_weights = _get_weights(global_model)
    history: List[Dict[str, Any]] = []

    for rnd in range(1, cfg.num_rounds + 1):
        round_weights = []
        round_ns = []
        for k in range(K):
            local_model = StatAvgMLP(D, C)
            _set_weights(local_model, global_weights)
            x_tr, y_tr = client_norm_train[k]
            wk = _train_local(
                local_model,
                x_tr,
                y_tr,
                epochs=cfg.local_epochs,
                batch_size=cfg.batch_size,
                lr=cfg.learning_rate,
                class_weights=client_class_weights[k],
                seed=cfg.seed * 1000 + rnd * 31 + k,
            )
            round_weights.append(wk)
            round_ns.append(len(x_tr))

        if cfg.weight_agg == "krum":
            global_weights = _multikrum_weights(round_weights, f_est=max(1, num_mal))
        else:
            global_weights = _fedavg_weights(round_weights, round_ns)

        _set_weights(global_model, global_weights)

        if rnd % cfg.eval_every == 0 or rnd == cfg.num_rounds:
            ev = _evaluate_model(global_model, x_test_norm, fed_data.y_test, C)
            ev["round"] = rnd
            history.append(ev)

    final_ev = history[-1]
    return {
        "config": asdict(cfg),
        "malicious_clients": mal_clients,
        "flagged_clients": flagged_clients,
        "mean_rel_err": mean_rel_err,
        "var_log_err": var_log_err,
        "final_accuracy": final_ev["accuracy"],
        "final_macro_f1": final_ev["macro_f1"],
        "final_weighted_f1": final_ev["weighted_f1"],
        "final_per_class_recall": final_ev["per_class_recall"],
        "class_names": fed_data.class_names,
        **det_metrics,
        "history": history,
    }
