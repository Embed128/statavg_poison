"""Main experiment runner for StatAvg Statistics Poisoning & Robust Aggregation.

Usage:
  python3 run_experiments.py --dataset dataset/dataset.csv --mode smoke
  python3 run_experiments.py --dataset dataset/dataset.csv --mode main --seeds 42 43 44 --workers 4
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from statavg_poison.dataset import prepare_federated_dataset
from statavg_poison.simulator import RunConfig, run_federated_experiment


def build_experiment_plan(mode: str, num_rounds: int, seeds: List[int]) -> List[Dict[str, Any]]:
    plan: List[Dict[str, Any]] = []

    if mode == "smoke":
        r = min(num_rounds, 15)
        s = seeds[0]
        plan = [
            {"group": "baseline", "label": "FedAvg_clean", "cfg": RunConfig(strategy="fedavg", num_rounds=r, seed=s)},
            {"group": "baseline", "label": "StatAvg_clean", "cfg": RunConfig(strategy="statavg", num_rounds=r, seed=s)},
            {"group": "attack", "label": "StatAvg_hijack_20_vanilla", "cfg": RunConfig(strategy="statavg", num_rounds=r, malicious_frac=0.2, attack_type="hijack", n_inflate=5.0, defense_type="vanilla", seed=s)},
            {"group": "attack", "label": "StatAvg_hijack_20_krum", "cfg": RunConfig(strategy="statavg", weight_agg="krum", num_rounds=r, malicious_frac=0.2, attack_type="hijack", n_inflate=5.0, defense_type="vanilla", seed=s)},
            {"group": "defense", "label": "StatAvg_hijack_20_median", "cfg": RunConfig(strategy="statavg", num_rounds=r, malicious_frac=0.2, attack_type="hijack", n_inflate=5.0, defense_type="median", seed=s)},
            {"group": "defense", "label": "StatAvg_hijack_20_trimmed", "cfg": RunConfig(strategy="statavg", num_rounds=r, malicious_frac=0.2, attack_type="hijack", n_inflate=5.0, defense_type="trimmed", seed=s)},
            {"group": "defense", "label": "StatAvg_hijack_20_mad", "cfg": RunConfig(strategy="statavg", num_rounds=r, malicious_frac=0.2, attack_type="hijack", n_inflate=5.0, defense_type="mad_filter", seed=s)},
        ]
        return plan

    # FULL PAPER MATRIX (--mode main)
    for s in seeds:
        # 1. Clean baselines & clean-case defense overhead
        plan.append({"group": "baseline", "label": "FedAvg_clean", "cfg": RunConfig(strategy="fedavg", num_rounds=num_rounds, seed=s)})
        for def_name in ("vanilla", "n_clip", "median", "trimmed", "mad_filter"):
            plan.append({
                "group": "clean_overhead",
                "label": f"StatAvg_clean_{def_name}",
                "cfg": RunConfig(strategy="statavg", num_rounds=num_rounds, malicious_frac=0.0, attack_type="none", defense_type=def_name, seed=s),
            })

        # 2. Attack matrix: 5 attacks x {10%, 20%, 30%} under vanilla vs mad_filter
        for atk in ("mean_shift", "var_inflate", "var_shrink", "hijack", "targeted_shift"):
            for mfrac in (0.1, 0.2, 0.3):
                pct = int(round(mfrac * 100))
                plan.append({
                    "group": "attack_matrix",
                    "label": f"{atk}_m{pct}_vanilla",
                    "cfg": RunConfig(strategy="statavg", num_rounds=num_rounds, malicious_frac=mfrac, attack_type=atk, n_inflate=5.0, defense_type="vanilla", seed=s),
                })
                plan.append({
                    "group": "attack_matrix",
                    "label": f"{atk}_m{pct}_mad",
                    "cfg": RunConfig(strategy="statavg", num_rounds=num_rounds, malicious_frac=mfrac, attack_type=atk, n_inflate=5.0, defense_type="mad_filter", seed=s),
                })

        # 3. Defense head-to-head under worst-case (30% hijack with 5x n_inflate) + Multi-Krum model defense
        for def_name in ("n_clip", "median", "trimmed"):
            plan.append({
                "group": "defense_compare",
                "label": f"hijack_m30_{def_name}",
                "cfg": RunConfig(strategy="statavg", num_rounds=num_rounds, malicious_frac=0.3, attack_type="hijack", n_inflate=5.0, defense_type=def_name, seed=s),
            })
        for mfrac in (0.1, 0.2, 0.3):
            pct = int(round(mfrac * 100))
            plan.append({
                "group": "krum_bypass",
                "label": f"hijack_m{pct}_krum",
                "cfg": RunConfig(strategy="statavg", weight_agg="krum", num_rounds=num_rounds, malicious_frac=mfrac, attack_type="hijack", n_inflate=5.0, defense_type="vanilla", seed=s),
            })

    # 4. Attack strength ablation (seed 0)
    s0 = seeds[0]
    for alpha in (2.0, 5.0, 10.0, 20.0, 50.0):
        for def_name in ("vanilla", "mad_filter"):
            plan.append({
                "group": "ablation_strength",
                "label": f"mean_shift_a{int(alpha)}_m20_{def_name}",
                "cfg": RunConfig(strategy="statavg", num_rounds=num_rounds, malicious_frac=0.2, attack_type="mean_shift", mean_alpha=alpha, n_inflate=5.0, defense_type=def_name, seed=s0),
            })

    return plan


# Per-worker dataset cache so each process loads & splits each seed once
_WORKER_DS_CACHE: Dict[int, Any] = {}


def _run_one_task(args_tuple: Tuple[Dict[str, Any], str, int, float, int | None, str]) -> Dict[str, Any]:
    item, dataset_path, num_clients, alpha, max_rows, raw_dir_str = args_tuple
    torch.set_num_threads(1)
    cfg: RunConfig = item["cfg"]
    if cfg.seed not in _WORKER_DS_CACHE:
        _WORKER_DS_CACHE[cfg.seed] = prepare_federated_dataset(
            csv_path=dataset_path,
            num_clients=num_clients,
            partition_mode="dirichlet",
            dirichlet_alpha=alpha,
            max_rows=max_rows,
            seed=cfg.seed,
        )
    fed_data = _WORKER_DS_CACHE[cfg.seed]

    run_id = f"{item['group']}__{item['label']}__seed{cfg.seed}"
    out_json = Path(raw_dir_str) / f"{run_id}.json"

    t_run = time.time()
    res = run_federated_experiment(fed_data, cfg)
    res["run_id"] = run_id
    res["group"] = item["group"]
    res["label"] = item["label"]
    res["dataset_path"] = dataset_path
    res["num_clients"] = num_clients
    res["dirichlet_alpha"] = alpha
    res["elapsed_sec"] = round(time.time() - t_run, 2)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

    return {
        "label": item["label"],
        "seed": cfg.seed,
        "acc": res["final_accuracy"],
        "f1": res["final_macro_f1"],
        "m_err": res["mean_rel_err"],
        "v_err": res["var_log_err"],
        "det_f1": res["det_f1"],
        "sec": res["elapsed_sec"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Path to CSV dataset")
    ap.add_argument("--out-dir", default="results", help="Output directory")
    ap.add_argument("--mode", choices=("smoke", "main"), default="smoke")
    ap.add_argument("--num-clients", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.5, help="Dirichlet alpha")
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--max-rows", type=int, default=0, help="Cap dataset size (0=use all rows)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--workers", type=int, default=4, help="Parallel worker processes")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    max_rows = None if args.max_rows <= 0 else args.max_rows
    plan = build_experiment_plan(args.mode, args.rounds, args.seeds)
    print(f"Dataset: {args.dataset} | mode={args.mode} | runs={len(plan)} | workers={args.workers}", flush=True)

    tasks = [(item, str(args.dataset), args.num_clients, args.alpha, max_rows, str(raw_dir)) for item in plan]
    t0 = time.time()

    if args.workers <= 1:
        for idx, t in enumerate(tasks, 1):
            r = _run_one_task(t)
            print(
                f"[{idx:03d}/{len(plan):03d}] {r['label']:<30s} (seed={r['seed']}) | "
                f"Acc={r['acc']*100:5.2f}% | MacroF1={r['f1']*100:5.2f}% | "
                f"StatErr(m={r['m_err']:5.2f},v={r['v_err']:4.2f}) | DetF1={r['det_f1']:.2f} | {r['sec']:.1f}s",
                flush=True,
            )
    else:
        done = 0
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_run_one_task, t) for t in tasks]
            for fut in as_completed(futs):
                done += 1
                r = fut.result()
                print(
                    f"[{done:03d}/{len(plan):03d}] {r['label']:<30s} (seed={r['seed']}) | "
                    f"Acc={r['acc']*100:5.2f}% | MacroF1={r['f1']*100:5.2f}% | "
                    f"StatErr(m={r['m_err']:5.2f},v={r['v_err']:4.2f}) | DetF1={r['det_f1']:.2f} | {r['sec']:.1f}s",
                    flush=True,
                )

    print(f"\nCompleted {len(plan)} runs in {time.time() - t0:.1f}s -> {out_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
