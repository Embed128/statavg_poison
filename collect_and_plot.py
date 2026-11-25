"""Aggregate experiment outputs into CSV tables, LaTeX snippets, and 300-DPI paper figures."""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import List

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_statavg")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_raw_runs(raw_dir: Path) -> List[dict]:
    files = sorted(glob.glob(str(raw_dir / "*.json")))
    runs = []
    for f in files:
        with open(f, "r", encoding="utf-8") as fp:
            runs.append(json.load(fp))
    return runs


def summarize_runs(runs: List[dict], out_dir: Path) -> pd.DataFrame:
    rows = []
    for r in runs:
        c = r["config"]
        rows.append({
            "group": r["group"],
            "label": r["label"],
            "seed": c["seed"],
            "strategy": c["strategy"],
            "weight_agg": c["weight_agg"],
            "attack_type": c["attack_type"],
            "malicious_frac": c["malicious_frac"],
            "mean_alpha": c["mean_alpha"],
            "n_inflate": c["n_inflate"],
            "defense_type": c["defense_type"],
            "accuracy": r["final_accuracy"],
            "macro_f1": r["final_macro_f1"],
            "weighted_f1": r["final_weighted_f1"],
            "mean_rel_err": r["mean_rel_err"],
            "var_log_err": r["var_log_err"],
            "det_precision": r["det_precision"],
            "det_recall": r["det_recall"],
            "det_f1": r["det_f1"],
            "det_fpr": r["det_fpr"],
            "elapsed_sec": r["elapsed_sec"],
        })
    df_raw = pd.DataFrame(rows)
    df_raw.to_csv(out_dir / "results_all_seeds.csv", index=False)

    grp_cols = ["group", "label", "strategy", "weight_agg", "attack_type", "malicious_frac", "mean_alpha", "defense_type"]
    summary = (
        df_raw.groupby(grp_cols, as_index=False)
        .agg(
            n_seeds=("seed", "count"),
            acc_mean=("accuracy", "mean"),
            acc_std=("accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            mean_err=("mean_rel_err", "mean"),
            var_log_err=("var_log_err", "mean"),
            det_precision=("det_precision", "mean"),
            det_recall=("det_recall", "mean"),
            det_f1=("det_f1", "mean"),
            det_fpr=("det_fpr", "mean"),
        )
        .fillna(0.0)
    )
    summary.to_csv(out_dir / "results_summary.csv", index=False)
    print(f"Saved summary table ({len(summary)} configs) -> {out_dir / 'results_summary.csv'}")
    return summary


def plot_convergence(runs: List[dict], fig_dir: Path) -> None:
    """Figure 1: Accuracy & Macro-F1 vs FL rounds for key configurations."""
    target_labels = [
        ("FedAvg_clean", "FedAvg (local norm)", "#64748B", "--"),
        ("StatAvg_clean_vanilla", "StatAvg (clean)", "#0E7490", "-"),
        ("StatAvg_clean", "StatAvg (clean)", "#0E7490", "-"),
        ("hijack_m20_vanilla", "StatAvg + Hijack 20% (No Defense)", "#C2416B", "-"),
        ("StatAvg_hijack_20_vanilla", "StatAvg + Hijack 20% (No Defense)", "#C2416B", "-"),
        ("hijack_m20_krum", "StatAvg + Hijack 20% + Multi-Krum", "#B08A12", ":"),
        ("StatAvg_hijack_20_krum", "StatAvg + Hijack 20% + Multi-Krum", "#B08A12", ":"),
        ("hijack_m20_mad", "StatAvg + Hijack 20% + MAD (Ours)", "#E8701A", "-"),
        ("StatAvg_hijack_20_mad", "StatAvg + Hijack 20% + MAD (Ours)", "#E8701A", "-"),
    ]

    by_label = {}
    for r in runs:
        by_label.setdefault(r["label"], []).append(r)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=300)
    plotted_names = set()

    for key, disp, col, ls in target_labels:
        if key not in by_label or disp in plotted_names:
            continue
        plotted_names.add(disp)
        group_runs = by_label[key]
        rounds = [h["round"] for h in group_runs[0]["history"]]
        accs = np.array([[h["accuracy"] * 100 for h in gr["history"]] for gr in group_runs])
        f1s = np.array([[h["macro_f1"] * 100 for h in gr["history"]] for gr in group_runs])
        lw = 2.6 if "MAD" in disp else 2.0

        for ax, arr in zip(axes, (accs, f1s)):
            mu = arr.mean(axis=0)
            sd = arr.std(axis=0)
            ax.plot(rounds, mu, label=disp, color=col, linestyle=ls, linewidth=lw)
            if len(group_runs) > 1:
                ax.fill_between(rounds, mu - sd, mu + sd, color=col, alpha=0.14)

    axes[0].set_title("Global Test Accuracy vs. FL Rounds", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Accuracy (%)")
    axes[1].set_title("Global Test Macro-F1 vs. FL Rounds", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Macro-F1 (%)")
    for ax in axes:
        ax.set_xlabel("Communication Round")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.set_ylim(0, 102)
    axes[1].legend(loc="lower right", fontsize=8.5, frameon=True)
    plt.tight_layout()
    fig.savefig(fig_dir / "fig1_convergence_curves.png", dpi=300)
    fig.savefig(fig_dir / "fig1_convergence_curves.pdf")
    plt.close(fig)
    print(f"Saved -> {fig_dir / 'fig1_convergence_curves.png'}")


def plot_attack_heatmap(summary: pd.DataFrame, fig_dir: Path) -> None:
    """Figure 2: Macro-F1 Heatmap across Attack Types x Malicious Ratios (Vanilla vs MAD)."""
    sub = summary[summary["group"] == "attack_matrix"]
    if sub.empty:
        return

    attacks = ["mean_shift", "var_inflate", "var_shrink", "hijack", "targeted_shift"]
    attacks = [a for a in attacks if a in sub["attack_type"].values]
    fracs = sorted(sub["malicious_frac"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), dpi=300, sharey=True)
    for ax, def_t, title in zip(
        axes,
        ("vanilla", "mad_filter"),
        ("Vanilla StatAvg (No Defense) — Macro-F1 (%)", "StatAvg + MAD Filter (Ours) — Macro-F1 (%)"),
    ):
        mat = np.zeros((len(attacks), len(fracs)))
        for i, a in enumerate(attacks):
            for j, mf in enumerate(fracs):
                row = sub[(sub["attack_type"] == a) & (sub["malicious_frac"] == mf) & (sub["defense_type"] == def_t)]
                mat[i, j] = row["macro_f1_mean"].values[0] * 100 if len(row) else np.nan

        im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(len(fracs)))
        ax.set_xticklabels([f"{int(round(f * 100))}%" for f in fracs])
        ax.set_yticks(range(len(attacks)))
        ax.set_yticklabels(attacks)
        ax.set_xlabel("Malicious Client Ratio")
        ax.set_title(title, fontsize=10.5, fontweight="bold")
        for i in range(len(attacks)):
            for j in range(len(fracs)):
                if not np.isnan(mat[i, j]):
                    txt_col = "white" if mat[i, j] < 35 else "#14213D"
                    ax.text(j, i, f"{mat[i, j]:.1f}%", ha="center", va="center", color=txt_col, fontweight="bold", fontsize=9.5)

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, label="Macro-F1 (%)")
    plt.savefig(fig_dir / "fig2_attack_defense_heatmap.png", dpi=300, bbox_inches="tight")
    plt.savefig(fig_dir / "fig2_attack_defense_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {fig_dir / 'fig2_attack_defense_heatmap.png'}")


def plot_defense_bar(summary: pd.DataFrame, fig_dir: Path) -> None:
    """Figure 3: Head-to-head comparison of defenses under worst-case Hijack attack."""
    # Pick 30% if present, else 20%
    target_frac = 0.3 if 0.3 in summary["malicious_frac"].values else 0.2
    sub = summary[(summary["attack_type"] == "hijack") & (summary["malicious_frac"] == target_frac)]
    if sub.empty:
        return

    order = [
        ("vanilla", "fedavg", "Vanilla\n(No Defense)", "#C2416B"),
        ("vanilla", "krum", "Multi-Krum\n(Model-Level)", "#B08A12"),
        ("n_clip", "fedavg", "n-Clip\nOnly", "#64748B"),
        ("median", "fedavg", "Coord.\nMedian", "#1F6FB2"),
        ("trimmed", "fedavg", "Trimmed\nMean", "#2E9E6A"),
        ("mad_filter", "fedavg", "MAD Filter\n(Ours)", "#E8701A"),
    ]
    labels, accs, f1s, colors = [], [], [], []
    for dt, wa, disp, col in order:
        row = sub[(sub["defense_type"] == dt) & (sub["weight_agg"] == wa)]
        if len(row):
            labels.append(disp)
            accs.append(row["acc_mean"].values[0] * 100)
            f1s.append(row["macro_f1_mean"].values[0] * 100)
            colors.append(col)

    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(9.5, 4.3), dpi=300)
    b1 = ax.bar(x - w / 2, accs, width=w, color=colors, alpha=0.85, label="Accuracy (%)")
    b2 = ax.bar(x + w / 2, f1s, width=w, color=colors, alpha=0.45, edgecolor=colors, hatch="//", linewidth=1.2, label="Macro-F1 (%)")

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1.2, f"{h:.1f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Score (%)")
    ax.set_title(f"Defense Comparison Under {int(target_frac*100)}% Hijack Attack (+5x Sample Inflation)", fontsize=11, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper left")
    plt.tight_layout()
    fig.savefig(fig_dir / "fig3_defense_comparison.png", dpi=300)
    fig.savefig(fig_dir / "fig3_defense_comparison.pdf")
    plt.close(fig)
    print(f"Saved -> {fig_dir / 'fig3_defense_comparison.png'}")


def plot_ablation_strength(summary: pd.DataFrame, fig_dir: Path) -> None:
    """Figure 4: Attack strength ablation (mean_shift alpha = 2..50) Vanilla vs MAD."""
    sub = summary[summary["group"] == "ablation_strength"]
    if sub.empty:
        return
    alphas = sorted(sub["mean_alpha"].unique())
    fig, ax = plt.subplots(figsize=(7.2, 4.1), dpi=300)
    for def_t, disp, col, mk in [
        ("vanilla", "Vanilla StatAvg (No Defense)", "#C2416B", "o"),
        ("mad_filter", "StatAvg + MAD Filter (Ours)", "#E8701A", "s"),
    ]:
        ys = []
        for a in alphas:
            row = sub[(sub["mean_alpha"] == a) & (sub["defense_type"] == def_t)]
            ys.append(row["macro_f1_mean"].values[0] * 100 if len(row) else np.nan)
        ax.plot(alphas, ys, marker=mk, linewidth=2.3, markersize=7, color=col, label=disp)
        for x_v, y_v in zip(alphas, ys):
            if not np.isnan(y_v):
                ax.annotate(f"{y_v:.1f}%", (x_v, y_v), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8.5, color=col, fontweight="bold")
    ax.set_xlabel(r"Mean-Shift Attack Strength ($\alpha$ in $\tilde{\mu}_k = \mu_k + \alpha\sigma_k$)")
    ax.set_ylabel("Macro-F1 (%)")
    ax.set_title("Sensitivity to Attack Strength (20% Malicious Clients, +5x Sample Inflation)", fontsize=10.5, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="lower left")
    plt.tight_layout()
    fig.savefig(fig_dir / "fig4_ablation_strength.png", dpi=300)
    fig.savefig(fig_dir / "fig4_ablation_strength.pdf")
    plt.close(fig)
    print(f"Saved -> {fig_dir / 'fig4_ablation_strength.png'}")


def write_markdown_report(summary: pd.DataFrame, out_dir: Path) -> None:
    lines = ["# StatAvg 统计量投毒与鲁棒聚合实验汇总报告\n"]
    lines.append("以下表格由 `collect_and_plot.py` 自动从真实运行结果汇总生成（均值 ± 标准差）。\n")
    for grp, title in [
        ("baseline", "1. 基准对比（FedAvg vs Clean StatAvg）"),
        ("clean_overhead", "2. 干净场景（0% 恶意）各防御精度代价"),
        ("attack_matrix", "3. 主攻防矩阵（5 种攻击 × 10%/20%/30% 恶意比例，Vanilla vs MAD）"),
        ("defense_compare", "4. 最坏场景（30% Hijack）各统计量防御横向对比"),
        ("krum_bypass", "5. 模型层拜占庭防御（Multi-Krum）失效验证"),
        ("ablation_strength", "6. 攻击强度消融实验（mean_shift α = 2..50）"),
    ]:
        sub = summary[summary["group"] == grp]
        if sub.empty:
            continue
        lines.append(f"\n## {title}\n")
        lines.append("| 配置 (Label) | 种子数 | Accuracy (%) | Macro-F1 (%) | 均值误差 | 方差对数误差 | 攻击者检测 F1 | 误报率 FPR |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for _, r in sub.iterrows():
            acc_s = f"{r['acc_mean']*100:.2f} ± {r['acc_std']*100:.2f}"
            f1_s = f"{r['macro_f1_mean']*100:.2f} ± {r['macro_f1_std']*100:.2f}"
            lines.append(
                f"| `{r['label']}` | {int(r['n_seeds'])} | {acc_s} | {f1_s} | "
                f"{r['mean_err']:.3f} | {r['var_log_err']:.3f} | {r['det_f1']:.3f} | {r['det_fpr']:.3f} |"
            )
    report_path = out_dir / "EXPERIMENT_REPORT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved report -> {report_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    out_dir = Path(args.results_dir)
    raw_dir = out_dir / "raw"
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    runs = load_raw_runs(raw_dir)
    if not runs:
        print("No raw JSON runs found in", raw_dir)
        return

    summary = summarize_runs(runs, out_dir)
    plot_convergence(runs, fig_dir)
    plot_attack_heatmap(summary, fig_dir)
    plot_defense_bar(summary, fig_dir)
    plot_ablation_strength(summary, fig_dir)
    write_markdown_report(summary, out_dir)
    print("\nSummary Preview:")
    print(summary[["group", "label", "acc_mean", "macro_f1_mean", "det_f1"]].to_string(index=False))


if __name__ == "__main__":
    main()
