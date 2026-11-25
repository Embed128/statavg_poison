"""Offline unit self-test for StatAvg statistics poisoning and robust aggregation."""

from __future__ import annotations

import numpy as np

from statavg_poison.poison import AttackConfig, poison_statistics
from statavg_poison.robust_stats import DefenseConfig, aggregate_statistics, evaluate_detection


def test_poison_and_defense() -> None:
    rng = np.random.default_rng(2026)
    K, D = 10, 8
    # Simulate 10 realistic heterogeneous clients
    base_mean = np.array([1e5, 1e4, 5e2, 8e4, 2e6, 7e5, 2e4, 1e4])
    base_std = np.array([1e4, 2e3, 1e2, 1e4, 3e5, 1e5, 5e3, 2e3])

    means = [base_mean + rng.normal(0, 0.15 * base_std) for _ in range(K)]
    vars_ = [(base_std * rng.uniform(0.85, 1.15, size=D)) ** 2 for _ in range(K)]
    ns = rng.integers(1500, 2500, size=K).tolist()

    clean = aggregate_statistics(means, vars_, ns, DefenseConfig("vanilla"))
    true_std = np.sqrt(clean.var_global)

    # 1. Clean overhead test: MAD filter should keep all honest clients
    clean_mad = aggregate_statistics(means, vars_, ns, DefenseConfig("mad_filter"))
    clean_err = np.mean(np.abs(clean_mad.mean_global - clean.mean_global) / true_std)
    assert len(clean_mad.flagged_clients) == 0, f"Unexpected false positives: {clean_mad.flagged_clients}"
    assert clean_err < 1e-6, f"Clean MAD error too high: {clean_err}"
    print(f"[PASS] Clean case: MAD false positives = 0, mean rel err = {clean_err:.2e}")

    # 2. Attack & defense tests with 20% malicious clients (clients 0, 1) + 5x n_inflate
    mal_ids = [0, 1]
    for atk in ("mean_shift", "var_inflate", "var_shrink", "hijack", "targeted_shift"):
        p_means, p_vars, p_ns = list(means), list(vars_), list(ns)
        acfg = AttackConfig(attack_type=atk, mean_alpha=20.0, var_beta=100.0, n_inflate=5.0)
        for m in mal_ids:
            p_means[m], p_vars[m], p_ns[m] = poison_statistics(means[m], vars_[m], ns[m], acfg)

        van = aggregate_statistics(p_means, p_vars, p_ns, DefenseConfig("vanilla"))
        med = aggregate_statistics(p_means, p_vars, p_ns, DefenseConfig("median"))
        trm = aggregate_statistics(p_means, p_vars, p_ns, DefenseConfig("trimmed"))
        mad = aggregate_statistics(p_means, p_vars, p_ns, DefenseConfig("mad_filter"))

        def err_metric(res):
            m_err = np.mean(np.abs(res.mean_global - clean.mean_global) / true_std)
            v_err = np.mean(np.abs(np.log10(res.var_global) - np.log10(clean.var_global)))
            return m_err, v_err

        vm, vv = err_metric(van)
        mm, mv = err_metric(med)
        tm, tv = err_metric(trm)
        dm, dv = err_metric(mad)
        det = evaluate_detection(mal_ids, mad.flagged_clients, K)

        assert (vm + vv) > 0.25, f"Attack {atk} barely distorted vanilla: {(vm, vv)}"
        assert (dm + dv) < 0.2 * (vm + vv), f"MAD failed to defend {atk}: mad={(dm, dv)} vs van={(vm, vv)}"
        assert det["det_recall"] == 1.0 and det["det_precision"] == 1.0, f"MAD detection imperfect on {atk}: {det}"
        print(
            f"[PASS] Attack={atk:<14s} | Vanilla(m={vm:5.2f}, v={vv:4.2f}) -> "
            f"Median(m={mm:4.2f}, v={mv:4.2f}) | Trimmed(m={tm:4.2f}, v={tv:4.2f}) | "
            f"MAD(m={dm:4.2f}, v={dv:4.2f}, F1={det['det_f1']:.2f})"
        )

    print("\nALL OFFLINE SELF-TESTS PASSED!")


if __name__ == "__main__":
    test_poison_and_defense()
