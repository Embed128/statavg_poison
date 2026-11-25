"""Generate synthetic TON_IoT Linux Memory-like CSV for pipeline verification."""

from __future__ import annotations

import argparse
from pathlib import Path

from statavg_poison.dataset import generate_synthetic_ton_iot


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset/synthetic_dataset.csv")
    ap.add_argument("--rows", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = generate_synthetic_ton_iot(n_samples=args.rows, seed=args.seed, save_path=args.out)
    print(f"Generated {len(df)} rows -> {Path(args.out).resolve()}")
    print("Class distribution:")
    print(df["type"].value_counts())


if __name__ == "__main__":
    main()
