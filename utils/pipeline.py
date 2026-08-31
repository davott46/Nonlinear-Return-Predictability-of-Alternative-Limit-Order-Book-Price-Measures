"""Shared pipeline pieces: day loading, run directories, daily diagnostics, target scale."""
from datetime import datetime, timezone
from pathlib import Path
import json
import os

import numpy as np
import pandas as pd

# Raw log returns at 100ms too small for XGBoost:
# Fit on y * TARGET_SCALE, then divide predictions by it.
TARGET_SCALE = 1e4


def load_day_cache(data_root, symbol, days, feature_cols, target_cols) -> dict:
    """Load one symbol's days into {day: {"X", "Y", "timestamp"}}. Missing days are skipped."""
    day_cache = {}
    for day in days:
        path = f"{data_root}/{symbol}/{day}.parquet"
        if not os.path.exists(path):
            print(f"WARNING: {symbol} {day}: no processed file, skipping")
            continue
        df = pd.read_parquet(path).sort_values("Timestamp")
        df = df.dropna(subset=feature_cols + target_cols) # drops entire row for all features/targets, even if values exist for some of them
        day_cache[day] = {
            "X": df[feature_cols].to_numpy(dtype=np.float32),
            "Y": df[target_cols].to_numpy(np.float32),
            "timestamp": df["Timestamp"].to_numpy(),
        }
    return day_cache


def start_run(output_root, name: str, manifest: dict) -> Path:
    """Create <output_root>/runs/<name>/ and write the manifest.

    `manifest` holds whatever describes the run (symbols, dates, columns, hp_config, ...);
    run_id, created_at and status are added. Returns the run dir.
    """
    run_dir = Path(output_root) / "runs" / name
    if run_dir.exists():
        print(f"resuming existing run {name}.")
        return run_dir
    run_dir.mkdir(parents=True)
    with open(run_dir / "manifest.json", "w") as f:
        json.dump({
            "run_id": name,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "runtime_seconds": None,
            "status": "running",
            **manifest,
        }, f, indent=2, default=str)
    return run_dir


def daily_diagnostic_rows(
        resid,
        Y_test,
        target_cols,
        train_day,
        test_day,
        symbol,
        run_id,
        n_train, n_test
) -> list[dict]:
    """Per-target metrics vs. the zero-return benchmark (mse_bench = mean(Y_test^2))."""
    mae = np.mean(np.abs(resid), axis=0)
    mse = np.mean(resid ** 2, axis=0)
    mse_bench = np.mean(Y_test ** 2, axis=0)

    mse_ratio = np.divide(
        mse_bench,
        mse,
        out=np.full_like(mse, np.nan),
        where=mse > 0
    )

    r2_oos = np.divide(
        mse,
        mse_bench,
        out=np.full_like(mse, np.nan),
        where=mse_bench > 0
    )

    r2_oos = 1.0 - r2_oos

    return [
        {
            "train_day": train_day,
            "test_day": test_day,
            "symbol": symbol,
            "target": target,
            "run_id": run_id,
            "n_train": int(n_train),
            "n_test": int(n_test),
            "mse": float(mse[j]),
            "mse_bench": float(mse_bench[j]),
            "mse_ratio": float(mse_ratio[j]),
            "mae": float(mae[j]),
            "r2_oos": float(r2_oos[j]),
        }
        for j, target in enumerate(target_cols)
    ]
