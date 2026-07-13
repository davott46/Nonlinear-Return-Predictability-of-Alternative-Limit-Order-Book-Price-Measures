from pathlib import Path
import numpy as np
import pandas as pd
import os


def initialize_run(
    output_root: str,
    model_version: str,
    feature_set_id: str,
    normalization_id: str
) -> int:
    """
    Creates sequential run_id and appends metadata to registry.
    """

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    registry_path = output_root / "run_registry.csv"

    if registry_path.exists():
        existing_runs = pd.read_csv(registry_path)
        run_id = 0 if existing_runs.empty else int(existing_runs["run_id"].max()) + 1
    else:
        run_id = 0

    pd.DataFrame([{
        "run_id": run_id,
        "model_version": model_version,
        "feature_set_id": feature_set_id,
        "normalization_id": normalization_id
    }]).to_csv(
        registry_path,
        mode="a",
        header=not registry_path.exists(),
        index=False
    )

    return run_id


def create_output_dirs(
    output_root: str,
    run_id: str,
) -> dict:
    """
    Creates output directory structure for a run.
    """
    output_root = Path(output_root)

    dirs = {
        "tick": output_root / "TickLevel" / str(run_id),
        "daily": output_root / "DailyDiagnostics" / str(run_id),
    }

    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    return dirs


def save_table(
    df,
    root_dir,
    filename,
    partition_cols=None,
    file_format="parquet",
    compression="snappy",
    index=False
) -> None:

    save_dir = root_dir

    if partition_cols is not None:
        for key, value in partition_cols.items():
            save_dir = f"{save_dir}/{key}={value}"

    os.makedirs(save_dir, exist_ok=True)

    path = f"{save_dir}/{filename}"

    if file_format == "parquet":
        df.to_parquet(
            path,
            compression=compression,
            index=index
        )

    elif file_format == "feather":
        # to_feather only supports zstd / lz4 / uncompressed (not snappy),
        # so fall back to zstd when a parquet-style codec is passed.
        feather_comp = compression if compression in ("zstd", "lz4", "uncompressed") else "zstd"
        df.reset_index(drop=True).to_feather(path, compression=feather_comp)

    else:
        raise ValueError(f"Unsupported file format: {file_format}")


def load_day_cache(
        data_root,
        symbol,
        days,
        feature_cols,
        target_cols
) -> dict[str, dict[str, np.ndarray]]:
    """Load all of one symbol's days into {day: {"X", "Y", "timestamp"}}."""
    day_cache = {}

    for day in days:
        df = pd.read_parquet(f"{data_root}/{symbol}/{day}.parquet").sort_values("Timestamp")
        df = df.dropna(subset=feature_cols + target_cols)

        day_cache[day] = {
            "X": df[feature_cols].to_numpy(dtype=np.float32),
            "Y": df[target_cols].to_numpy(np.float32),
            "timestamp": df["Timestamp"].to_numpy(),
        }

    return day_cache


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
    """Per-target metrics vs. the zero-return benchmark, as Table-1 rows.

    Benchmark = zero-return (martingale) forecast: mse_bench = mean(Y_test^2).
    """
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


def save_tick_residuals(
        resid,
        timestamps,
        target_cols,
        tick_output_dir,
        test_day, symbol
) -> None:
    """Minimal tick-level store for Diebold-Mariano: Timestamp + one residual
    column per target. y_true is NOT stored.

    float16 residuals: ~2x smaller than float32, aggregate MSE error ~1e-6
    (negligible for DM). feather+zstd is the smallest lossless container here
    (parquet inflates incompressible floats).
    """
    tick_df = pd.DataFrame(resid.astype(np.float16), columns=target_cols)
    tick_df.insert(0, "Timestamp", timestamps)

    save_table(
        df=tick_df,
        root_dir=tick_output_dir,
        filename="tick_residuals.feather",
        partition_cols={
            "date": test_day,
            "symbol": symbol
        },
        file_format="feather",
        compression="zstd"
    )