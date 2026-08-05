from datetime import datetime, timezone
from pathlib import Path
import json
import numpy as np
import pandas as pd
import os
import subprocess


def select_device(min_free_mib: int = 8192) -> str:
    """
    Pick the CUDA device with the most free memory, e.g. "cuda:1".

    Falls back to "cpu" when nvidia-smi is unavailable or no GPU has at
    least `min_free_mib` MiB free (the GPUs are shared with other users).
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError,
            subprocess.TimeoutExpired):
        return "cpu"

    best_idx, best_free = None, -1
    for line in out.strip().splitlines():
        try:
            idx, free = (int(v) for v in line.split(","))
        except ValueError:
            continue
        if free > best_free:
            best_idx, best_free = idx, free

    if best_idx is None or best_free < min_free_mib:
        return "cpu"
    return f"cuda:{best_idx}"


def _next_run_id(runs_dir: Path, pad: int = 2) -> str:
    """Next sequential zero-padded run id from the numeric dir names in runs_dir."""
    if runs_dir.exists():
        ids = [int(d.name) for d in runs_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    else:
        ids = []
    next_id = max(ids) + 1 if ids else 0
    return f"{next_id:0{pad}d}"


def _write_manifest(run_dir: Path, manifest: dict) -> None:
    """Atomic write: a crash mid-dump never leaves a corrupt manifest.json."""
    tmp = run_dir / "manifest.json.tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    os.replace(tmp, run_dir / "manifest.json")


def load_manifest(run_dir) -> dict:
    with open(Path(run_dir) / "manifest.json") as f:
        return json.load(f)


def start_run(
    output_root: str,
    model_family: str,
    feature_set_id: str,
    symbols: list,
    train_start: str,
    test_end: str,
    horizons: list,
    feature_cols: list,
    target_cols: list,
    training_scheme: str,
    purpose: str = "",
    hp_config: dict | None = None,
    save_tick_level_data: bool = True,
    pad: int = 2,
) -> tuple[str, dict]:
    """Create <output_root>/runs/<run_id>/ and write its manifest.json.

    The manifest is the metadata store per run
    run_id is allocated by scanning the existing runs/ dir names.
    Written with status="running"; call finalize_run when the run is done.
    `hp_config` is an opaque dict for the hyperparameters the run loaded
    (embed the values, not just a path: source files get regenerated).

    Returns (run_id, dirs) with dirs = {"run": Path, "tick": Path};
    tick/ is only created when save_tick_level_data is set.
    """
    runs_dir = Path(output_root) / "runs"
    run_id = _next_run_id(runs_dir, pad=pad)
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True)

    dirs = {"run": run_dir, "tick": run_dir / "tick"}
    if save_tick_level_data:
        dirs["tick"].mkdir()

    _write_manifest(run_dir, {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runtime_seconds": None,
        "status": "running",
        "model_family": model_family,
        "feature_set_id": feature_set_id,
        "symbols": list(symbols),
        "train_start": train_start,
        "test_end": test_end,
        "horizons": list(horizons),
        "feature_cols": list(feature_cols),
        "target_cols": list(target_cols),
        "training_scheme": training_scheme,
        "save_tick_level_data": save_tick_level_data,
        "purpose": purpose,
        "hp_config": hp_config,
    })
    return run_id, dirs


def finalize_run(run_dir, status: str = "complete", updates: dict | None = None) -> None:
    """Mark a run finished: set status, compute runtime_seconds from
    created_at, apply any extra manifest updates, rewrite atomically."""
    run_dir = Path(run_dir)
    manifest = load_manifest(run_dir)
    manifest["status"] = status
    elapsed = datetime.now(timezone.utc) - datetime.fromisoformat(manifest["created_at"])
    manifest["runtime_seconds"] = round(elapsed.total_seconds(), 1)
    if updates:
        manifest.update(updates)
    _write_manifest(run_dir, manifest)


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