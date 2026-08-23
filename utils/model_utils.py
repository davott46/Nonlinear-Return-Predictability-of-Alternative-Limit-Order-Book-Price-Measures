from datetime import datetime, timezone
from pathlib import Path
import json
import numpy as np
import pandas as pd
import os
import subprocess
import time


class Progress:
    """File-based progress display for server runs: `cat <path>` from a new
    ssh session shows all active loops. Replaces tqdm where the terminal
    that started the run may be gone.

    One line per named loop; nested loops just use different names. The
    header timestamp shows when the file was last touched, so a stalled or
    dead run is distinguishable from a slow one. ETA is based on progress
    made *within this process* (the first `current` seen per name is the
    baseline), so runs resumed from checkpoints don't report absurdly fast
    ETAs. `done` keeps the bar in the file, marked "done" with its final
    elapsed time. Writes are atomic (tmp + rename): a concurrent `cat`
    never sees a half-written file. Not safe for concurrent writers — one
    instance per run.

    `update` calls within `write_interval` seconds of the last write are
    counted but not flushed to disk (state changes like a new bar or `done`
    always flush), so tight loops can call it per iteration without I/O cost.
    """

    def __init__(self, path, write_interval: float = 60.0):
        self.path = str(path)
        self.write_interval = write_interval
        self._last_write = -float("inf")
        # name -> [current, total, start_time, baseline_current, done_time]
        self.bars = {}

    def update(self, name, current, total):
        if name not in self.bars:
            self.bars[name] = [current, total, time.time(), current, None]
            self._write()
        else:
            self.bars[name][0] = current
            self.bars[name][1] = total
            if time.time() - self._last_write >= self.write_interval:
                self._write()

    def done(self, name):
        if name in self.bars:
            self.bars[name][4] = time.time()
            self._write()

    @staticmethod
    def _fmt(seconds):
        seconds = int(seconds)
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _write(self):
        now = time.time()
        stamp = datetime.now().astimezone().isoformat(timespec="seconds")
        lines = [f"last update: {stamp}"]
        for name, (cur, tot, start, base, done_at) in self.bars.items():
            pct = 100.0 * cur / tot if tot else 0.0
            if done_at is not None:
                lines.append(f"{name:<12} {pct:5.1f}%  ({cur}/{tot})  "
                             f"elapsed {self._fmt(done_at - start)}  done")
                continue
            elapsed = now - start
            done_here = cur - base
            eta_str = (self._fmt(elapsed * (tot - cur) / done_here)
                       if done_here > 0 else "--:--:--")
            lines.append(f"{name:<12} {pct:5.1f}%  ({cur}/{tot})  "
                         f"elapsed {self._fmt(elapsed)}  eta {eta_str}")
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            f.write("\n".join(lines) + "\n")
        os.replace(tmp, self.path)
        self._last_write = now


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


# XGBoost silently learns nothing on raw log-return targets: split gains scale
# with the target variance (~1e-9 for 100ms returns) and fall below hardcoded
# absolute epsilons in the tree builder, so no tree ever splits. Fitting on
# y * TARGET_SCALE and dividing predictions back is exact for squared-error
# models (tree structure, OLS and ridge fits are all scale-equivariant). The
# constant is fixed rather than per-day so target-unit parameters (reg_alpha,
# gamma) and regularization strength stay comparable across days.
TARGET_SCALE = 1e4  # log returns -> basis points


def scale_target(y):
    """Scale a target (or residual target) into fitting units."""
    return y * TARGET_SCALE


def unscale_prediction(pred):
    """Map model predictions back to raw log-return units."""
    return pred / TARGET_SCALE


def load_day_cache(
        data_root,
        symbol,
        days,
        feature_cols,
        target_cols
) -> dict[str, dict[str, np.ndarray]]:
    """Load all of one symbol's days into {day: {"X", "Y", "timestamp"}}.

    Days without a parquet file are skipped with a warning (stocks can have
    missing trading days), so the returned dict may hold fewer days than
    requested. Iterate over its keys, not the requested list.
    """
    day_cache = {}

    for day in days:
        path = f"{data_root}/{symbol}/{day}.parquet"
        if not os.path.exists(path):
            print(f"WARNING: {symbol} {day}: no processed file, skipping")
            continue
        df = pd.read_parquet(path).sort_values("Timestamp")
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


def standardize_features(
        X_train,
        X_test,
        clip_sd: float = 5.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Z-score both sets by TRAIN-day mean/std, then winsorize at +-clip_sd.

    Winsorizing is essential for downstream polynomial feature expansions:
    squared terms of test-day observations far outside the train range
    otherwise explode out of sample. Zero-variance columns map to zeros.

    Returns (Z_train, Z_test, mean, sd).
    """
    mean = X_train.mean(axis=0)
    sd = X_train.std(axis=0)
    safe_sd = np.where(sd > 0, sd, 1.0)

    Z_train = np.clip((X_train - mean) / safe_sd, -clip_sd, clip_sd)
    Z_test = np.clip((X_test - mean) / safe_sd, -clip_sd, clip_sd)

    zero_var = sd == 0
    if zero_var.any():
        Z_train[:, zero_var] = 0.0
        Z_test[:, zero_var] = 0.0

    return (Z_train.astype(np.float32), Z_test.astype(np.float32), mean, sd)


def ladder_blocks(
        Z_train,
        Z_test,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Cumulative design matrices for the 4-rung linearity ladder.

    Inputs must already be standardized + winsorized (see
    standardize_features). Each rung adds one kind of non-linearity:

        "linear" : Z                              (p columns)
        "asym"   : + max(Z, 0)                    (2p)     sign asymmetry
        "curv"   : + Z*|Z| and Z**2               (4p)     marginal curvature
        "inter"  : + all pairwise products        (4p + C(p, 2))

    Returns {rung: (A_train, A_test)} as float32 column stacks.
    """
    from itertools import combinations

    def stack(blocks):
        return tuple(
            np.column_stack(mats).astype(np.float32)
            for mats in zip(*blocks)
        )

    blocks = [(Z_train, Z_test)]
    out = {"linear": stack(blocks)}

    blocks.append((np.maximum(Z_train, 0), np.maximum(Z_test, 0)))
    out["asym"] = stack(blocks)

    blocks.append((Z_train * np.abs(Z_train), Z_test * np.abs(Z_test)))
    blocks.append((Z_train ** 2, Z_test ** 2))
    out["curv"] = stack(blocks)

    pairs = list(combinations(range(Z_train.shape[1]), 2))
    blocks.append((
        np.column_stack([Z_train[:, a] * Z_train[:, b] for a, b in pairs]),
        np.column_stack([Z_test[:, a] * Z_test[:, b] for a, b in pairs]),
    ))
    out["inter"] = stack(blocks)

    return out


def save_tick_residuals(
        resid,
        timestamps,
        target_cols,
        tick_output_dir,
        test_day, symbol,
        dtype=np.float16
) -> None:
    """Minimal tick-level store for Diebold-Mariano: Timestamp + one residual
    column per target. y_true is NOT stored.

    float16 residuals: ~2x smaller than float32, aggregate MSE error ~1e-6
    (negligible for DM). feather+zstd is the smallest lossless container here
    (parquet inflates incompressible floats). Pass dtype=np.float32 when
    residuals sit near float16's subnormal floor (~6e-5, e.g. 100ms-horizon
    returns) and DM differentials between near-tied models matter.
    """
    tick_df = pd.DataFrame(resid.astype(dtype), columns=target_cols)
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