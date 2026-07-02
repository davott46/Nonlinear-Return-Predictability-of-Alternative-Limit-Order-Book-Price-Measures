from pathlib import Path
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