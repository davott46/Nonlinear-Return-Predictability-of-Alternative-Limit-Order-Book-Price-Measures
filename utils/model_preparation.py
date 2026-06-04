from pathlib import Path
import pandas as pd
import numpy as np
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


def normalize_train_test(
    X_train,
    X_test,
    y_train,
    y_test
):
    # ----------------------------------------
    # Feature normalization
    # ----------------------------------------

    X_mean = X_train.mean(axis=0,dtype=np.float64)
    X_std = X_train.std(axis=0,dtype=np.float64)

    X_std[X_std==0.0] = 1.0

    X_train_norm = ((X_train - X_mean) / X_std).astype(np.float32)

    X_test_norm = ((X_test - X_mean) / X_std).astype(np.float32)

    # ----------------------------------------
    # Target normalization
    # ----------------------------------------

    y_mean = y_train.mean(dtype=np.float64)
    y_std = y_train.std(dtype=np.float64)

    if y_std==0.0:
        y_std = 1.0

    y_train_norm = (
        (y_train - y_mean)/y_std
    ).astype(np.float32)

    y_test_norm = (
        (y_test - y_mean)/y_std
    ).astype(np.float32)

    return {
        "X_train": X_train_norm,
        "X_test": X_test_norm,

        "y_train": y_train_norm,
        "y_test": y_test_norm,

        "X_mean": X_mean.astype(np.float32),
        "X_std": X_std.astype(np.float32),

        "y_mean": np.float32(y_mean),
        "y_std": np.float32(y_std)
    }


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
        df.reset_index(drop=True).to_feather(path)

    else:
        raise ValueError(f"Unsupported file format: {file_format}")