"""Per-stock model workers, run via utils.execution.run_parallel:

    tune_xgb(run_dir, symbol, device)   # XGBoost random search for one stock
    run_xgb(run_dir, symbol, device)    # XGBoost walk-forward for one stock
    run_ols(run_dir, symbol)            # OLS walk-forward for one stock

All read their settings from <run_dir>/manifest.json.
"""

import json
import os
import shutil
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

import utils.data_processing as du
from utils.pipeline import TARGET_SCALE, daily_diagnostic_rows, load_day_cache

warnings.filterwarnings("ignore", category=UserWarning)

PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = f"{PARENT}/data/processed"


def feature_target_cols(symbol: str, horizons, features) -> tuple[list, list]:
    """Feature (F_<feature>_<lag>) and target (T_..._<horizon>) columns of the symbol's first day file."""
    first = sorted(f for f in os.listdir(f"{DATA_ROOT}/{symbol}") if f.endswith(".parquet"))[0]
    cols = pd.read_parquet(f"{DATA_ROOT}/{symbol}/{first}").columns
    feature_cols = [c for c in cols if c.startswith("F_") and c.split("_")[1] in features]
    target_cols = [c for c in cols if c.startswith("T_") and c.rsplit("_", 1)[-1] in horizons]
    return feature_cols, target_cols


def stack_days(day_cache: dict, days: list, j: int | None = None):
    """Concatenate X and Y over `days`; j selects a single target column."""
    X = np.concatenate([day_cache[d]["X"] for d in days])
    Y = np.concatenate([day_cache[d]["Y"] if j is None else day_cache[d]["Y"][:, j] for d in days])
    return X, Y


def sample_params(rng: np.random.Generator, search_space: dict) -> dict:
    """Draw one trial from search_space ({name: (kind, lo, hi)})."""
    params = {}
    for name, (kind, lo, hi) in search_space.items():
        if kind == "int":
            params[name] = int(rng.integers(lo, hi + 1))
        elif kind == "uniform":
            params[name] = float(rng.uniform(lo, hi))
        elif kind == "log":
            params[name] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        elif kind == "logint":
            params[name] = int(round(np.exp(rng.uniform(np.log(lo), np.log(hi)))))
        else:
            raise ValueError(f"Unknown distribution kind: {kind}")
    return params


def random_search(pairs: list, search_space: dict, base_params: dict, n_trials: int,
                  seed: int) -> pd.DataFrame:
    """Random search for one target over (X_tr, y_tr, X_val, y_val) pairs.

    Score = mean MSE ratio over pairs; n_estimators_frozen = median best_iteration.
    """
    rng = np.random.default_rng(seed)
    rows = []

    for trial in range(n_trials):
        params = sample_params(rng, search_space)
        ratios, iterations = [], []

        start = time.perf_counter()
        for X_tr, y_tr, X_val, y_val in pairs:
            model = XGBRegressor(**{**base_params, **params})
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            ratios.append(float(np.mean(y_val ** 2)) / float(model.best_score) ** 2) # MSE-ratio
            iterations.append(int(model.best_iteration)) # 0-indexed (!) best boosting round

        rows.append({
            "trial": trial,
            **params,
            "mean_mse_ratio": float(np.mean(ratios)),
            "n_estimators_frozen": int(np.median(iterations)), # median number of rounds WITHIN trial
            "fit_seconds": time.perf_counter() - start,
        })

    return pd.DataFrame(rows)


def freeze_winners(all_trials: pd.DataFrame, search_space: dict) -> dict:
    """Best trial per symbol x target as {symbol: {target: params}}, n_estimators = its frozen count."""
    best_rows = all_trials.loc[all_trials.groupby(["symbol", "target"])["mean_mse_ratio"].idxmax()]
    best = {}
    for _, row in best_rows.iterrows():
        params = {name: (int(row[name]) if search_space[name][0] in ("int", "logint") # search_space[name][0] returns type of variable
                                        else float(row[name]))                # cast to python int/float from numpy types
                                            for name in search_space}
        
        params["n_estimators"] = int(row["n_estimators_frozen"] + 1) # account for zero-indexing
        if row["symbol"] not in best:
            best[row["symbol"]] = {}
        best[row["symbol"]][row["target"]] = params
    return best


def window_pairs(days: list, train_days: int, n_pairs: int) -> list:
    """n_pairs (train window, validation day) pairs spread evenly over `days`."""
    n_starts = len(days) - train_days  # number of possible window start positions
    if n_starts < n_pairs:
        raise ValueError(f"Too few days to build n_pairs for given train_days")

    starts = np.linspace(0, n_starts - 1, n_pairs).round().astype(int) # build <n_pairs> evelny spaced; starts from 0 to n_starts-1

    pairs = []
    for i in starts:
        window = days[i:i + train_days]
        val_day = days[i + train_days]
        pairs.append((window, val_day))
    return pairs

def tune_xgb(run_dir: str, symbol: str, device: str) -> None:
    """XGBoost random search for one stock; per-target checkpoints, merged and removed at the end."""
    out_dir = f"{run_dir}/trials"
    target_dir = f"{out_dir}/{symbol}"
    with open(f"{run_dir}/manifest.json") as f:
        m = json.load(f)
    t = m["tuning"]
    feature_cols, target_cols = m["feature_cols"], m["target_cols"]
    cache = load_day_cache(DATA_ROOT, symbol, m["tune_dates"], feature_cols, target_cols)
    pairs_idx = window_pairs(sorted(cache), m["train_days"], t["n_pairs"])  # sorted(cache) = list of loaded days from dict_keys

    base_params = {**m["xgb_params"], **t["early_stopping"], "device": device}
    os.makedirs(target_dir, exist_ok=True)
    for j, target in enumerate(target_cols):
        if os.path.exists(f"{target_dir}/{target}.parquet"):
            continue
        pairs = []
        for window, val_day in pairs_idx:
            X_tr, y_tr = stack_days(cache, window, j)
            X_val = cache[val_day]["X"]
            y_val = cache[val_day]["Y"][:, j]
            pairs.append((X_tr, y_tr * TARGET_SCALE, X_val, y_val * TARGET_SCALE))
        trials = random_search(pairs, t["search_space"], base_params, t["n_trials"], t["seed"])
        trials.insert(0, "target", target)
        trials.insert(0, "symbol", symbol)
        trials.to_parquet(f"{target_dir}/{target}.parquet", index=False)

    all_trials = pd.concat([pd.read_parquet(f"{target_dir}/{c}.parquet") for c in target_cols], ignore_index=True)
    all_trials.to_parquet(f"{out_dir}/{symbol}.parquet", index=False)
    shutil.rmtree(target_dir)


def run_xgb(run_dir: str, symbol: str, device: str) -> None:
    """XGBoost walk-forward for one stock; per-day checkpoints in partial/<symbol>/, merged and removed at the end."""
    partial_dir = f"{run_dir}/partial"
    day_dir = f"{partial_dir}/{symbol}"
    with open(f"{run_dir}/manifest.json") as f:
        m = json.load(f)
    window = m["train_days"]  # training window length in days
    xgb_params = {**m["xgb_params"], "device": device}
    symbol_params = m["params"][symbol]
    feature_cols, target_cols = m["feature_cols"], m["target_cols"]

    # all days after the tuning block that exist for this stock, in order
    load = [d for d in du.SAMPLE_DATES if d > max(m["tune_dates"])]
    day_cache = load_day_cache(DATA_ROOT, symbol, load, feature_cols, target_cols)
    days = sorted(day_cache) # sorted(cache) = list of loaded days from dict_keys
    os.makedirs(day_dir, exist_ok=True)

    for i in range(window, len(days)):
        test_day = days[i]
        if os.path.exists(f"{day_dir}/{test_day}.parquet"):
            continue
        X_train, Y_train = stack_days(day_cache, days[i - window:i])  # days[i - window:i] = training days to stack
        X_test, Y_test = day_cache[test_day]["X"], day_cache[test_day]["Y"]

        # One booster per target, frozen per-stock params, targets in scaled units
        Y_pred = np.empty((X_test.shape[0], len(target_cols)), dtype=np.float32)
        for j, target in enumerate(target_cols):
            model = XGBRegressor(**{**xgb_params, **symbol_params[target]})
            model.fit(X_train, Y_train[:, j] * TARGET_SCALE)
            Y_pred[:, j] = model.predict(X_test) / TARGET_SCALE

        resid = (Y_test - Y_pred).astype(np.float32)

        rows = daily_diagnostic_rows(
            resid=resid,
            Y_test=Y_test,
            target_cols=target_cols,
            train_day=days[i - 1],
            test_day=test_day,
            symbol=symbol,
            run_id=m["run_id"],
            n_train=X_train.shape[0],
            n_test=X_test.shape[0],
        )
        pd.DataFrame(rows).to_parquet(f"{day_dir}/{test_day}.parquet", index=False)

    daily = pd.concat([pd.read_parquet(f"{day_dir}/{d}.parquet") for d in days[window:]], ignore_index=True)
    daily.to_parquet(f"{partial_dir}/{symbol}.parquet", index=False)
    shutil.rmtree(day_dir)



def run_ols(run_dir: str, symbol: str) -> None:
    """OLS walk-forward for one stock over manifest["dates"] -> partial/<symbol>.parquet."""
    partial_dir = f"{run_dir}/partial"
    with open(f"{run_dir}/manifest.json") as f:
        m = json.load(f)
    window = m["train_days"]
    feature_cols, target_cols = m["feature_cols"], m["target_cols"]

    day_cache = load_day_cache(DATA_ROOT, symbol, m["dates"], feature_cols, target_cols)
    days = sorted(day_cache)
    rows = []
    for i in range(window, len(days)):
        test_day = days[i]
        X_train, Y_train = stack_days(day_cache, days[i - window:i])
        X_test, Y_test = day_cache[test_day]["X"], day_cache[test_day]["Y"]

        model = LinearRegression().fit(X_train, Y_train * TARGET_SCALE)
        Y_pred = (model.predict(X_test) / TARGET_SCALE).astype(np.float32)
        resid = (Y_test - Y_pred).astype(np.float32)

        rows += daily_diagnostic_rows(
            resid=resid,
            Y_test=Y_test,
            target_cols=target_cols,
            train_day=days[i - 1],
            test_day=test_day,
            symbol=symbol,
            run_id=m["run_id"],
            n_train=X_train.shape[0],
            n_test=X_test.shape[0],
        )

    os.makedirs(partial_dir, exist_ok=True)
    pd.DataFrame(rows).to_parquet(f"{partial_dir}/{symbol}.parquet", index=False)
