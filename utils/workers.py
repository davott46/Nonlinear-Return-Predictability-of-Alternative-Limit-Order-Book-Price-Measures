"""Per-stock model workers, run via utils.execution.run_parallel:

    tune_xgb(run_dir, symbol, device)            # XGBoost random search for one tuning stock
    walk_forward(run_dir, symbol, leg, device)   # leg in {"ols", "xgb", "hybrid"}

Settings come from <run_dir>/manifest.json. Output: one file per (stock, target),
trials/<symbol>/<target>.parquet and partial/<leg>/<symbol>/<target>.parquet (with a `model_leg`
column); existing files are skipped on re-run. Test days are all days after the tuning block;
the training window (`train_days` preceding days) may reach into it.
"""

import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

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


def horizon_of(target: str) -> str:
    """'T_MidPrice_LogReturn_2s' -> '2s'."""
    return target.rsplit("_", 1)[-1]


def stack_days(day_cache: dict, days: list, j: int | None = None):
    """Concatenate X and Y over `days`; j selects a single target column."""
    X = np.concatenate([day_cache[d]["X"] for d in days])
    Y = np.concatenate([day_cache[d]["Y"] if j is None else day_cache[d]["Y"][:, j] for d in days])
    return X, Y


# --------------------------------------------------------------------------------------
# Tuning
# --------------------------------------------------------------------------------------

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
    Same seed gives the same trial sequence for every stock x target (needed by freeze_winners).
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
            ratios.append(float(np.mean(y_val ** 2)) / float(model.best_score) ** 2)  # MSE ratio
            iterations.append(int(model.best_iteration))  # 0-indexed best boosting round
        rows.append({
            "trial": trial,
            **params,
            "mean_mse_ratio": float(np.mean(ratios)),
            "n_estimators_frozen": int(np.median(iterations)),
            "fit_seconds": time.perf_counter() - start,
        })
    return pd.DataFrame(rows)


def freeze_winners(all_trials: pd.DataFrame, search_space: dict) -> dict:
    """Pooled winner per horizon (D026): the trial with the best mean validation MSE ratio
    across all tuning stocks and price measures. Returns {horizon: params};
    n_estimators = median frozen count of that trial across the pooled units (+1 for 0-indexing).
    """
    names = list(search_space)
    t = all_trials.assign(horizon=all_trials["target"].map(horizon_of))
    if (t.groupby(["horizon", "trial"])[names].nunique() > 1).any().any():
        raise ValueError("trial configs differ across stocks/targets; pooled selection needs "
                         "the same seeded draw sequence in every unit")
    pooled = t.groupby(["horizon", "trial"]).agg(
        score=("mean_mse_ratio", "mean"),
        n_est=("n_estimators_frozen", "median"),
        **{name: (name, "first") for name in names},
    )
    best = {}
    for horizon, g in pooled.groupby(level="horizon"):
        row = g.loc[g["score"].idxmax()]
        params = {name: (int(row[name]) if search_space[name][0] in ("int", "logint") else float(row[name]))
                  for name in names}
        params["n_estimators"] = int(row["n_est"]) + 1
        best[horizon] = params
    return best


def window_pairs(days: list, train_days: int, n_pairs: int) -> list:
    """n_pairs (train window, validation day) pairs spread evenly over `days`."""
    n_starts = len(days) - train_days
    if n_starts < n_pairs:
        raise ValueError(f"tuning block of {len(days)} days is too short for {n_pairs} pairs "
                         f"with a {train_days}-day window (needs >= {train_days + n_pairs})")
    starts = np.linspace(0, n_starts - 1, n_pairs).round().astype(int)
    return [(days[i:i + train_days], days[i + train_days]) for i in starts]


def tune_xgb(run_dir: str, symbol: str, device: str) -> None:
    """XGBoost random search for one stock -> trials/<symbol>/<target>.parquet, one file per target
    (existing files are skipped, so a resumed run only tunes new targets)."""
    target_dir = f"{run_dir}/trials/{symbol}"
    with open(f"{run_dir}/manifest.json") as f:
        m = json.load(f)
    t = m["tuning"]
    feature_cols, target_cols = m["feature_cols"], m["target_cols"]
    if all(os.path.exists(f"{target_dir}/{c}.parquet") for c in target_cols):
        return
    cache = load_day_cache(DATA_ROOT, symbol, m["tune_dates"], feature_cols, target_cols)
    pairs_idx = window_pairs(sorted(cache), m["train_days"], t["n_pairs"])

    base_params = {**m["xgb_params"], **t["early_stopping"], "device": device}
    os.makedirs(target_dir, exist_ok=True)
    for j, target in enumerate(target_cols):
        if os.path.exists(f"{target_dir}/{target}.parquet"):
            continue
        pairs = []
        for window, val_day in pairs_idx:
            X_tr, y_tr = stack_days(cache, window, j)
            X_val, y_val = cache[val_day]["X"], cache[val_day]["Y"][:, j]
            pairs.append((X_tr, y_tr * TARGET_SCALE, X_val, y_val * TARGET_SCALE))
        trials = random_search(pairs, t["search_space"], base_params, t["n_trials"], t["seed"])
        trials.insert(0, "target", target)
        trials.insert(0, "symbol", symbol)
        trials.to_parquet(f"{target_dir}/{target}.parquet", index=False)


def load_units(run_dir: str, kind: str, symbols, target_cols) -> pd.DataFrame:
    """Concatenate the per-(stock, target) files <run_dir>/<kind>/<symbol>/<target>.parquet
    (kind = "trials" or "partial/<leg>"); missing files are skipped."""
    paths = [f"{run_dir}/{kind}/{s}/{c}.parquet" for s in symbols for c in target_cols]
    return pd.concat([pd.read_parquet(p) for p in paths if os.path.exists(p)], ignore_index=True)


# --------------------------------------------------------------------------------------
# Walk-forward: one loop, one leg per call
# --------------------------------------------------------------------------------------

def walk_forward(run_dir: str, symbol: str, leg: str, device: str = "cpu") -> None:
    """Walk-forward for one stock and one leg -> partial/<leg>/<symbol>/<target>.parquet, one file
    per target with all test days (existing files are skipped, so a resumed run only computes
    new stocks and new targets)."""
    with open(f"{run_dir}/manifest.json") as f:
        m = json.load(f)
    window, feature_cols = m["train_days"], m["feature_cols"]
    out_dir = f"{run_dir}/partial/{leg}/{symbol}"
    todo = [c for c in m["target_cols"] if not os.path.exists(f"{out_dir}/{c}.parquet")]
    if not todo:
        return

    cache = load_day_cache(DATA_ROOT, symbol, m["dates"], feature_cols, todo)
    days = sorted(cache)
    block_end = max(m["tune_dates"])
    test_idx = [i for i, d in enumerate(days) if d > block_end and i >= window]
    os.makedirs(out_dir, exist_ok=True)

    for j, target in enumerate(todo):
        rows = []
        for i in test_idx:
            test_day, train = days[i], days[i - window:i]
            X_train, y_train = stack_days(cache, train, j)
            X_test, y_test = cache[test_day]["X"], cache[test_day]["Y"][:, j]
            y_train = y_train * TARGET_SCALE  # all fits in scaled. prediction unscaled below

            if leg == "ols":
                y_pred = LinearRegression().fit(X_train, y_train).predict(X_test)
            elif leg in ("xgb", "hybrid"):
                # hybrid: OLS first, booster on the OLS train residuals; xgb: booster on the target
                if leg == "hybrid":
                    ols = LinearRegression().fit(X_train, y_train)
                    y_pred, fit_target = ols.predict(X_test), y_train - ols.predict(X_train)
                else:
                    y_pred, fit_target = 0.0, y_train
                params = {**m["xgb_params"], **m["params"][horizon_of(target)], "device": device}
                y_pred = y_pred + XGBRegressor(**params).fit(X_train, fit_target.astype(np.float32)).predict(X_test)
            else:
                raise ValueError(f"unknown leg {leg!r}")

            resid = (y_test - y_pred / TARGET_SCALE).astype(np.float32)
            rows += [dict(r, model_leg=leg) for r in daily_diagnostic_rows(
                resid=resid[:, None], Y_test=y_test[:, None], target_cols=[target], train_day=train[-1],
                test_day=test_day, symbol=symbol, run_id=m["run_id"], n_train=X_train.shape[0], n_test=X_test.shape[0])]
        pd.DataFrame(rows).to_parquet(f"{out_dir}/{target}.parquet", index=False)
