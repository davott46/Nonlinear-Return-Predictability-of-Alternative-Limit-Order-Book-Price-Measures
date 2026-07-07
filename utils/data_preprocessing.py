import pandas as pd
import numpy as np
from datetime import datetime, timedelta


SYMBOLS = [
 'Adidas',
 'Deutsche Post',
 'Henkel',
 'Fresenius',
 'Siemens',
 'Beiersdorf',
 'RWE',
 'Muenchener Rueckversicherungs-Gesellschaft',
 'Continental',
 'Mercedes Benz',
 'Zalando',
 'Volkswagen',
 'Brenntag',
 'Qiagen',
 'Infineon',
 'Heidelberg Cement',
 'Deutsche Boerse',
 'Sartorius',
 'Fresenius Medical Care',
 'Airbus',
 'FUT_DAX Futures']




SAMPLE_DATES = [
 '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05', '2023-01-06', '2023-01-09', '2023-01-10', '2023-01-11',
 '2023-01-12', '2023-01-13', '2023-01-16', '2023-01-17', '2023-01-18', '2023-01-19', '2023-01-20', '2023-01-23',
 '2023-01-24', '2023-01-25', '2023-01-26', '2023-01-27', '2023-01-30', '2023-01-31', '2023-02-01', '2023-02-02',
 '2023-02-03', '2023-02-06', '2023-02-07', '2023-02-08', '2023-02-09', '2023-02-10', '2023-02-13', '2023-02-14',
 '2023-02-15', '2023-02-16', '2023-02-17', '2023-02-20', '2023-02-21', '2023-02-22', '2023-02-23', '2023-02-24',
 '2023-02-27', '2023-02-28', '2023-03-01', '2023-03-02', '2023-03-03', '2023-03-06', '2023-03-07', '2023-03-08',
 '2023-03-09', '2023-03-10', '2023-03-13', '2023-03-14', '2023-03-15', '2023-03-16', '2023-03-17', '2023-03-20',
 '2023-03-21', '2023-03-22', '2023-03-23', '2023-03-24', '2023-03-27', '2023-03-28', '2023-03-29', '2023-03-30',
 '2023-03-31', '2023-04-03', '2023-04-04', '2023-04-05', '2023-04-06', '2023-04-11', '2023-04-12', '2023-04-13',
 '2023-04-14', '2023-04-17', '2023-04-18', '2023-04-19', '2023-04-20', '2023-04-21', '2023-04-24', '2023-04-25',
 '2023-04-26', '2023-04-27', '2023-04-28', '2023-05-02', '2023-05-03', '2023-05-04', '2023-05-05', '2023-05-08',
 '2023-05-09', '2023-05-10', '2023-05-11', '2023-05-12', '2023-05-15', '2023-05-16', '2023-05-17', '2023-05-18',
 '2023-05-19', '2023-05-22', '2023-05-23', '2023-05-24', '2023-05-25', '2023-05-26', '2023-05-29', '2023-05-30',
 '2023-05-31', '2023-06-01', '2023-06-02', '2023-06-05', '2023-06-06', '2023-06-07', '2023-06-08', '2023-06-09',
 '2023-06-12', '2023-06-13', '2023-06-14', '2023-06-15', '2023-06-16', '2023-06-19', '2023-06-20', '2023-06-21',
 '2023-06-22', '2023-06-23', '2023-06-26', '2023-06-27', '2023-06-28', '2023-06-29','2023-06-30']


PRICE_MEASURES = ['TransactionPrice','MidPrice', 'MidPriceQW', 'MidPriceCQW', 'MicroPrice']


# Make this faster using int timestamp instead of datetime ts later.
def filter_trading_hours(
    df: pd.DataFrame,
    ts_col: str = 'Timestamp_Europe/Berlin'
) -> pd.DataFrame:
    """
    Filters a DataFrame to only keep rows whose datetime falls within
    the Xetra Europe/Berlin trading windows:

    - 09:17 to 12:45
    - 13:17 to 17:15
    """
    dt = pd.to_datetime(df[ts_col])

    # Ensure Europe/Berlin timezone
    if dt.dt.tz is None:
        dt = dt.dt.tz_localize("Europe/Berlin")
    else:
        dt = dt.dt.tz_convert("Europe/Berlin")

    # Minutes since midnight
    minutes = dt.dt.hour * 60 + dt.dt.minute

    morning = (557 <= minutes) & (minutes < 765)
    afternoon = (797 <= minutes) & (minutes < 1035)

    return df[morning | afternoon]


def resample_last(
    df: pd.DataFrame,
    ts_col: str,
    freq: str = "100ms",
) -> pd.DataFrame:
    """
    Keep one row per `freq` bucket: the last observation in each bucket. Empty
    buckets are omitted.

    `ts_col` must be integer nanoseconds since the epoch. Timestamps are NOT
    floored to the grid; each retained row keeps the actual timestamp of its
    last snapshot. This makes the column an exact, sorted, leak-free time axis
    for both `compute_feature_target_matrix` (which locates horizons via
    `searchsorted` and needs no regular grid) and a backward `merge_asof`
    (flooring would stamp a snapshot earlier than observed and leak future
    information into the as-of join).

    Parameters
    ----------
    df     : pd.DataFrame
    ts_col : Integer-nanosecond timestamp column to bucket on.
    freq   : Bucket width as a pandas offset string, e.g. "100ms", "1s".
    """
    if len(df) == 0:
        return df.reset_index(drop=True)

    freq_ns = pd.Timedelta(freq).value
    ts = df[ts_col].to_numpy()

    # Stable sort so "last per bucket" is well defined even for ties 
    order = np.argsort(ts, kind="stable")
    bucket = ts[order] // freq_ns

    # Rows are contiguous per bucket after sorting; keep each bucket's last row,
    # i.e. the position where the next row falls into a different bucket.
    keep = np.empty(len(bucket), dtype=bool)
    keep[-1] = True
    keep[:-1] = bucket[1:] != bucket[:-1]

    return df.iloc[order[keep]].reset_index(drop=True)


def compute_transaction_price(
    df: pd.DataFrame
) -> pd.Series:
    side_arr = df['Side'].to_numpy()
    ask_arr  = df['L1-AskPrice'].to_numpy()
    bid_arr  = df['L1-BidPrice'].to_numpy()

    out = np.empty(len(side_arr), dtype=np.float64)
    out[:] = np.nan

    buy_mask  = side_arr == 1
    sell_mask = side_arr == -1

    out[buy_mask]  = ask_arr[buy_mask]
    out[sell_mask] = bid_arr[sell_mask]

    return pd.Series(out, index=df.index).ffill()    


def compute_feature_target_matrix(
    df: pd.DataFrame,
    ts_col: str,
    target_cols: list[str],
    feature_cols: list[str],
    horizons: list[str],
    dtype: np.dtype = np.float64, # For testing, this needs to be float64 at 1e-6
) -> pd.DataFrame:
    """
    1) Forward returns for all price measures
    2) Non-overlapping backward returns for microprice
    3) Non-overlapping change in order book imbalance at first level

    Parameters
    ----------
    df            : pd.DataFrame
    ts_col        : Timestamp column.
    target_cols   : Price columns used for return computation.
    feature_cols  : Columns for lagged feature extraction (MicroPrice and L1-Qimb).
    horizons      : Neg value represent past horizons, i.e. ["1s", "5s", "-1s", "-5s"]
    dtype         : Numerical dtype for output arrays.
    """

    # accumulate new columns here
    out_cols = {}
    # Horizon classification
    deltas = np.array([pd.Timedelta(h).value for h in horizons], dtype=np.int64)

    ts = df[ts_col].to_numpy(dtype=np.int64)

    forward_idx = np.flatnonzero(deltas >= 0)
    backward_idx = np.flatnonzero(deltas < 0)
    backward_idx = backward_idx[np.argsort(deltas[backward_idx])]
    # ------------------------------------------------------------------
    # Build index matrix
    # ------------------------------------------------------------------

    N = len(df)        # sample size
    H = len(horizons)  # number of horizons
    idx_matrix = np.empty((N, H), dtype=np.int32)

    # for targets, find first timestamp right after ts + h
    for j in forward_idx:
        target_ts = ts + deltas[j]

        idx_matrix[:, j] = np.searchsorted(
            ts,
            target_ts,
            side="left")

    # for features, find last timestamp right before ts - h
    for j in backward_idx:
        target_ts = ts + deltas[j]

        idx_matrix[:, j] = (
            np.searchsorted(
            ts,
            target_ts,
            side="right") - 1)

    # ------------------------------------------------------------------
    # Compute targets (returns at t+h)
    # ------------------------------------------------------------------

    for target in target_cols:
        px = df[target].to_numpy(dtype=dtype)

        for j in forward_idx:
            idx = idx_matrix[:, j] # select column j of targets
            
            valid_j = ((idx >= 0) & (idx < N))

            vals = np.full(N, np.nan, dtype=dtype)
            vals[valid_j] = np.log(px[idx[valid_j]] / px[valid_j])

            out_cols[f"T_{target}_LogReturn_{horizons[j]}"] = vals
    
    for feature in feature_cols:
        ft = df[feature].to_numpy(dtype=dtype)

        # Non-overlapping features
        for k, j in enumerate(backward_idx): # k is index into backward_idx, j is col-index into idx_matrix
    
            start_idx = idx_matrix[:, j]

            # End index: next backward horizon OR current timestamp
            if k < len(backward_idx) - 1:
                end_idx = idx_matrix[:, backward_idx[k + 1]]
            else:
                end_idx = np.arange(N)
    
            valid_j = (
                (start_idx >= 0) & (start_idx < N)
                & (end_idx >= 0) & (end_idx < N)) # select all valid rows for column j of price data

            vals = np.full(N, np.nan, dtype=dtype)
            
            if feature in PRICE_MEASURES:
                vals[valid_j] = np.log(ft[end_idx[valid_j]]/ ft[start_idx[valid_j]])
            else:
                vals[valid_j] = (ft[end_idx[valid_j]] - ft[start_idx[valid_j]])

            out_cols[f"F_{feature}_{horizons[j]}"] = vals

    return pd.concat([df, pd.DataFrame(out_cols, index=df.index)], axis=1)

