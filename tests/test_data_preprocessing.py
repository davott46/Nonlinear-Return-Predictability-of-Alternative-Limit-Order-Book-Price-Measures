"""
Test suite for the data-processing pipeline in utils/data_preprocessing.py.

Covers the behaviour agreed on:
  * compute_feature_target_matrix
        - forward log-returns:  log(px[t+h] / px[t]), first ts >= t+h
        - backward features: non-overlapping, additive segments,
                             last ts <= t-|h|
        - price measures use log-differences, other features plain differences
        - out-of-range horizons produce NaN
  * compute_transaction_price -> buy=ask, sell=bid, forward-filled
  * filter_trading_hours      -> Xetra Berlin windows, boundaries
  * merge_asof integration    -> backward as-of join is leak-free

Run with:  python -m unittest tests.test_data_preprocessing
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils.data_preprocessing as du  # noqa: E402

SEC = 1_000_000_000  # one second in nanoseconds
LN2 = np.log(2.0)


def _regular_df(px_map, n=5):
    """DataFrame on a 1-second int-ns grid with the given price columns."""
    data = {"Timestamp": np.arange(n, dtype=np.int64) * SEC}
    data.update(px_map)
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# compute_transaction_price
# ---------------------------------------------------------------------------

class TestComputeTransactionPrice(unittest.TestCase):

    def test_buy_sell_ffill(self):
        df = pd.DataFrame({
            "Side":        [1, -1, 0, 1],
            "L1-AskPrice": [10, 11, 12, 13],
            "L1-BidPrice": [9, 8, 7, 6],
        })
        out = du.compute_transaction_price(df)
        # buy -> ask (10, 13); sell -> bid (8); side 0 -> NaN then forward-filled (8)
        self.assertEqual(out.tolist(), [10.0, 8.0, 8.0, 13.0])


# ---------------------------------------------------------------------------
# filter_trading_hours
# ---------------------------------------------------------------------------

class TestFilterTradingHours(unittest.TestCase):

    def test_boundaries(self):
        times = [
            "2023-01-02 09:16",  # before morning open   -> drop
            "2023-01-02 09:17",  # morning open          -> keep
            "2023-01-02 12:44",  # last morning minute   -> keep
            "2023-01-02 12:45",  # morning close (excl.) -> drop
            "2023-01-02 13:16",  # before afternoon open -> drop
            "2023-01-02 13:17",  # afternoon open        -> keep
            "2023-01-02 17:14",  # last afternoon minute -> keep
            "2023-01-02 17:15",  # afternoon close (excl)-> drop
        ]
        df = pd.DataFrame({
            "Timestamp_Europe/Berlin": pd.to_datetime(times).tz_localize("Europe/Berlin"),
            "row": range(len(times)),
        })
        kept = du.filter_trading_hours(df)["row"].tolist()
        self.assertEqual(kept, [1, 2, 5, 6])

    def test_converts_from_other_tz(self):
        # same instants, but stored in a different tz (as the raw data actually is);
        # the filter must convert to Berlin before applying the window
        berlin = pd.to_datetime(
            ["2023-01-02 09:17", "2023-01-02 09:16"]
        ).tz_localize("Europe/Berlin")
        df = pd.DataFrame({
            "Timestamp_Europe/Berlin": berlin.tz_convert("Asia/Tokyo"),
            "row": [0, 1],
        })
        self.assertEqual(du.filter_trading_hours(df)["row"].tolist(), [0])


# ---------------------------------------------------------------------------
# compute_feature_target_matrix : forward targets
# ---------------------------------------------------------------------------

class TestForwardTargets(unittest.TestCase):

    def test_forward_log_returns(self):
        # geometric price -> every 1s log-return equals ln2
        df = _regular_df({"MidPrice": [1.0, 2.0, 4.0, 8.0, 16.0]})
        out = du.compute_feature_target_matrix(
            df, ts_col="Timestamp",
            target_cols=["MidPrice"], feature_cols=[],
            horizons=["1s", "2s"],
        )

        r1 = out["T_MidPrice_LogReturn_1s"].to_numpy()
        np.testing.assert_allclose(r1[:4], [LN2, LN2, LN2, LN2], atol=1e-9)
        self.assertTrue(np.isnan(r1[4]))  # no t+1s available for the last row

        r2 = out["T_MidPrice_LogReturn_2s"].to_numpy()
        np.testing.assert_allclose(r2[:3], [2 * LN2, 2 * LN2, 2 * LN2], atol=1e-9)
        self.assertTrue(np.isnan(r2[3]) and np.isnan(r2[4]))

    def test_forward_uses_first_ts_at_or_after_horizon(self):
        # irregular grid: gap between t=1s and t=5s. For row0 (t=0), horizon +2s
        # should snap forward to the first ts >= 2s, which is t=5s (index 2).
        df = pd.DataFrame({
            "Timestamp": np.array([0, 1, 5], dtype=np.int64) * SEC,
            "MidPrice": [1.0, 3.0, 9.0],
        })
        out = du.compute_feature_target_matrix(
            df, ts_col="Timestamp",
            target_cols=["MidPrice"], feature_cols=[],
            horizons=["2s"],
        )
        r = out["T_MidPrice_LogReturn_2s"].to_numpy()
        # row0: log(px[idx of first ts>=2s] / px[0]) = log(9/1)
        np.testing.assert_allclose(r[0], np.log(9.0 / 1.0), atol=1e-9)


# ---------------------------------------------------------------------------
# compute_feature_target_matrix : backward features
# ---------------------------------------------------------------------------

class TestBackwardFeatures(unittest.TestCase):

    def test_price_feature_is_logdiff(self):
        df = _regular_df({"MicroPrice": [10.0, 20.0, 40.0, 80.0, 160.0]})
        out = du.compute_feature_target_matrix(
            df, ts_col="Timestamp",
            target_cols=[], feature_cols=["MicroPrice"],
            horizons=["-1s"],
        )
        f = out["F_MicroPrice_-1s"].to_numpy()
        self.assertTrue(np.isnan(f[0]))  # no ts at t-1s for the first row
        np.testing.assert_allclose(f[1:], [LN2, LN2, LN2, LN2], atol=1e-9)

    def test_nonprice_feature_is_plain_diff(self):
        # L1-QImb is NOT in PRICE_MEASURES -> plain difference, not log
        df = _regular_df({"L1-QImb": [0.0, 1.0, 3.0, 6.0, 10.0]})
        out = du.compute_feature_target_matrix(
            df, ts_col="Timestamp",
            target_cols=[], feature_cols=["L1-QImb"],
            horizons=["-1s"],
        )
        f = out["F_L1-QImb_-1s"].to_numpy()
        self.assertTrue(np.isnan(f[0]))
        np.testing.assert_allclose(f[1:], [1.0, 2.0, 3.0, 4.0], atol=1e-9)

    def test_features_are_non_overlapping_and_additive(self):
        # Two backward horizons should decompose the total change with no overlap:
        #   F_-2s (segment [t-2s, t-1s]) + F_-1s (segment [t-1s, t]) == log(px[t]/px[t-2s])
        px = [10.0, 20.0, 40.0, 80.0, 160.0]
        df = _regular_df({"MicroPrice": px})
        out = du.compute_feature_target_matrix(
            df, ts_col="Timestamp",
            target_cols=[], feature_cols=["MicroPrice"],
            horizons=["-1s", "-2s"],
        )
        f1 = out["F_MicroPrice_-1s"].to_numpy()
        f2 = out["F_MicroPrice_-2s"].to_numpy()

        # rows 2..4 have both segments defined
        total = f2[2:] + f1[2:]
        expected = np.log(np.array(px[2:]) / np.array(px[:-2]))  # log(px[t]/px[t-2s])
        np.testing.assert_allclose(total, expected, atol=1e-9)

    def test_uses_last_ts_at_or_before_horizon(self):
        # irregular grid: t = [0, 1, 5]s. For row2 (t=5s), horizon -2s targets 3s;
        # the last ts <= 3s is t=1s (index 1).
        df = pd.DataFrame({
            "Timestamp": np.array([0, 1, 5], dtype=np.int64) * SEC,
            "MicroPrice": [10.0, 20.0, 40.0],
        })
        out = du.compute_feature_target_matrix(
            df, ts_col="Timestamp",
            target_cols=[], feature_cols=["MicroPrice"],
            horizons=["-2s"],
        )
        f = out["F_MicroPrice_-2s"].to_numpy()
        # row2: segment start = last ts <= 3s = index1 (px 20), end = now (px 40)
        np.testing.assert_allclose(f[2], np.log(40.0 / 20.0), atol=1e-9)


# ---------------------------------------------------------------------------
# compute_feature_target_matrix : structure / column naming
# ---------------------------------------------------------------------------

class TestOutputStructure(unittest.TestCase):

    def test_column_names_and_original_preserved(self):
        df = _regular_df({"MidPrice": [1.0, 2.0, 4.0, 8.0, 16.0],
                          "MicroPrice": [1.0, 2.0, 4.0, 8.0, 16.0]})
        out = du.compute_feature_target_matrix(
            df, ts_col="Timestamp",
            target_cols=["MidPrice"], feature_cols=["MicroPrice"],
            horizons=["1s", "-1s"],
        )
        self.assertIn("T_MidPrice_LogReturn_1s", out.columns)
        self.assertIn("F_MicroPrice_-1s", out.columns)
        # original columns are retained
        self.assertIn("MidPrice", out.columns)
        self.assertIn("Timestamp", out.columns)
        self.assertEqual(len(out), len(df))


# ---------------------------------------------------------------------------
# Integration: backward merge_asof is leak-free
# ---------------------------------------------------------------------------

class TestMergeAsofIntegration(unittest.TestCase):

    def test_backward_is_leak_free(self):
        stock = pd.DataFrame({"Timestamp": np.array([1, 2, 3], dtype=np.int64) * SEC,
                              "s": [1, 2, 3]})
        # futures ticks strictly interleaved; one lands exactly on a stock timestamp
        fut = pd.DataFrame({"Timestamp": np.array([0.5e9, 1.5e9, 2.0e9, 2.5e9]).astype(np.int64),
                            "F_fut": [100, 200, 999, 300]})

        merged = pd.merge_asof(stock, fut, on="Timestamp", direction="backward")

        # each stock row gets the last futures value with fut.ts <= stock.ts
        #   t=1s -> 0.5s (100);  t=2s -> exactly 2.0s (999, equal is allowed);
        #   t=3s -> 2.5s (300).  No future futures value is ever used.
        self.assertEqual(merged["F_fut"].tolist(), [100, 999, 300])


if __name__ == "__main__":
    unittest.main()
