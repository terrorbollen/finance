"""Tests for data/fetcher.py — ticker resolution, preprocessing, and fallback behaviour."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from data.fetcher import StockDataFetcher


def _ohlcv_df(n: int = 10, start: str = "2023-01-02") -> pd.DataFrame:
    """Minimal OHLCV DataFrame with a tz-naive DatetimeIndex."""
    dates = pd.date_range(start, periods=n, freq="B")
    return pd.DataFrame(
        {
            "Open": [100.0] * n,
            "High": [105.0] * n,
            "Low": [95.0] * n,
            "Close": [101.0] * n,
            "Volume": [1_000_000] * n,
        },
        index=dates,
    )


# ---------------------------------------------------------------------------
# _resolve_ticker
# ---------------------------------------------------------------------------


class TestResolveTicker:
    def test_omxs30_alias_resolved(self):
        f = StockDataFetcher()
        assert f._resolve_ticker("OMXS30") == "^OMX"

    def test_omxspi_alias_resolved(self):
        f = StockDataFetcher()
        assert f._resolve_ticker("OMXSPI") == "^OMXSPI"

    def test_stock_alias_resolved(self):
        f = StockDataFetcher()
        assert f._resolve_ticker("VOLVO-B") == "VOLV-B.ST"

    def test_unknown_ticker_returned_as_is(self):
        f = StockDataFetcher()
        assert f._resolve_ticker("XYZ.ST") == "XYZ.ST"

    def test_case_insensitive(self):
        f = StockDataFetcher()
        assert f._resolve_ticker("omxs30") == "^OMX"

    def test_caret_prefix_stripped_before_lookup(self):
        f = StockDataFetcher()
        assert f._resolve_ticker("^OMXS30") == "^OMX"


# ---------------------------------------------------------------------------
# _preprocess
# ---------------------------------------------------------------------------


class TestPreprocess:
    def test_column_names_lowercased(self):
        f = StockDataFetcher()
        df = _ohlcv_df()
        result = f._preprocess(df)
        assert list(result.columns) == ["open", "high", "low", "close", "volume"]

    def test_timezone_removed(self):
        f = StockDataFetcher()
        df = _ohlcv_df()
        df.index = df.index.tz_localize("UTC")
        result = f._preprocess(df)
        assert result.index.tz is None

    def test_extra_columns_dropped(self):
        f = StockDataFetcher()
        df = _ohlcv_df()
        df["Dividends"] = 0.0
        df["Stock Splits"] = 0.0
        result = f._preprocess(df)
        assert "dividends" not in result.columns
        assert "stock_splits" not in result.columns

    def test_no_nans_in_output(self):
        f = StockDataFetcher()
        df = _ohlcv_df()
        result = f._preprocess(df)
        assert not result.isna().any().any()


# ---------------------------------------------------------------------------
# fetch — error when yfinance returns empty DataFrame
# ---------------------------------------------------------------------------


class TestFetch:
    def test_raises_on_empty_response(self):
        f = StockDataFetcher()
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()
        with patch("yfinance.Ticker", return_value=mock_ticker), pytest.raises(
            ValueError, match="No data found"
        ):
            f.fetch("INVALID.ST")

    def test_returns_ohlcv_dataframe(self):
        f = StockDataFetcher()
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = _ohlcv_df()
        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = f.fetch("VOLV-B.ST")
        assert list(result.columns) == ["open", "high", "low", "close", "volume"]
        assert len(result) == 10

    def test_get_latest_price_returns_last_close(self):
        f = StockDataFetcher()
        mock_ticker = MagicMock()
        df = _ohlcv_df()
        df["Close"] = list(range(1, 11))  # last close = 10
        mock_ticker.history.return_value = df
        with patch("yfinance.Ticker", return_value=mock_ticker):
            price = f.get_latest_price("VOLV-B.ST")
        assert price == 10.0

    def test_get_latest_price_returns_none_on_error(self):
        f = StockDataFetcher()
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()
        with patch("yfinance.Ticker", return_value=mock_ticker):
            price = f.get_latest_price("INVALID.ST")
        assert price is None


# ---------------------------------------------------------------------------
# fetch_reference_series — fallback when data is unavailable
# ---------------------------------------------------------------------------


class TestFetchReferenceSeries:
    def test_empty_response_returns_zeros(self):
        """When yfinance returns nothing, the series must be all-zero (not NaN)."""
        f = StockDataFetcher()
        align_to = pd.date_range("2023-01-02", periods=5, freq="B")
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()
        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = f.fetch_reference_series("^OMX", align_to)
        assert "close" in result.columns
        assert (result["close"] == 0.0).all()
        assert not result.isna().any().any()

    def test_result_indexed_to_align_to(self):
        """Returned DataFrame must have exactly the same index as align_to."""
        f = StockDataFetcher()
        align_to = pd.date_range("2023-01-02", periods=5, freq="B")
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()
        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = f.fetch_reference_series("^OMX", align_to)
        assert list(result.index) == list(align_to)

    def test_valid_data_aligned_and_no_nans(self):
        """Valid data should be reindexed to align_to with forward-fill, no NaNs."""
        f = StockDataFetcher()
        # Reference series covers only 3 of the 5 trading days
        ref_dates = pd.date_range("2023-01-02", periods=3, freq="B")
        ref_df = pd.DataFrame(
            {"Open": [10.0] * 3, "High": [11.0] * 3, "Low": [9.0] * 3,
             "Close": [10.0] * 3, "Volume": [0] * 3},
            index=ref_dates,
        )
        align_to = pd.date_range("2023-01-02", periods=5, freq="B")
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = ref_df
        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = f.fetch_reference_series("^OMX", align_to)
        assert len(result) == 5
        assert not result.isna().any().any()


# ---------------------------------------------------------------------------
# fetch_cross_asset_data — all 7 series returned concurrently
# ---------------------------------------------------------------------------


class TestFetchCrossAssetData:
    def test_returns_all_expected_keys(self):
        f = StockDataFetcher()
        align_to = pd.date_range("2023-01-02", periods=5, freq="B")

        # Mock every yfinance call to return an empty frame → falls back to zeros
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()
        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = f.fetch_cross_asset_data(align_to)

        expected_keys = {"omxs30", "usdsek", "eursek", "vix", "vstoxx", "oil", "rates"}
        assert set(result.keys()) == expected_keys

    def test_all_series_have_close_column(self):
        f = StockDataFetcher()
        align_to = pd.date_range("2023-01-02", periods=5, freq="B")
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()
        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = f.fetch_cross_asset_data(align_to)
        for key, df in result.items():
            assert "close" in df.columns, f"Key {key!r} missing 'close' column"

    def test_all_series_indexed_to_align_to(self):
        f = StockDataFetcher()
        align_to = pd.date_range("2023-01-02", periods=5, freq="B")
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()
        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = f.fetch_cross_asset_data(align_to)
        for key, df in result.items():
            assert list(df.index) == list(align_to), f"Key {key!r} index mismatch"
