"""Data fetching module for Swedish stocks and indexes via yfinance."""

from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import yfinance as yf

# Common Swedish stock tickers
SWEDISH_INDEXES = {
    "OMXS30": "^OMX",
    "OMXSPI": "^OMXSPI",
}

SWEDISH_STOCKS = {
    "VOLVO-B": "VOLV-B.ST",
    "ERICSSON-B": "ERIC-B.ST",
    "H&M-B": "HM-B.ST",
    "ATLAS-COPCO-A": "ATCO-A.ST",
    "INVESTOR-B": "INVE-B.ST",
    "NORDEA": "NDA-SE.ST",
    "SEB-A": "SEB-A.ST",
    "SWEDBANK-A": "SWED-A.ST",
    "SANDVIK": "SAND.ST",
    "ABB": "ABB.ST",
}


class StockDataFetcher:
    """Fetches and preprocesses stock data from Yahoo Finance."""

    def __init__(self, period: str = "2y", interval: str = "1d"):
        """
        Initialize the data fetcher.

        Args:
            period: Data period (e.g., '1y', '2y', '5y', 'max')
            interval: Data interval (e.g., '1d', '1wk', '1mo')
        """
        self.period = period
        self.interval = interval

    def fetch(self, ticker: str) -> pd.DataFrame:
        """
        Fetch historical data for a given ticker.

        Args:
            ticker: Stock ticker symbol (e.g., '^OMX', 'VOLV-B.ST')

        Returns:
            DataFrame with OHLCV data
        """
        # Resolve common aliases
        resolved_ticker = self._resolve_ticker(ticker)

        stock = yf.Ticker(resolved_ticker)
        df = stock.history(period=self.period, interval=self.interval, timeout=30)

        if df.empty:
            raise ValueError(f"No data found for ticker: {ticker}")

        # Clean and preprocess
        df = self._preprocess(df)

        return df

    def fetch_multiple(self, tickers: list[str]) -> dict[str, pd.DataFrame]:
        """
        Fetch data for multiple tickers.

        Args:
            tickers: List of ticker symbols

        Returns:
            Dictionary mapping ticker to DataFrame
        """
        results = {}
        for ticker in tickers:
            try:
                results[ticker] = self.fetch(ticker)
            except ValueError as e:
                print(f"Warning: {e}")
        return results

    def _resolve_ticker(self, ticker: str) -> str:
        """Resolve common ticker aliases to Yahoo Finance format."""
        ticker_upper = ticker.upper()

        # Strip leading ^ for alias lookup
        ticker_clean = ticker_upper.lstrip("^")

        # Check indexes
        if ticker_clean in SWEDISH_INDEXES:
            return SWEDISH_INDEXES[ticker_clean]

        # Check stocks
        if ticker_clean in SWEDISH_STOCKS:
            return SWEDISH_STOCKS[ticker_clean]

        # Return as-is (assume it's already in correct format)
        return ticker

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the data."""
        # Remove rows with missing values
        df = df.dropna()

        # Ensure index is datetime
        df.index = pd.to_datetime(df.index)

        # Remove timezone info for easier handling
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # Standardize column names
        df.columns = [col.lower().replace(" ", "_") for col in df.columns]

        # Keep only relevant columns
        columns_to_keep = ["open", "high", "low", "close", "volume"]
        df = df[[col for col in columns_to_keep if col in df.columns]]

        return df

    def get_latest_price(self, ticker: str) -> float | None:
        """Get the most recent closing price for a ticker."""
        try:
            df = self.fetch(ticker)
            return float(df["close"].iloc[-1])
        except (ValueError, IndexError):
            return None

    def fetch_reference_series(self, ticker: str, align_to: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Fetch a reference price series (e.g. OMXS30, FX rates) and align it
        to the supplied DatetimeIndex.

        The result always has a ``close`` column.  Any dates in ``align_to``
        that have no data in the downloaded series are forward-filled first,
        then any remaining gaps (e.g. at the very start) are back-filled, and
        finally filled with 0 to guarantee no NaNs remain.

        Args:
            ticker: Yahoo Finance ticker symbol, e.g. ``"^OMX"``,
                    ``"USDSEK=X"``, or a common alias like ``"OMXS30"``.
            align_to: DatetimeIndex of the primary stock DataFrame that the
                      returned series should be aligned to.

        Returns:
            DataFrame with a ``close`` column indexed by ``align_to``.
        """
        resolved = self._resolve_ticker(ticker)
        stock = yf.Ticker(resolved)
        df = stock.history(period=self.period, interval=self.interval, timeout=30)

        if df.empty:
            # Return zeros for the whole requested index
            return pd.DataFrame({"close": 0.0}, index=align_to)

        df = self._preprocess(df)

        # Reindex to the primary stock's dates, fill gaps, ensure no NaNs
        aligned = df[["close"]].reindex(align_to)
        aligned = aligned.ffill().bfill().fillna(0)

        return aligned

    def fetch_cross_asset_data(self, align_to: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
        """Fetch OMXS30, USD/SEK, EUR/SEK, VIX, VSTOXX, Brent crude, and 10Y rates aligned to align_to.

        All series are fetched concurrently to minimise wall-clock time.
        """
        series_to_fetch = {
            "omxs30": "^OMX",
            "usdsek": "USDSEK=X",
            "eursek": "EURSEK=X",
            "vix": "^VIX",
            "vstoxx": "^V2TX",
            "oil": "BZ=F",
            "rates": "^TNX",
        }
        result: dict[str, pd.DataFrame] = {}
        with ThreadPoolExecutor(max_workers=len(series_to_fetch)) as pool:
            futures = {
                pool.submit(self.fetch_reference_series, ticker, align_to): key
                for key, ticker in series_to_fetch.items()
            }
            for future in as_completed(futures):
                result[futures[future]] = future.result()
        return result

    def fetch_weekly_trend(self, ticker: str) -> str:
        """Return the medium-term weekly trend for a ticker.

        Fetches up to 3 years of weekly OHLCV data and compares the latest
        closing price against the 12-week (≈ 3-month) simple moving average.

        Returns:
            ``'bullish'``  — price is above the 12-week MA (uptrend).
            ``'bearish'``  — price is at or below the 12-week MA (downtrend).
            ``'neutral'``  — insufficient data or fetch failed; caller should
                             not block a signal on a neutral reading.
        """
        try:
            weekly_fetcher = StockDataFetcher(period="3y", interval="1wk")
            df = weekly_fetcher.fetch(ticker)
            return self._compute_weekly_trend(df)
        except (ValueError, OSError):
            return "neutral"

    @staticmethod
    def _compute_weekly_trend(df: pd.DataFrame) -> str:
        """Compute trend from a weekly OHLCV DataFrame (no network calls).

        Args:
            df: DataFrame with at least a ``close`` column.

        Returns:
            ``'bullish'``, ``'bearish'``, or ``'neutral'``.
        """
        if len(df) < 12:
            return "neutral"
        ma12 = df["close"].rolling(12).mean()
        last_close = df["close"].iloc[-1]
        last_ma = ma12.iloc[-1]
        if pd.isna(last_ma):
            return "neutral"
        return "bullish" if last_close > last_ma else "bearish"

    @staticmethod
    def list_swedish_tickers() -> dict[str, str]:
        """Return a dictionary of available Swedish ticker aliases."""
        return {**SWEDISH_INDEXES, **SWEDISH_STOCKS}
