"""Data fetching module for Swedish stocks and indexes via yfinance."""

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
        df = stock.history(period=self.period, interval=self.interval)

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

    @staticmethod
    def list_swedish_tickers() -> dict[str, str]:
        """Return a dictionary of available Swedish ticker aliases."""
        return {**SWEDISH_INDEXES, **SWEDISH_STOCKS}
