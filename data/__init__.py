"""Data fetching and feature engineering.

This module is the entry point of the pipeline. It owns everything up to the
point where a normalized, feature-enriched DataFrame is ready for the model.

Public API:
  StockDataFetcher  — fetches OHLCV data from Yahoo Finance; also fetches
                      cross-asset reference data (OMXS30, USD/SEK, EUR/SEK)
  FeatureEngineer   — adds ~37 scale-independent technical, market-regime,
                      cross-asset, and calendar features to an OHLCV DataFrame

Boundary rules:
  - This module must not import from models/, signals/, or backtesting/.
  - All features must be scale-independent (ratios, percentages, bounded
    indicators). The model must never see raw prices.
  - ATR is stored as % of price, not absolute value.
  - FeatureEngineer drops NaN rows after adding features (rolling warm-up).
    Downstream code must account for the reduced row count.

See ARCHITECTURE.md §1-2 for data shapes and INVARIANTS.md for ATR encoding.
"""

from data.features import FeatureEngineer
from data.fetcher import StockDataFetcher

__all__ = ["StockDataFetcher", "FeatureEngineer"]
