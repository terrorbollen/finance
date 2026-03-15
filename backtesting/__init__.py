"""Historical simulation and performance measurement.

This module evaluates the trained model against past data it never saw during
training. It is independent of signals/ and must not import from it.

Public API:
  Backtester        — runs the model day-by-day over a holdout window with no
                      lookahead; enforces the holdout start date from the config
  MetricsCalculator — computes HorizonMetrics from a list of HorizonPredictions
                      (Sharpe, Sortino, drawdown, Calmar, win rate + p-value)
  BacktestResult    — top-level result container; exports CSV, JSON, equity curve
  HorizonPrediction — one prediction for one day at one horizon; stores both
                      the prediction and the actual outcome once it's known
  DailyPrediction   — all horizon predictions made on a single day
  HorizonMetrics    — computed statistics for a single prediction horizon
  ClassMetrics      — per-class precision and recall

Key contracts:
  - Backtester enforces strict_holdout by default — the backtest start date
    must be >= holdout_start_date from the model config. Never disable this
    in production (see INVARIANTS.md).
  - Normalization uses training statistics from the config, not current-data
    stats (see INVARIANTS.md).
  - Commission and slippage are applied to every round-trip trade. Net return
    is always after costs. Gross return is logged separately for diagnostics.
  - This module must not import from signals/.

See backtesting/STRATEGY.md for how to interpret results and what thresholds
make a backtest result trustworthy.
"""

from backtesting.backtester import Backtester
from backtesting.metrics import MetricsCalculator
from backtesting.portfolio import PortfolioBacktester, PortfolioResult
from backtesting.results import (
    BacktestResult,
    ClassMetrics,
    DailyPrediction,
    HorizonMetrics,
    HorizonPrediction,
    Signal,
)

__all__ = [
    "Signal",
    "HorizonPrediction",
    "DailyPrediction",
    "ClassMetrics",
    "HorizonMetrics",
    "BacktestResult",
    "MetricsCalculator",
    "Backtester",
    "PortfolioBacktester",
    "PortfolioResult",
]
