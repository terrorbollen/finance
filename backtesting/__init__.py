"""Backtesting module for evaluating trading signal model performance."""

from backtesting.backtester import Backtester
from backtesting.metrics import MetricsCalculator
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
]
