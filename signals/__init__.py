"""Live signal generation and confidence calibration.

This module takes a trained model and produces actionable trading signals for
a given ticker. It is independent of backtesting/ and must not import from it.

Public API:
  SignalGenerator   — loads model weights + config, fetches live data, runs
                      the full inference pipeline, and returns a Signal
  Signal            — dataclass: direction, confidence, entry/target/stop
                      prices, position size, and calibration metadata
  (calibration.py is internal — use SignalGenerator which loads it automatically)

Key contracts:
  - Normalization uses training statistics from the config, never current-data
    stats (see INVARIANTS.md).
  - Confidence calibration priority: DirectionalCalibrator (per BUY/SELL/HOLD)
    → ConfidenceCalibrator (global) → raw model probability.
  - Calibrators are loaded from checkpoints/ at construction time. Refit after
    every retraining run with: uv run python main.py calibrate
  - Stop-loss and take-profit are ATR-based. ATR is % of price (see INVARIANTS.md).
  - Position size is half-Kelly, capped at max_position_size (default 25%).
  - This module must not import from backtesting/.

See ARCHITECTURE.md §6-8 for the inference flow.
"""

from signals.generator import Signal, SignalGenerator

__all__ = ["SignalGenerator", "Signal"]
