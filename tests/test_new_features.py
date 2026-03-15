"""Tests for leverage, annualization, threshold loading, feature mismatch,
Signal/Direction enum unification, and equity curve correctness."""

import math
from datetime import date
from unittest.mock import MagicMock

import numpy as np
import pytest

from backtesting.backtester import Backtester
from backtesting.metrics import MetricsCalculator
from backtesting.results import HorizonPrediction, Signal
from signals.direction import Direction


# ---------------------------------------------------------------------------
# Helpers (mirrors test_metrics.py style)
# ---------------------------------------------------------------------------

def _pred(
    predicted: Signal,
    actual: Signal,
    actual_change: float = 0.0,
    prediction_date: date = date(2024, 1, 2),
    relative_volume: float | None = None,
) -> HorizonPrediction:
    p = HorizonPrediction(
        prediction_date=prediction_date,
        horizon_days=5,
        predicted_signal=predicted,
        confidence=0.75,
        predicted_price_change=2.0,
        relative_volume=relative_volume,
    )
    p.actual_signal = actual
    p.actual_price_change = actual_change
    return p


def _calc(**kwargs) -> MetricsCalculator:
    defaults = {
        "buy_threshold": 0.02,
        "sell_threshold": -0.02,
        "commission_pct": 0.0,
        "slippage_factor": 0.0,
    }
    defaults.update(kwargs)
    return MetricsCalculator(**defaults)


# ---------------------------------------------------------------------------
# Signal / Direction enum unification
# ---------------------------------------------------------------------------

class TestSignalDirectionEnum:
    def test_signal_is_direction(self):
        assert Signal is Direction

    def test_signal_buy_name(self):
        assert Signal.BUY.name == "BUY"

    def test_signal_hold_name(self):
        assert Signal.HOLD.name == "HOLD"

    def test_signal_sell_name(self):
        assert Signal.SELL.name == "SELL"

    def test_signal_buy_value(self):
        assert Signal.BUY.value == "BUY"

    def test_horizon_prediction_serialises_enum_name(self):
        p = _pred(Signal.BUY, Signal.BUY)
        d = p.to_dict()
        assert d["predicted_signal"] == "BUY"
        assert d["actual_signal"] == "BUY"

    def test_index_lookup_matches_enum(self):
        # This is how backtester constructs the signal from model output
        for idx, expected in enumerate([Signal.BUY, Signal.HOLD, Signal.SELL]):
            assert [Signal.BUY, Signal.HOLD, Signal.SELL][idx] == expected


# ---------------------------------------------------------------------------
# Leverage scaling
# ---------------------------------------------------------------------------

class TestLeverage:
    def test_1x_leverage_unchanged(self):
        calc = _calc(leverage=1.0)
        preds = [_pred(Signal.BUY, Signal.BUY, actual_change=3.0)]
        result = calc._calculate_trading_metrics(preds)
        assert result["total_return"] == pytest.approx(3.0)

    def test_2x_leverage_doubles_gross(self):
        calc = _calc(leverage=2.0)
        preds = [_pred(Signal.BUY, Signal.BUY, actual_change=3.0)]
        result = calc._calculate_trading_metrics(preds)
        assert result["total_return"] == pytest.approx(6.0)

    def test_3x_leverage_triples_gross(self):
        calc = _calc(leverage=3.0)
        preds = [_pred(Signal.BUY, Signal.BUY, actual_change=2.0)]
        result = calc._calculate_trading_metrics(preds)
        assert result["total_return"] == pytest.approx(6.0)

    def test_leverage_scales_loss(self):
        calc = _calc(leverage=2.0)
        preds = [_pred(Signal.BUY, Signal.SELL, actual_change=-3.0)]
        result = calc._calculate_trading_metrics(preds)
        assert result["total_return"] == pytest.approx(-6.0)

    def test_leverage_scales_commission(self):
        # With 1x: net = gross - 2*commission
        # With 2x: net = 2*gross - 2*2*commission
        gross = 2.0
        commission = 0.001
        calc_1x = _calc(leverage=1.0, commission_pct=commission)
        calc_2x = _calc(leverage=2.0, commission_pct=commission)
        preds_1x = [_pred(Signal.BUY, Signal.BUY, actual_change=gross)]
        preds_2x = [_pred(Signal.BUY, Signal.BUY, actual_change=gross)]
        result_1x = calc_1x._calculate_trading_metrics(preds_1x)
        result_2x = calc_2x._calculate_trading_metrics(preds_2x)
        expected_net_1x = gross - 2 * commission * 100
        expected_net_2x = gross * 2 - 2 * 2 * commission * 100
        assert result_1x["net_total_return"] == pytest.approx(expected_net_1x)
        assert result_2x["net_total_return"] == pytest.approx(expected_net_2x)

    def test_leverage_scales_max_drawdown(self):
        # Drawdown should scale linearly with leverage
        returns = [3.0, -2.0, 1.0]
        preds = [_pred(Signal.BUY, Signal.BUY, actual_change=r) for r in returns]
        calc_1x = _calc(leverage=1.0)
        calc_2x = _calc(leverage=2.0)
        r1 = calc_1x._calculate_trading_metrics(preds)
        r2 = calc_2x._calculate_trading_metrics(preds)
        assert r2["max_drawdown"] == pytest.approx(r1["max_drawdown"] * 2, rel=0.01)

    def test_leverage_does_not_affect_win_rate(self):
        # Win rate is directional — leverage doesn't change which trades won
        preds = [
            _pred(Signal.BUY, Signal.BUY, actual_change=2.0),
            _pred(Signal.BUY, Signal.SELL, actual_change=-1.0),
        ]
        calc_1x = _calc(leverage=1.0)
        calc_3x = _calc(leverage=3.0)
        r1 = calc_1x._calculate_trading_metrics(preds)
        r3 = calc_3x._calculate_trading_metrics(preds)
        assert r1["win_rate"] == r3["win_rate"]

    def test_sell_signal_with_leverage(self):
        # SELL signal: profit when price goes down
        calc = _calc(leverage=2.0)
        preds = [_pred(Signal.SELL, Signal.SELL, actual_change=-3.0)]
        result = calc._calculate_trading_metrics(preds)
        assert result["total_return"] == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# Annualization factor
# ---------------------------------------------------------------------------

class TestAnnualizationFactor:
    def _sharpe(self, interval: str, returns: list[float]) -> float:
        calc = _calc(interval=interval)
        sharpe, _, _, _ = calc._calculate_risk_metrics(returns)
        return sharpe

    def test_1d_uses_252_bars(self):
        returns = [1.0, 2.0, -1.0, 3.0, 0.5] * 10
        sharpe_1d = self._sharpe("1d", returns)
        arr = np.array(returns)
        expected = float(np.mean(arr) / np.std(arr, ddof=1) * math.sqrt(252))
        assert sharpe_1d == pytest.approx(expected, rel=1e-6)

    def test_1h_uses_252x7_bars(self):
        returns = [1.0, 2.0, -1.0, 3.0, 0.5] * 10
        sharpe_1h = self._sharpe("1h", returns)
        arr = np.array(returns)
        expected = float(np.mean(arr) / np.std(arr, ddof=1) * math.sqrt(252 * 7))
        assert sharpe_1h == pytest.approx(expected, rel=1e-6)

    def test_1h_sharpe_higher_than_1d_same_returns(self):
        # Higher annualization factor → higher Sharpe for positive-mean returns
        returns = [1.0, 2.0, 1.5, 1.0, 2.0] * 5
        sharpe_1d = self._sharpe("1d", returns)
        sharpe_1h = self._sharpe("1h", returns)
        assert sharpe_1h > sharpe_1d

    def test_ratio_of_sharpes_equals_sqrt_7(self):
        returns = [1.0, 2.0, -0.5, 1.5, 0.8] * 10
        sharpe_1d = self._sharpe("1d", returns)
        sharpe_1h = self._sharpe("1h", returns)
        assert sharpe_1h / sharpe_1d == pytest.approx(math.sqrt(7), rel=1e-6)

    def test_sortino_also_uses_correct_factor(self):
        # Must include enough negative returns so downside_std > 0 in both cases
        returns = [1.0, -1.0, 2.0, -2.0, 1.5, -0.5, 1.0, -1.0, 2.0, -1.5] * 5
        calc_1d = _calc(interval="1d")
        calc_1h = _calc(interval="1h")
        _, sortino_1d, _, _ = calc_1d._calculate_risk_metrics(returns)
        _, sortino_1h, _, _ = calc_1h._calculate_risk_metrics(returns)
        assert sortino_1h / sortino_1d == pytest.approx(math.sqrt(7), rel=1e-6)


# ---------------------------------------------------------------------------
# Threshold loading from ModelConfig
# ---------------------------------------------------------------------------

class TestThresholdLoading:
    def _write_config(self, path: str, buy_threshold: float, sell_threshold: float) -> None:
        from models.config import ModelConfig
        cfg = ModelConfig(
            feature_columns=[f"f{i}" for i in range(5)],
            feature_mean=[0.0] * 5,
            feature_std=[1.0] * 5,
            sequence_length=20,
            input_dim=5,
            interval="1d",
            training_fetch_date=date(2024, 1, 1),
            holdout_start_date=date(2024, 6, 1),
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
        )
        cfg.save(path)

    def test_thresholds_loaded_from_config(self, tmp_path):
        weights = str(tmp_path / "signal_model.weights.h5")
        config = weights.replace(".weights.h5", "_config.json")
        self._write_config(config, buy_threshold=0.03, sell_threshold=-0.03)

        bt = Backtester.__new__(Backtester)
        bt.model_path = weights
        bt.buy_threshold = 0.02  # default — should be overridden
        bt.sell_threshold = -0.02
        bt.metrics_calculator = MetricsCalculator()
        bt.sequence_length = 20
        bt.feature_columns = None
        bt.feature_mean = None
        bt.feature_std = None
        bt.input_dim = None
        bt.holdout_start_date = None
        bt._load_config()

        assert bt.buy_threshold == pytest.approx(0.03)
        assert bt.sell_threshold == pytest.approx(-0.03)

    def test_thresholds_propagated_to_metrics_calculator(self, tmp_path):
        weights = str(tmp_path / "signal_model.weights.h5")
        config = weights.replace(".weights.h5", "_config.json")
        self._write_config(config, buy_threshold=0.025, sell_threshold=-0.025)

        bt = Backtester.__new__(Backtester)
        bt.model_path = weights
        bt.buy_threshold = 0.02
        bt.sell_threshold = -0.02
        bt.metrics_calculator = MetricsCalculator()
        bt.sequence_length = 20
        bt.feature_columns = None
        bt.feature_mean = None
        bt.feature_std = None
        bt.input_dim = None
        bt.holdout_start_date = None
        bt._load_config()

        assert bt.metrics_calculator.buy_threshold == pytest.approx(0.025)
        assert bt.metrics_calculator.sell_threshold == pytest.approx(-0.025)

    def test_determine_actual_signal_respects_loaded_threshold(self):
        # With threshold 0.03, a 2.5% move is HOLD (not BUY)
        calc = MetricsCalculator(buy_threshold=0.03, sell_threshold=-0.03)
        assert calc.determine_actual_signal(0.025) == Signal.HOLD

    def test_determine_actual_signal_default_threshold(self):
        # With default 0.02, a 2.5% move is BUY
        calc = MetricsCalculator(buy_threshold=0.02, sell_threshold=-0.02)
        assert calc.determine_actual_signal(0.025) == Signal.BUY


# ---------------------------------------------------------------------------
# Feature mismatch — calls real backtester.run() code path
# ---------------------------------------------------------------------------

class TestFeatureMismatch:
    """Patches StockDataFetcher so run() hits the real feature-check code."""

    def _make_df(self, columns: list[str], n_rows: int = 60) -> "pd.DataFrame":
        import pandas as pd
        dates = pd.date_range("2023-01-02", periods=n_rows, freq="B")
        data = {c: np.ones(n_rows) for c in columns}
        return pd.DataFrame(data, index=dates)

    def _backtester(self, feature_columns: list[str]) -> Backtester:
        bt = Backtester.__new__(Backtester)
        bt.interval = "1d"
        bt.model_path = "nonexistent.weights.h5"
        bt.sequence_length = 5
        bt.buy_threshold = 0.02
        bt.sell_threshold = -0.02
        bt.strict_holdout = False
        bt.slippage_factor = 0.0
        bt.holdout_start_date = None
        bt.feature_columns = feature_columns
        bt.feature_mean = np.zeros(len(feature_columns))
        bt.feature_std = np.ones(len(feature_columns))
        bt.input_dim = len(feature_columns)
        bt.model = MagicMock()
        bt.model.predict.return_value = (
            np.array([[0.8, 0.1, 0.1]]),
            np.array([0]),
            np.array([2.0]),
        )
        bt.metrics_calculator = MetricsCalculator()
        return bt

    def test_raises_when_over_10_percent_missing(self, monkeypatch):
        import backtesting.backtester as bt_module

        feature_columns = [f"f{i}" for i in range(20)]
        # DataFrame only has 17 of 20 features (15% missing → should raise)
        df = self._make_df([f"f{i}" for i in range(17)])

        mock_fetcher = MagicMock()
        mock_fetcher.fetch.return_value = df
        mock_fetcher.fetch_cross_asset_data.return_value = {}
        monkeypatch.setattr(bt_module, "StockDataFetcher", lambda **_: mock_fetcher)
        monkeypatch.setattr(bt_module, "FeatureEngineer", lambda df, **_: MagicMock(
            add_all_features=lambda: df,
            get_feature_columns=lambda: list(df.columns),
        ))

        bt = self._backtester(feature_columns)
        with pytest.raises(ValueError, match="Too many features missing"):
            bt.run("FAKE.ST")

    def test_no_raise_when_within_10_percent(self, monkeypatch):
        import backtesting.backtester as bt_module
        from datetime import date as dt

        feature_columns = [f"f{i}" for i in range(20)]
        # DataFrame has 19 of 20 features + required close/volume cols (5% missing → should not raise)
        df = self._make_df([f"f{i}" for i in range(19)] + ["close", "volume"])

        mock_fetcher = MagicMock()
        mock_fetcher.fetch.return_value = df
        mock_fetcher.fetch_cross_asset_data.return_value = {}
        monkeypatch.setattr(bt_module, "StockDataFetcher", lambda **_: mock_fetcher)
        monkeypatch.setattr(bt_module, "FeatureEngineer", lambda df, **_: MagicMock(
            add_all_features=lambda: df,
            get_feature_columns=lambda: list(df.columns),
        ))

        bt = self._backtester(feature_columns)
        # Should not raise — runs to completion (may fail later for unrelated reasons)
        try:
            bt.run("FAKE.ST", start_date=dt(2023, 1, 2), end_date=dt(2023, 3, 31))
        except ValueError as e:
            assert "Too many features missing" not in str(e)


# ---------------------------------------------------------------------------
# Equity curve
# ---------------------------------------------------------------------------

class TestEquityCurve:
    def test_curve_sorted_chronologically(self):
        preds = [
            _pred(Signal.BUY, Signal.BUY, actual_change=2.0, prediction_date=date(2024, 1, 5)),
            _pred(Signal.BUY, Signal.BUY, actual_change=1.0, prediction_date=date(2024, 1, 3)),
            _pred(Signal.BUY, Signal.BUY, actual_change=3.0, prediction_date=date(2024, 1, 7)),
        ]
        result = _calc()._calculate_trading_metrics(preds)
        dates = [pt[0] for pt in result["equity_curve"]]
        assert dates == sorted(dates)

    def test_curve_is_cumulative(self):
        preds = [
            _pred(Signal.BUY, Signal.BUY, actual_change=2.0, prediction_date=date(2024, 1, 1)),
            _pred(Signal.BUY, Signal.BUY, actual_change=1.0, prediction_date=date(2024, 1, 2)),
            _pred(Signal.BUY, Signal.BUY, actual_change=1.5, prediction_date=date(2024, 1, 3)),
        ]
        result = _calc()._calculate_trading_metrics(preds)
        curve = result["equity_curve"]
        assert curve[0][1] == pytest.approx(2.0)
        assert curve[1][1] == pytest.approx(3.0)
        assert curve[2][1] == pytest.approx(4.5)

    def test_curve_reflects_leverage(self):
        preds = [
            _pred(Signal.BUY, Signal.BUY, actual_change=2.0, prediction_date=date(2024, 1, 1)),
            _pred(Signal.BUY, Signal.BUY, actual_change=1.0, prediction_date=date(2024, 1, 2)),
        ]
        result = _calc(leverage=2.0)._calculate_trading_metrics(preds)
        curve = result["equity_curve"]
        assert curve[0][1] == pytest.approx(4.0)   # 2.0 * 2x
        assert curve[1][1] == pytest.approx(6.0)   # (2.0 + 1.0) * 2x

    def test_curve_empty_when_all_hold(self):
        preds = [_pred(Signal.HOLD, Signal.HOLD)]
        result = _calc()._calculate_trading_metrics(preds)
        assert result["equity_curve"] == []

    def test_curve_length_matches_trade_count(self):
        preds = [
            _pred(Signal.BUY, Signal.BUY, actual_change=1.0),
            _pred(Signal.HOLD, Signal.HOLD),  # not a trade
            _pred(Signal.SELL, Signal.SELL, actual_change=-2.0),
        ]
        result = _calc()._calculate_trading_metrics(preds)
        assert len(result["equity_curve"]) == result["trade_count"] == 2


# ---------------------------------------------------------------------------
# ModelConfig.checkpoint_paths
# ---------------------------------------------------------------------------

class TestCheckpointPaths:
    def test_1d_paths_have_no_suffix(self):
        from models.config import ModelConfig
        paths = ModelConfig.checkpoint_paths("1d")
        assert "signal_model.weights.h5" in paths["weights"]
        assert "calibration.json" in paths["calibration"]
        assert "_1d" not in paths["weights"]

    def test_1h_paths_have_1h_suffix(self):
        from models.config import ModelConfig
        paths = ModelConfig.checkpoint_paths("1h")
        assert "signal_model_1h.weights.h5" in paths["weights"]
        assert "calibration_1h.json" in paths["calibration"]
        assert "calibration_1h_directional.json" in paths["calibration_directional"]

    def test_all_keys_present(self):
        from models.config import ModelConfig
        paths = ModelConfig.checkpoint_paths("1d")
        for key in ("weights", "config", "calibration", "calibration_directional"):
            assert key in paths
