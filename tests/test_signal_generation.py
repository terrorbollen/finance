"""Tests for signal generation classes."""

import numpy as np
import pytest

from signals.generator import Direction, Signal, SignalGenerator


class TestDirection:
    """Tests for the Direction enum."""

    def test_direction_values(self):
        """Test that Direction has expected values."""
        assert Direction.BUY.value == "BUY"
        assert Direction.HOLD.value == "HOLD"
        assert Direction.SELL.value == "SELL"

    def test_direction_from_string(self):
        """Test creating Direction from string."""
        assert Direction("BUY") == Direction.BUY
        assert Direction("HOLD") == Direction.HOLD
        assert Direction("SELL") == Direction.SELL

    def test_direction_iteration(self):
        """Test that all directions can be iterated."""
        directions = list(Direction)
        assert len(directions) == 3


class TestSignal:
    """Tests for the Signal dataclass."""

    @pytest.fixture
    def sample_signal(self) -> Signal:
        """Create a sample signal for testing."""
        return Signal(
            ticker="AAPL",
            direction=Direction.BUY,
            confidence=75.5,
            current_price=150.00,
            entry_price=150.00,
            target_price=157.50,
            stop_loss=142.50,
            predicted_change=5.0,
            timestamp="2024-01-15 10:30:00",
        )

    def test_signal_creation(self, sample_signal: Signal):
        """Test that Signal can be created with all fields."""
        assert sample_signal.ticker == "AAPL"
        assert sample_signal.direction == Direction.BUY
        assert sample_signal.confidence == 75.5
        assert sample_signal.current_price == 150.00

    def test_signal_str_format(self, sample_signal: Signal):
        """Test that Signal string representation is formatted correctly."""
        signal_str = str(sample_signal)

        assert "AAPL" in signal_str
        assert "BUY" in signal_str
        assert "75.5%" in signal_str
        assert "150.00" in signal_str

    def test_signal_calibrated_display(self):
        """Test that calibrated signal shows both raw and calibrated confidence."""
        signal = Signal(
            ticker="AAPL",
            direction=Direction.BUY,
            confidence=65.0,
            current_price=150.00,
            entry_price=150.00,
            target_price=157.50,
            stop_loss=142.50,
            predicted_change=5.0,
            timestamp="2024-01-15 10:30:00",
            raw_confidence=80.0,
            is_calibrated=True,
        )

        signal_str = str(signal)
        assert "65.0%" in signal_str
        assert "raw: 80.0%" in signal_str

    def test_signal_default_calibration(self, sample_signal: Signal):
        """Test that Signal has correct default calibration values."""
        assert sample_signal.raw_confidence is None
        assert sample_signal.is_calibrated is False

    def test_signal_stop_loss_buy(self, sample_signal: Signal):
        """Test that BUY signal has stop loss below entry."""
        assert sample_signal.stop_loss < sample_signal.entry_price

    def test_signal_target_buy(self, sample_signal: Signal):
        """Test that BUY signal has target above entry."""
        assert sample_signal.target_price > sample_signal.entry_price

    def test_signal_predicted_change_matches_target(self, sample_signal: Signal):
        """Test that predicted change matches target calculation."""
        expected_change = (
            (sample_signal.target_price - sample_signal.current_price)
            / sample_signal.current_price
            * 100
        )
        assert abs(sample_signal.predicted_change - expected_change) < 0.1

    def test_sell_signal(self):
        """Test SELL signal has correct price relationships."""
        signal = Signal(
            ticker="AAPL",
            direction=Direction.SELL,
            confidence=70.0,
            current_price=150.00,
            entry_price=150.00,
            target_price=142.50,  # Lower for SELL
            stop_loss=157.50,  # Higher for SELL (stop loss above)
            predicted_change=-5.0,
            timestamp="2024-01-15 10:30:00",
        )

        assert signal.target_price < signal.entry_price
        assert signal.stop_loss > signal.entry_price
        assert signal.predicted_change < 0


class TestKellyPositionSize:
    """Tests for Kelly criterion position sizing."""

    @pytest.fixture
    def gen(self) -> SignalGenerator:
        gen = SignalGenerator.__new__(SignalGenerator)
        gen.stop_loss_pct = 0.05
        gen.max_position_size = 0.25
        return gen

    def test_hold_returns_zero(self, gen: SignalGenerator):
        assert gen._kelly_position_size(70, 3.0, 2.0, Direction.HOLD) == 0.0

    def test_negative_kelly_returns_zero(self, gen: SignalGenerator):
        # 40% confidence with bad odds → negative Kelly
        assert gen._kelly_position_size(40, 1.0, 3.0, Direction.BUY) == 0.0

    def test_position_capped_at_max(self, gen: SignalGenerator):
        # Very high confidence should still be capped
        size = gen._kelly_position_size(95, 10.0, 1.0, Direction.BUY)
        assert size <= gen.max_position_size

    def test_higher_confidence_gives_larger_position(self, gen: SignalGenerator):
        low = gen._kelly_position_size(55, 3.0, 2.0, Direction.BUY)
        high = gen._kelly_position_size(80, 3.0, 2.0, Direction.BUY)
        assert high > low

    def test_sell_direction_same_as_buy(self, gen: SignalGenerator):
        buy = gen._kelly_position_size(70, 3.0, 2.0, Direction.BUY)
        sell = gen._kelly_position_size(70, 3.0, 2.0, Direction.SELL)
        assert buy == sell

    def test_returns_fraction_between_zero_and_max(self, gen: SignalGenerator):
        size = gen._kelly_position_size(65, 3.0, 2.0, Direction.BUY)
        assert 0.0 <= size <= gen.max_position_size


class TestPortfolioRiskLimits:
    """Tests for portfolio-level risk limit enforcement."""

    @pytest.fixture
    def gen(self) -> SignalGenerator:
        gen = SignalGenerator.__new__(SignalGenerator)
        gen.max_drawdown_pct = 10.0
        gen.max_positions = 5
        return gen

    def test_no_limit_breached_returns_original(self, gen: SignalGenerator):
        result = gen._apply_portfolio_limits(Direction.BUY, "AAPL", 5.0, 3)
        assert result == Direction.BUY

    def test_drawdown_at_limit_forces_hold(self, gen: SignalGenerator):
        result = gen._apply_portfolio_limits(Direction.BUY, "AAPL", 10.0, 0)
        assert result == Direction.HOLD

    def test_drawdown_above_limit_forces_hold(self, gen: SignalGenerator):
        result = gen._apply_portfolio_limits(Direction.SELL, "AAPL", 15.0, 0)
        assert result == Direction.HOLD

    def test_drawdown_below_limit_passes_through(self, gen: SignalGenerator):
        result = gen._apply_portfolio_limits(Direction.BUY, "AAPL", 9.9, 0)
        assert result == Direction.BUY

    def test_max_positions_at_limit_forces_hold(self, gen: SignalGenerator):
        result = gen._apply_portfolio_limits(Direction.BUY, "AAPL", 0.0, 5)
        assert result == Direction.HOLD

    def test_max_positions_above_limit_forces_hold(self, gen: SignalGenerator):
        result = gen._apply_portfolio_limits(Direction.SELL, "AAPL", 0.0, 7)
        assert result == Direction.HOLD

    def test_max_positions_below_limit_passes_through(self, gen: SignalGenerator):
        result = gen._apply_portfolio_limits(Direction.BUY, "AAPL", 0.0, 4)
        assert result == Direction.BUY

    def test_hold_is_never_suppressed_by_drawdown(self, gen: SignalGenerator):
        result = gen._apply_portfolio_limits(Direction.HOLD, "AAPL", 99.0, 0)
        assert result == Direction.HOLD

    def test_hold_is_never_suppressed_by_position_count(self, gen: SignalGenerator):
        result = gen._apply_portfolio_limits(Direction.HOLD, "AAPL", 0.0, 99)
        assert result == Direction.HOLD

    def test_none_drawdown_limit_disabled(self):
        gen = SignalGenerator.__new__(SignalGenerator)
        gen.max_drawdown_pct = None
        gen.max_positions = None
        result = gen._apply_portfolio_limits(Direction.BUY, "AAPL", 99.0, 99)
        assert result == Direction.BUY

    def test_sell_direction_also_suppressed(self, gen: SignalGenerator):
        result = gen._apply_portfolio_limits(Direction.SELL, "AAPL", 12.0, 0)
        assert result == Direction.HOLD


class TestAtrTakeProfit:
    """Tests for ATR-based take-profit target calculation."""

    @pytest.fixture
    def gen(self) -> SignalGenerator:
        gen = SignalGenerator.__new__(SignalGenerator)
        gen.stop_loss_pct = 0.05
        gen.atr_multiplier = 2.0
        gen.take_profit_atr_multiplier = 3.0
        gen.max_position_size = 0.25
        return gen

    def _run(self, gen, direction, atr_pct, predicted_change, current_price=100.0):
        """Simulate the price-target block from generate()."""
        stop_distance = (atr_pct / 100) * gen.atr_multiplier
        take_profit_distance = (atr_pct / 100) * gen.take_profit_atr_multiplier

        if direction == Direction.SELL:
            stop_loss = current_price * (1 + stop_distance)
            target_price = current_price * (1 - take_profit_distance)
            predicted_change = -take_profit_distance * 100
        elif direction == Direction.BUY:
            stop_loss = current_price * (1 - stop_distance)
            target_price = current_price * (1 + take_profit_distance)
            predicted_change = take_profit_distance * 100
        else:
            stop_loss = current_price * (1 - stop_distance)
            target_price = current_price * (1 + predicted_change / 100)

        return target_price, stop_loss, predicted_change

    def test_buy_target_above_entry(self, gen: SignalGenerator):
        target, stop, _ = self._run(gen, Direction.BUY, atr_pct=1.0, predicted_change=2.0)
        assert target > 100.0

    def test_sell_target_below_entry(self, gen: SignalGenerator):
        target, stop, _ = self._run(gen, Direction.SELL, atr_pct=1.0, predicted_change=-2.0)
        assert target < 100.0

    def test_buy_reward_risk_ratio(self, gen: SignalGenerator):
        # With atr_multiplier=2, take_profit_atr_multiplier=3 → R:R = 3/2 = 1.5
        atr_pct = 1.0
        target, stop, _ = self._run(gen, Direction.BUY, atr_pct=atr_pct, predicted_change=0.0)
        reward = target - 100.0
        risk = 100.0 - stop
        assert abs(reward / risk - 1.5) < 0.01

    def test_sell_reward_risk_ratio(self, gen: SignalGenerator):
        atr_pct = 1.0
        target, stop, _ = self._run(gen, Direction.SELL, atr_pct=atr_pct, predicted_change=0.0)
        reward = 100.0 - target
        risk = stop - 100.0
        assert abs(reward / risk - 1.5) < 0.01

    def test_predicted_change_matches_target(self, gen: SignalGenerator):
        target, _, change = self._run(gen, Direction.BUY, atr_pct=2.0, predicted_change=0.0)
        expected_change = (target - 100.0) / 100.0 * 100
        assert abs(change - expected_change) < 0.001

    def test_larger_atr_gives_wider_target(self, gen: SignalGenerator):
        target_small, _, _ = self._run(gen, Direction.BUY, atr_pct=1.0, predicted_change=0.0)
        target_large, _, _ = self._run(gen, Direction.BUY, atr_pct=2.0, predicted_change=0.0)
        assert target_large > target_small


class TestSignalTimestamp:
    """S4 — signal timestamp must reflect the last data bar, not wall-clock time."""

    def _make_df_features(self, last_date: str):
        import pandas as pd

        dates = pd.date_range(end=last_date, periods=5, freq="B")
        return pd.DataFrame({"close": [100.0] * 5}, index=dates)

    def _make_gen(self, df_features, last_date: str):
        """Build a minimal SignalGenerator with mocked internals."""
        from unittest.mock import MagicMock, patch

        gen = SignalGenerator.__new__(SignalGenerator)
        gen.feature_mean = None
        gen.feature_std = None
        gen.feature_columns = None
        gen.input_dim = 1
        gen.sequence_length = 3
        gen.prediction_horizons = [5]
        gen.min_confidence = None
        gen.max_drawdown_pct = None
        gen.max_positions = None
        gen.min_volume_ratio = None
        gen.earnings_buffer_days = None
        gen.require_weekly_confirmation = False
        gen.atr_multiplier = 2.0
        gen.take_profit_atr_multiplier = 3.0
        gen.stop_loss_pct = 0.05
        gen.max_position_size = 0.25
        gen.calibrator = None
        gen.directional_calibrator = None

        fake_model = MagicMock()
        fake_model.predict.return_value = (
            np.array([[0.80, 0.10, 0.10]]),
            np.array([0]),
            np.array([2.0]),
        )
        gen.model = fake_model

        # fetcher is an instance attribute set in __init__; assign directly since
        # we bypassed __init__ with __new__
        mock_fetcher = MagicMock()
        mock_fetcher.fetch.return_value = df_features
        mock_fetcher.fetch_cross_asset_data.return_value = {}
        gen.fetcher = mock_fetcher

        return gen, patch

    def test_timestamp_equals_last_data_bar(self):
        """generate() must stamp the signal with the last bar's date, not today."""
        import pandas as pd

        last_date = "2023-06-15"
        df_features = self._make_df_features(last_date)
        gen, patch = self._make_gen(df_features, last_date)

        with patch("signals.generator.FeatureEngineer") as MockFE:
            from unittest.mock import MagicMock

            mock_fe = MagicMock()
            mock_fe.add_all_features.return_value = df_features
            mock_fe.get_feature_columns.return_value = ["close"]
            MockFE.return_value = mock_fe

            signal = gen.generate("TEST.ST")

        expected_ts = pd.Timestamp(last_date).strftime("%Y-%m-%d %H:%M:%S")
        assert signal.timestamp == expected_ts, (
            f"Expected timestamp {expected_ts!r}, got {signal.timestamp!r}. "
            "Signal timestamp must use the last data bar's date."
        )

    def test_timestamp_is_not_today_for_historical_data(self):
        """Timestamp must not silently use today when historical data is passed."""
        import pandas as pd

        last_date = "2021-01-04"
        df_features = self._make_df_features(last_date)
        gen, patch = self._make_gen(df_features, last_date)

        with patch("signals.generator.FeatureEngineer") as MockFE:
            from unittest.mock import MagicMock

            mock_fe = MagicMock()
            mock_fe.add_all_features.return_value = df_features
            mock_fe.get_feature_columns.return_value = ["close"]
            MockFE.return_value = mock_fe

            signal = gen.generate("TEST.ST")

        today = pd.Timestamp.now().strftime("%Y-%m-%d")
        assert not signal.timestamp.startswith(today), (
            "Historical signal must not be stamped with today's date."
        )
        assert signal.timestamp.startswith("2021-01-04")


class TestCheckOod:
    """Tests for the out-of-distribution feature detection warning."""

    def _gen(self, n_features: int = 4) -> SignalGenerator:
        gen = SignalGenerator.__new__(SignalGenerator)
        gen.feature_mean = np.zeros(n_features)
        gen.feature_std = np.ones(n_features)
        return gen

    def test_no_warning_when_all_in_distribution(self, capsys):
        gen = self._gen(n_features=3)
        features_norm = np.array([[0.5, -1.0, 2.9]])  # all within ±3σ
        gen._check_ood(features_norm, ["a", "b", "c"])
        captured = capsys.readouterr()
        assert captured.err == ""

    def test_warning_printed_to_stderr_for_ood_feature(self, capsys):
        gen = self._gen(n_features=3)
        # Feature "b" is 4.5σ from mean → OOD
        features_norm = np.array([[1.0, 4.5, 0.0]])
        gen._check_ood(features_norm, ["a", "b", "c"])
        captured = capsys.readouterr()
        assert "OOD WARNING" in captured.err
        assert "b" in captured.err
        assert "+4.5σ" in captured.err

    def test_multiple_ood_features_all_named(self, capsys):
        gen = self._gen(n_features=3)
        features_norm = np.array([[-5.0, 0.0, 3.5]])  # a and c are OOD
        gen._check_ood(features_norm, ["alpha", "beta", "gamma"])
        captured = capsys.readouterr()
        assert "alpha" in captured.err
        assert "gamma" in captured.err
        assert "beta" not in captured.err

    def test_most_recent_row_is_checked(self, capsys):
        gen = self._gen(n_features=2)
        # Only the last row is extreme; earlier rows are fine
        features_norm = np.array([[0.0, 0.0], [0.0, 0.0], [10.0, 0.0]])
        gen._check_ood(features_norm, ["x", "y"])
        captured = capsys.readouterr()
        assert "OOD WARNING" in captured.err

    def test_no_warning_without_training_stats(self, capsys):
        gen = SignalGenerator.__new__(SignalGenerator)
        gen.feature_mean = None
        gen.feature_std = None
        features_norm = np.array([[100.0, -50.0]])
        gen._check_ood(features_norm, ["x", "y"])
        captured = capsys.readouterr()
        assert captured.err == ""

    def test_custom_threshold_respected(self, capsys):
        gen = self._gen(n_features=2)
        # 2.5σ is fine at threshold=3.0 but triggers at threshold=2.0
        features_norm = np.array([[2.5, 0.0]])
        gen._check_ood(features_norm, ["x", "y"], threshold=2.0)
        captured = capsys.readouterr()
        assert "OOD WARNING" in captured.err


# ---------------------------------------------------------------------------
# S1 — signal filters: low-volume and earnings
# ---------------------------------------------------------------------------


def _make_gen_with_filters(
    min_volume_ratio: float | None = None,
    earnings_buffer_days: int | None = None,
) -> "SignalGenerator":
    """Build a minimal SignalGenerator with S1 filter attributes set."""
    gen = SignalGenerator.__new__(SignalGenerator)
    gen.min_volume_ratio = min_volume_ratio
    gen.earnings_buffer_days = earnings_buffer_days
    return gen


def _df_with_volume_ratio(ratio: float):  # noqa: ANN201
    import pandas as pd

    return pd.DataFrame({"volume_ratio": [ratio]})


class TestVolumeFilter:
    """S1 — low-volume filter inside _apply_signal_filters()."""

    def test_low_volume_forces_hold(self, capsys):
        gen = _make_gen_with_filters(min_volume_ratio=0.5)
        df = _df_with_volume_ratio(0.3)
        result = gen._apply_signal_filters(Direction.BUY, "X.ST", df)
        assert result == Direction.HOLD
        assert "Low-volume" in capsys.readouterr().out

    def test_adequate_volume_passes_through(self):
        gen = _make_gen_with_filters(min_volume_ratio=0.5)
        df = _df_with_volume_ratio(0.8)
        result = gen._apply_signal_filters(Direction.BUY, "X.ST", df)
        assert result == Direction.BUY

    def test_exactly_at_threshold_passes_through(self):
        gen = _make_gen_with_filters(min_volume_ratio=0.5)
        df = _df_with_volume_ratio(0.5)
        result = gen._apply_signal_filters(Direction.BUY, "X.ST", df)
        assert result == Direction.BUY

    def test_disabled_when_min_volume_ratio_is_none(self):
        gen = _make_gen_with_filters(min_volume_ratio=None)
        df = _df_with_volume_ratio(0.01)  # would trigger if enabled
        result = gen._apply_signal_filters(Direction.BUY, "X.ST", df)
        assert result == Direction.BUY

    def test_missing_volume_column_does_not_crash(self):
        import pandas as pd

        gen = _make_gen_with_filters(min_volume_ratio=0.5)
        df = pd.DataFrame({"close": [100.0]})  # no volume_ratio column
        result = gen._apply_signal_filters(Direction.BUY, "X.ST", df)
        assert result == Direction.BUY  # no crash, filter simply skipped

    def test_sell_direction_also_suppressed(self, capsys):
        gen = _make_gen_with_filters(min_volume_ratio=0.5)
        df = _df_with_volume_ratio(0.2)
        result = gen._apply_signal_filters(Direction.SELL, "X.ST", df)
        assert result == Direction.HOLD

    def test_hold_short_circuits_no_volume_check(self, capsys):
        gen = _make_gen_with_filters(min_volume_ratio=0.5)
        df = _df_with_volume_ratio(0.1)  # would trigger if BUY/SELL
        result = gen._apply_signal_filters(Direction.HOLD, "X.ST", df)
        assert result == Direction.HOLD
        assert "Low-volume" not in capsys.readouterr().out


class TestEarningsFilter:
    """S1 — earnings proximity filter inside _apply_signal_filters()."""

    def test_within_buffer_forces_hold(self, capsys):
        from unittest.mock import MagicMock, patch

        gen = _make_gen_with_filters(earnings_buffer_days=5)
        import pandas as pd

        # Earnings 2 days from now
        earnings_date = (pd.Timestamp.now() + pd.Timedelta(days=2)).date()
        mock_ticker = MagicMock()
        mock_ticker.calendar = {"Earnings Date": [earnings_date]}

        with patch("signals.generator.yf.Ticker", return_value=mock_ticker):
            result = gen._apply_signal_filters(
                Direction.BUY, "X.ST", _df_with_volume_ratio(1.0)
            )

        assert result == Direction.HOLD
        assert "Earnings" in capsys.readouterr().out

    def test_outside_buffer_passes_through(self):
        from unittest.mock import MagicMock, patch

        gen = _make_gen_with_filters(earnings_buffer_days=5)
        import pandas as pd

        earnings_date = (pd.Timestamp.now() + pd.Timedelta(days=30)).date()
        mock_ticker = MagicMock()
        mock_ticker.calendar = {"Earnings Date": [earnings_date]}

        with patch("signals.generator.yf.Ticker", return_value=mock_ticker):
            result = gen._apply_signal_filters(
                Direction.BUY, "X.ST", _df_with_volume_ratio(1.0)
            )

        assert result == Direction.BUY

    def test_unavailable_earnings_data_does_not_crash(self):
        from unittest.mock import MagicMock, patch

        gen = _make_gen_with_filters(earnings_buffer_days=5)
        mock_ticker = MagicMock()
        mock_ticker.calendar = None  # no data

        with patch("signals.generator.yf.Ticker", return_value=mock_ticker):
            result = gen._apply_signal_filters(
                Direction.BUY, "X.ST", _df_with_volume_ratio(1.0)
            )

        assert result == Direction.BUY  # graceful fallback

    def test_yfinance_exception_does_not_crash(self):
        from unittest.mock import patch

        gen = _make_gen_with_filters(earnings_buffer_days=5)

        with patch("signals.generator.yf.Ticker", side_effect=Exception("network error")):
            result = gen._apply_signal_filters(
                Direction.BUY, "X.ST", _df_with_volume_ratio(1.0)
            )

        assert result == Direction.BUY  # graceful fallback

    def test_disabled_when_earnings_buffer_days_is_none(self):
        gen = _make_gen_with_filters(earnings_buffer_days=None)
        # No yfinance call should happen at all
        result = gen._apply_signal_filters(
            Direction.BUY, "X.ST", _df_with_volume_ratio(1.0)
        )
        assert result == Direction.BUY

    def test_hold_short_circuits_no_earnings_check(self):
        from unittest.mock import patch

        gen = _make_gen_with_filters(earnings_buffer_days=5)

        called = []
        with patch("signals.generator.yf.Ticker", side_effect=lambda t: called.append(t)):
            gen._apply_signal_filters(Direction.HOLD, "X.ST", _df_with_volume_ratio(1.0))

        assert called == []  # yfinance never called for HOLD


# ---------------------------------------------------------------------------
# S3 — multi-timeframe confirmation (daily signal confirmed by weekly trend)
# ---------------------------------------------------------------------------


class TestComputeWeeklyTrend:
    """Unit tests for StockDataFetcher._compute_weekly_trend().

    Tests the pure trend calculation with synthetic DataFrames — no network calls.
    """

    def _df(self, closes: list[float]):
        import pandas as pd

        dates = pd.date_range(end="2024-01-01", periods=len(closes), freq="W")
        return pd.DataFrame({"close": closes}, index=dates)

    def test_price_above_ma12_is_bullish(self):
        from data.fetcher import StockDataFetcher

        # 11 bars at 100 + 1 bar at 120 → MA12 ≈ 101.7, close = 120 → bullish
        closes = [100.0] * 11 + [120.0]
        assert StockDataFetcher._compute_weekly_trend(self._df(closes)) == "bullish"

    def test_price_below_ma12_is_bearish(self):
        from data.fetcher import StockDataFetcher

        # 11 bars at 100 + 1 bar at 80 → MA12 ≈ 98.3, close = 80 → bearish
        closes = [100.0] * 11 + [80.0]
        assert StockDataFetcher._compute_weekly_trend(self._df(closes)) == "bearish"

    def test_fewer_than_12_bars_returns_neutral(self):
        from data.fetcher import StockDataFetcher

        closes = [100.0] * 10  # only 10 weeks of data
        assert StockDataFetcher._compute_weekly_trend(self._df(closes)) == "neutral"

    def test_exactly_12_bars_uses_full_ma(self):
        from data.fetcher import StockDataFetcher

        # Flat at 100 for 11, then spike to 150 → above MA → bullish
        closes = [100.0] * 11 + [150.0]
        assert StockDataFetcher._compute_weekly_trend(self._df(closes)) == "bullish"

    def test_empty_dataframe_returns_neutral(self):
        import pandas as pd

        from data.fetcher import StockDataFetcher

        df = pd.DataFrame({"close": []})
        assert StockDataFetcher._compute_weekly_trend(df) == "neutral"


class TestApplyWeeklyConfirmation:
    """Unit tests for SignalGenerator._apply_weekly_confirmation().

    Mocks fetcher.fetch_weekly_trend to avoid network calls.
    """

    def _gen(self, require: bool, trend: str) -> "SignalGenerator":
        from unittest.mock import MagicMock

        gen = SignalGenerator.__new__(SignalGenerator)
        gen.require_weekly_confirmation = require
        mock_fetcher = MagicMock()
        mock_fetcher.fetch_weekly_trend.return_value = trend
        gen.fetcher = mock_fetcher
        return gen

    def test_buy_confirmed_by_bullish_weekly(self):
        gen = self._gen(require=True, trend="bullish")
        assert gen._apply_weekly_confirmation(Direction.BUY, "X.ST") == Direction.BUY

    def test_buy_rejected_by_bearish_weekly(self):
        gen = self._gen(require=True, trend="bearish")
        assert gen._apply_weekly_confirmation(Direction.BUY, "X.ST") == Direction.HOLD

    def test_sell_confirmed_by_bearish_weekly(self):
        gen = self._gen(require=True, trend="bearish")
        assert gen._apply_weekly_confirmation(Direction.SELL, "X.ST") == Direction.SELL

    def test_sell_rejected_by_bullish_weekly(self):
        gen = self._gen(require=True, trend="bullish")
        assert gen._apply_weekly_confirmation(Direction.SELL, "X.ST") == Direction.HOLD

    def test_buy_passes_on_neutral_trend(self):
        """Neutral trend (insufficient data) must not block the signal."""
        gen = self._gen(require=True, trend="neutral")
        assert gen._apply_weekly_confirmation(Direction.BUY, "X.ST") == Direction.BUY

    def test_sell_passes_on_neutral_trend(self):
        gen = self._gen(require=True, trend="neutral")
        assert gen._apply_weekly_confirmation(Direction.SELL, "X.ST") == Direction.SELL

    def test_hold_always_passes_through(self):
        """HOLD is never affected by weekly confirmation."""
        gen = self._gen(require=True, trend="bearish")
        assert gen._apply_weekly_confirmation(Direction.HOLD, "X.ST") == Direction.HOLD

    def test_disabled_does_not_call_fetcher(self):
        """When require_weekly_confirmation=False, fetcher must never be called."""
        gen = self._gen(require=False, trend="bearish")
        result = gen._apply_weekly_confirmation(Direction.BUY, "X.ST")
        assert result == Direction.BUY
        gen.fetcher.fetch_weekly_trend.assert_not_called()

    def test_disabled_buy_not_blocked(self):
        gen = self._gen(require=False, trend="bearish")
        assert gen._apply_weekly_confirmation(Direction.BUY, "X.ST") == Direction.BUY

    def test_disabled_sell_not_blocked(self):
        gen = self._gen(require=False, trend="bullish")
        assert gen._apply_weekly_confirmation(Direction.SELL, "X.ST") == Direction.SELL
