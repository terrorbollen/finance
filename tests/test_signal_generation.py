"""Tests for signal generation classes."""

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
