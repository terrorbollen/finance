"""Tests for signal generation classes."""

import pytest

from signals.generator import Direction, Signal


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
