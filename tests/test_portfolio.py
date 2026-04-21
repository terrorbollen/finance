"""Tests for portfolio backtester — PortfolioTrade, PortfolioResult, and PortfolioBacktester."""

from datetime import date

import pytest

from backtesting.portfolio import PortfolioBacktester, PortfolioResult, PortfolioTrade

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trade(
    actual_return: float | None = 0.05,
    entry_value: float = 1000.0,
    is_short: bool = False,
    effective_leverage: float = 1.0,
    commission_paid: float = 0.0,
) -> PortfolioTrade:
    return PortfolioTrade(
        ticker="VOLV-B.ST",
        open_date=date(2023, 1, 2),
        close_date=date(2023, 1, 9),
        entry_value=entry_value,
        is_short=is_short,
        effective_leverage=effective_leverage,
        actual_return=actual_return,
        commission_paid=commission_paid,
    )


def _make_result(trades: list[PortfolioTrade], initial: float = 10_000.0) -> PortfolioResult:
    final = initial + sum(t.pnl for t in trades if t.pnl is not None)
    curve = [(date(2023, 1, 2), initial), (date(2023, 1, 9), final)]
    return PortfolioResult(
        tickers=["VOLV-B.ST"],
        model_name=None,
        start_date=date(2023, 1, 2),
        end_date=date(2023, 1, 9),
        horizon_days=5,
        initial_capital=initial,
        final_capital=final,
        leverage=1.0,
        use_kelly=False,
        equity_curve=curve,
        trades=trades,
    )


# ---------------------------------------------------------------------------
# PortfolioTrade
# ---------------------------------------------------------------------------


class TestPortfolioTradePnl:
    def test_long_winning_trade(self):
        """+5% return on $1000 with no commission = $50 P&L."""
        t = _make_trade(actual_return=0.05, entry_value=1000.0, commission_paid=0.0)
        assert abs(t.pnl - 50.0) < 1e-9

    def test_long_losing_trade(self):
        """-10% return on $1000 with no commission = -$100 P&L."""
        t = _make_trade(actual_return=-0.10, entry_value=1000.0, commission_paid=0.0)
        assert abs(t.pnl - (-100.0)) < 1e-9

    def test_short_winning_trade(self):
        """Short: price falls -5% → positive P&L."""
        t = _make_trade(actual_return=-0.05, entry_value=1000.0, is_short=True, commission_paid=0.0)
        assert t.pnl > 0
        assert abs(t.pnl - 50.0) < 1e-9

    def test_short_losing_trade(self):
        """Short: price rises +5% → negative P&L."""
        t = _make_trade(actual_return=0.05, entry_value=1000.0, is_short=True, commission_paid=0.0)
        assert t.pnl < 0
        assert abs(t.pnl - (-50.0)) < 1e-9

    def test_commission_reduces_pnl(self):
        t = _make_trade(actual_return=0.05, entry_value=1000.0, commission_paid=5.0)
        assert abs(t.pnl - 45.0) < 1e-9

    def test_leverage_amplifies_pnl(self):
        """2x leverage on +5% = $100 P&L (2 × $50)."""
        t = _make_trade(
            actual_return=0.05, entry_value=1000.0, effective_leverage=2.0, commission_paid=0.0
        )
        assert abs(t.pnl - 100.0) < 1e-9

    def test_incomplete_trade_has_none_pnl(self):
        t = _make_trade(actual_return=None)
        assert t.pnl is None

    def test_is_complete(self):
        assert _make_trade(actual_return=0.02).is_complete is True
        assert _make_trade(actual_return=None).is_complete is False

    def test_is_winner_true(self):
        assert _make_trade(actual_return=0.01, commission_paid=0.0).is_winner is True

    def test_is_winner_false(self):
        assert _make_trade(actual_return=-0.01, commission_paid=0.0).is_winner is False

    def test_is_winner_none_when_incomplete(self):
        assert _make_trade(actual_return=None).is_winner is None


# ---------------------------------------------------------------------------
# PortfolioResult
# ---------------------------------------------------------------------------


class TestPortfolioResult:
    def test_total_return_positive(self):
        t = _make_trade(actual_return=0.10, entry_value=1000.0, commission_paid=0.0)
        result = _make_result([t], initial=10_000.0)
        assert abs(result.total_return - 1.0) < 0.01  # +$100 on $10k = +1%

    def test_total_return_negative(self):
        t = _make_trade(actual_return=-0.10, entry_value=1000.0, commission_paid=0.0)
        result = _make_result([t], initial=10_000.0)
        assert result.total_return < 0

    def test_trade_count(self):
        trades = [_make_trade(actual_return=0.01), _make_trade(actual_return=-0.01)]
        result = _make_result(trades)
        assert result.trade_count == 2

    def test_incomplete_trades_not_counted(self):
        complete = _make_trade(actual_return=0.05)
        incomplete = _make_trade(actual_return=None)
        result = _make_result([complete, incomplete])
        assert result.trade_count == 1

    def test_win_rate(self):
        trades = [
            _make_trade(actual_return=0.05, commission_paid=0.0),
            _make_trade(actual_return=0.03, commission_paid=0.0),
            _make_trade(actual_return=-0.02, commission_paid=0.0),
            _make_trade(actual_return=-0.04, commission_paid=0.0),
        ]
        result = _make_result(trades)
        assert abs(result.win_rate - 0.5) < 1e-9

    def test_win_rate_no_trades(self):
        result = _make_result([])
        assert result.win_rate == 0.0

    def test_max_drawdown_flat_curve(self):
        result = _make_result([])
        result.equity_curve = [(date(2023, 1, 2), 10_000.0), (date(2023, 1, 9), 10_000.0)]
        assert result.max_drawdown == 0.0

    def test_max_drawdown_peak_to_trough(self):
        result = _make_result([])
        result.equity_curve = [
            (date(2023, 1, 2), 10_000.0),
            (date(2023, 1, 3), 12_000.0),   # peak
            (date(2023, 1, 4), 9_000.0),    # trough: (9k-12k)/12k = -25%
        ]
        assert abs(result.max_drawdown - (-25.0)) < 0.01

    def test_sharpe_ratio_two_points(self):
        """With fewer than 2 completed trades, Sharpe should be 0."""
        result = _make_result([_make_trade(actual_return=0.05)])
        assert result.sharpe_ratio == 0.0

    def test_sharpe_ratio_all_winners_returns_nonzero(self):
        trades = [
            _make_trade(actual_return=0.05, entry_value=1000.0, commission_paid=0.0),
            _make_trade(actual_return=0.03, entry_value=1000.0, commission_paid=0.0),
            _make_trade(actual_return=0.04, entry_value=1000.0, commission_paid=0.0),
        ]
        result = _make_result(trades)
        assert result.sharpe_ratio != 0.0

    def test_per_ticker_stats_empty_when_no_trades(self):
        result = _make_result([], initial=10_000.0)
        stats = result.per_ticker_stats()
        assert stats["VOLV-B.ST"]["trades"] == 0


# ---------------------------------------------------------------------------
# PortfolioBacktester._kelly_fraction
# ---------------------------------------------------------------------------


class TestKellyFraction:
    @pytest.fixture
    def bt(self) -> PortfolioBacktester:
        bt = PortfolioBacktester.__new__(PortfolioBacktester)
        bt.stop_loss_pct = 0.05
        bt.kelly_max = 3.0
        bt.max_positions = 5
        bt._cal_buckets = []
        return bt

    def test_negative_kelly_returns_zero(self, bt: PortfolioBacktester):
        """When p*b < q, expected value is negative — no trade."""
        fraction = bt._kelly_fraction(confidence=0.3, predicted_change_pct=1.0)
        assert fraction == 0.0

    def test_positive_confidence_gives_positive_fraction(self, bt: PortfolioBacktester):
        fraction = bt._kelly_fraction(confidence=0.7, predicted_change_pct=5.0)
        assert fraction > 0.0

    def test_capped_at_kelly_max_over_max_positions(self, bt: PortfolioBacktester):
        """Fraction must not exceed kelly_max / max_positions."""
        assert bt.max_positions is not None
        cap = bt.kelly_max / bt.max_positions
        fraction = bt._kelly_fraction(confidence=0.99, predicted_change_pct=50.0)
        assert fraction <= cap + 1e-9

    def test_confidence_clamped_to_one(self, bt: PortfolioBacktester):
        """Confidence > 1.0 is clamped to 1.0 — fraction must still be finite and positive."""
        assert bt.max_positions is not None
        fraction = bt._kelly_fraction(confidence=2.0, predicted_change_pct=5.0)
        assert 0.0 < fraction <= bt.kelly_max / bt.max_positions

    def test_kelly_position_uses_current_capital(self, bt: PortfolioBacktester):
        """After a drawdown, sizing must scale to current capital, not initial capital."""
        kelly_f = 0.5
        current_capital = 2_000.0

        position_capital = bt._kelly_position_size(current_capital, kelly_f)

        assert abs(position_capital - 1_000.0) < 1e-9
        assert position_capital < current_capital

    def test_kelly_position_capped_at_available_capital(self, bt: PortfolioBacktester):
        """kelly_f > 1.0 must not produce a position larger than available capital."""
        # kelly_f can exceed 1.0 when kelly_max / max_positions > 1 (e.g. 3.0/1).
        # Committing more cash than available would cause the downstream guard to skip
        # the trade entirely; capping at capital is preferable (invest at most 100%).
        position_capital = bt._kelly_position_size(capital=1_000.0, kelly_f=2.0)
        assert position_capital <= 1_000.0


# ---------------------------------------------------------------------------
# PortfolioBacktester._calibrate
# ---------------------------------------------------------------------------


class TestCalibrateMethod:
    def test_no_buckets_returns_raw(self):
        bt = PortfolioBacktester.__new__(PortfolioBacktester)
        bt._cal_buckets = []
        assert bt._calibrate(0.75) == 0.75

    def test_matching_bucket_returns_calibrated(self):
        bt = PortfolioBacktester.__new__(PortfolioBacktester)
        bt._cal_buckets = [{"raw_min": 0.7, "raw_max": 0.8, "calibrated": 0.65}]
        assert abs(bt._calibrate(0.75) - 0.65) < 1e-9

    def test_no_matching_bucket_returns_raw(self):
        bt = PortfolioBacktester.__new__(PortfolioBacktester)
        bt._cal_buckets = [{"raw_min": 0.0, "raw_max": 0.5, "calibrated": 0.4}]
        assert bt._calibrate(0.9) == 0.9


# ---------------------------------------------------------------------------
# PortfolioBacktester._portfolio_value
# ---------------------------------------------------------------------------


class TestPortfolioValue:
    def test_no_open_positions(self):
        assert PortfolioBacktester._portfolio_value(5_000.0, {}) == 5_000.0

    def test_with_open_positions(self):
        open_pos = {
            "A": (0, 5, 1_000.0, None),
            "B": (0, 5, 2_000.0, None),
        }
        val = PortfolioBacktester._portfolio_value(3_000.0, open_pos)
        assert abs(val - 6_000.0) < 1e-9
