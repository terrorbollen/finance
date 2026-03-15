"""Portfolio backtester — runs signals on multiple tickers with shared capital.

Design:
- Fixed allocation per ticker: 1 / max_positions of total capital.
- Positions open on BUY, close automatically after horizon trading days.
- One position per ticker at a time (no pyramiding).
- Commission applied on entry and exit.
- Shared capital pool — allocations are drawn from and returned to it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np

from backtesting.backtester import Backtester
from backtesting.results import HorizonPrediction, Signal
from models.config import ModelConfig


@dataclass
class PortfolioTrade:
    ticker: str
    open_date: date
    close_date: date          # The prediction_date that closes this trade
    entry_value: float        # Capital committed at open
    actual_return: float | None = None   # Decimal (e.g. 0.05 = +5%)
    commission_paid: float = 0.0

    @property
    def is_complete(self) -> bool:
        return self.actual_return is not None

    @property
    def pnl(self) -> float | None:
        if self.actual_return is None:
            return None
        return self.entry_value * self.actual_return - self.commission_paid

    @property
    def is_winner(self) -> bool | None:
        p = self.pnl
        return p > 0 if p is not None else None


@dataclass
class PortfolioResult:
    tickers: list[str]
    model_name: str | None
    start_date: date
    end_date: date
    horizon_days: int
    initial_capital: float
    final_capital: float
    equity_curve: list[tuple[date, float]]
    trades: list[PortfolioTrade] = field(default_factory=list)

    @property
    def completed_trades(self) -> list[PortfolioTrade]:
        return [t for t in self.trades if t.is_complete]

    @property
    def total_return(self) -> float:
        return (self.final_capital / self.initial_capital - 1) * 100

    @property
    def trade_count(self) -> int:
        return len(self.completed_trades)

    @property
    def win_rate(self) -> float:
        c = self.completed_trades
        if not c:
            return 0.0
        return sum(1 for t in c if t.is_winner) / len(c)

    @property
    def sharpe_ratio(self) -> float:
        rets = [t.pnl / t.entry_value for t in self.completed_trades if t.entry_value > 0 and t.pnl is not None]
        if len(rets) < 2:
            return 0.0
        arr = np.array(rets)
        if arr.std() == 0:
            return 0.0
        trades_per_year = 252 / self.horizon_days
        return float(arr.mean() / arr.std() * np.sqrt(trades_per_year))

    @property
    def max_drawdown(self) -> float:
        if len(self.equity_curve) < 2:
            return 0.0
        values = [v for _, v in self.equity_curve]
        peak, max_dd = values[0], 0.0
        for v in values:
            if v > peak:
                peak = v
            dd = (v - peak) / peak * 100
            if dd < max_dd:
                max_dd = dd
        return max_dd

    def per_ticker_stats(self) -> dict[str, dict]:
        stats: dict[str, dict] = {}
        for ticker in self.tickers:
            ticker_trades = [t for t in self.completed_trades if t.ticker == ticker]
            if not ticker_trades:
                stats[ticker] = {"trades": 0, "win_rate": 0.0, "net_pnl": 0.0}
                continue
            wins = sum(1 for t in ticker_trades if t.is_winner)
            net = sum(t.pnl for t in ticker_trades if t.pnl is not None)
            stats[ticker] = {
                "trades": len(ticker_trades),
                "win_rate": wins / len(ticker_trades),
                "net_pnl": net,
            }
        return stats

    def summary(self) -> str:
        lines = [
            "=" * 72,
            "  PORTFOLIO BACKTEST RESULTS",
            "=" * 72,
            f"  Tickers : {', '.join(self.tickers)}",
            f"  Model   : {self.model_name or 'default'}",
            f"  Period  : {self.start_date} → {self.end_date}  |  {self.horizon_days}d horizon",
            f"  Capital : {self.initial_capital:,.0f} → {self.final_capital:,.2f}",
            "",
            f"  Total Return  : {self.total_return:+.2f}%",
            f"  Sharpe Ratio  : {self.sharpe_ratio:.2f}",
            f"  Max Drawdown  : {self.max_drawdown:.1f}%",
            f"  Total Trades  : {self.trade_count}",
            f"  Win Rate      : {self.win_rate * 100:.1f}%",
            "",
            f"  {'Ticker':<14} {'Trades':>7} {'Win Rate':>9} {'Net P&L':>10}",
            "  " + "-" * 44,
        ]
        for ticker, s in self.per_ticker_stats().items():
            if s["trades"] > 0:
                wr_str = f"{s['win_rate'] * 100:.1f}%"
                pnl_str = f"{s['net_pnl']:+.2f}"
            else:
                wr_str = pnl_str = "—"
            lines.append(f"  {ticker:<14} {s['trades']:>7} {wr_str:>9} {pnl_str:>10}")
        lines.append("=" * 72)
        return "\n".join(lines)


class PortfolioBacktester:
    """
    Runs signals on multiple tickers simultaneously with shared capital.

    Capital allocation: each ticker gets a fixed slice (1 / max_positions).
    Positions open on BUY signals and close after `horizon` trading days.
    Only one open position per ticker at a time.
    """

    def __init__(
        self,
        model_name: str | None = None,
        commission_pct: float = 0.001,
        initial_capital: float = 10_000.0,
        max_positions: int | None = None,
        strict_holdout: bool = True,
    ):
        self.model_name = model_name
        self.commission_pct = commission_pct
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.strict_holdout = strict_holdout

    def run(
        self,
        tickers: list[str],
        horizon: int = 5,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> PortfolioResult:
        paths = ModelConfig.checkpoint_paths(self.model_name)
        max_pos = self.max_positions or len(tickers)
        alloc_value = self.initial_capital / max_pos

        # --- Step 1: collect per-ticker predictions ---
        print(f"Collecting predictions for {len(tickers)} tickers (horizon={horizon}d)...")
        # pred_by_ticker[ticker][prediction_date] = HorizonPrediction
        pred_by_ticker: dict[str, dict[date, HorizonPrediction]] = {}

        for ticker in tickers:
            bt = Backtester(model_path=paths["weights"], strict_holdout=self.strict_holdout)
            result = bt.run(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                horizons=[horizon],
            )
            ticker_preds: dict[date, HorizonPrediction] = {}
            for daily in result.daily_predictions:
                pred = daily.predictions.get(horizon)
                if pred is not None:
                    ticker_preds[pred.prediction_date] = pred
            pred_by_ticker[ticker] = ticker_preds
            n = len(ticker_preds)
            buys = sum(1 for p in ticker_preds.values() if p.predicted_signal == Signal.BUY)
            print(f"  {ticker:<16} {n} predictions  ({buys} BUY)")

        # --- Step 2: build unified sorted trading day timeline ---
        all_dates = sorted({d for preds in pred_by_ticker.values() for d in preds})
        if not all_dates:
            raise ValueError("No predictions collected — check tickers and date range")

        # --- Step 3: day-by-day simulation ---
        capital = self.initial_capital
        # open_pos: ticker -> (open_date_idx, close_date_idx, entry_value, PortfolioTrade)
        open_pos: dict[str, tuple[int, int, float, PortfolioTrade]] = {}
        completed: list[PortfolioTrade] = []
        equity_curve: list[tuple[date, float]] = [(all_dates[0], capital)]

        for i, current_date in enumerate(all_dates):
            # Close positions whose horizon has elapsed
            to_close = [tkr for tkr, (_, close_idx, _, _) in open_pos.items() if close_idx <= i]
            for tkr in to_close:
                _, _, entry_val, trade = open_pos.pop(tkr)
                pred = pred_by_ticker[tkr].get(trade.open_date)
                actual_return = (pred.actual_price_change / 100.0) if (
                    pred is not None and pred.actual_price_change is not None
                ) else None
                trade.actual_return = actual_return
                commission = entry_val * self.commission_pct * 2
                trade.commission_paid = commission
                if actual_return is not None:
                    capital += entry_val + entry_val * actual_return - commission
                else:
                    capital += entry_val  # return capital as-is for incomplete predictions
                completed.append(trade)
                equity_curve.append((current_date, self._portfolio_value(capital, open_pos)))

            # Open new positions on BUY signals
            for ticker in tickers:
                if ticker in open_pos:
                    continue  # already holding
                if len(open_pos) >= max_pos:
                    break
                pred = pred_by_ticker[ticker].get(current_date)
                if pred is None or pred.predicted_signal != Signal.BUY:
                    continue
                if capital < alloc_value:
                    continue  # not enough cash
                capital -= alloc_value
                close_idx = min(i + horizon, len(all_dates) - 1)
                close_date = all_dates[close_idx]
                trade = PortfolioTrade(
                    ticker=ticker,
                    open_date=current_date,
                    close_date=close_date,
                    entry_value=alloc_value,
                )
                open_pos[ticker] = (i, close_idx, alloc_value, trade)
                equity_curve.append((current_date, self._portfolio_value(capital, open_pos)))

        # Mark any still-open trades as incomplete and return their capital
        remaining: list[PortfolioTrade] = []
        for _, (_, _, entry_val, trade) in open_pos.items():
            capital += entry_val
            remaining.append(trade)

        all_trades = completed + remaining
        final_capital = capital
        equity_curve.append((all_dates[-1], final_capital))

        return PortfolioResult(
            tickers=tickers,
            model_name=self.model_name,
            start_date=all_dates[0],
            end_date=all_dates[-1],
            horizon_days=horizon,
            initial_capital=self.initial_capital,
            final_capital=final_capital,
            equity_curve=equity_curve,
            trades=all_trades,
        )

    @staticmethod
    def _portfolio_value(cash: float, open_pos: dict) -> float:
        return cash + sum(entry_val for _, (_, _, entry_val, _) in open_pos.items())
