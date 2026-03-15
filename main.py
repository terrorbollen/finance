#!/usr/bin/env python3
"""CLI entry point for the trading signal generator."""

import argparse
import sys
from datetime import UTC, datetime

from backtesting import Backtester, PortfolioBacktester
from backtesting.results import HorizonMetrics, Signal
from data.fetcher import StockDataFetcher
from models.mlflow_tracking import (
    get_recent_runs,
    log_hyperparameters,
    log_metrics,
    setup_mlflow,
    training_run,
)
from models.config import ModelConfig
from models.training import ModelTrainer
from models.walk_forward import WalkForwardTrainer
from signals.calibration import ConfidenceCalibrator
from signals.generator import SignalGenerator


def cmd_analyze(args):
    """Analyze a single ticker and generate a signal."""
    print(f"\nAnalyzing {args.ticker}...")

    paths = ModelConfig.checkpoint_paths(getattr(args, "name", None))
    generator = SignalGenerator(
        model_path=paths["weights"],
        calibration_path=paths["calibration"],
        min_confidence=args.min_confidence,
    )
    try:
        signal = generator.generate(args.ticker)
        print(signal)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def cmd_scan(args):
    """Scan multiple tickers and generate signals."""
    tickers = args.tickers or list(StockDataFetcher.list_swedish_tickers().keys())[:10]

    print(f"\nScanning {len(tickers)} tickers...")
    print("-" * 50)

    paths = ModelConfig.checkpoint_paths(getattr(args, "name", None))
    generator = SignalGenerator(
        model_path=paths["weights"],
        calibration_path=paths["calibration"],
        min_confidence=args.min_confidence,
    )
    signals = generator.scan(tickers)

    if not signals:
        print("No signals generated.")
        return

    print(f"\nGenerated {len(signals)} signals (sorted by confidence):\n")

    for signal in signals:
        print(signal)
        print()


def cmd_train(args):
    """Train the signal model."""
    if args.tickers:
        tickers = args.tickers
    else:
        # Default training tickers - expanded for more diverse training data
        tickers = [
            "VOLV-B.ST",  # Volvo
            "ERIC-B.ST",  # Ericsson
            "HM-B.ST",  # H&M
            "SEB-A.ST",  # SEB Bank
            "ATCO-A.ST",  # Atlas Copco
            "INVE-B.ST",  # Investor
            "NDA-SE.ST",  # Nordea
            "SWED-A.ST",  # Swedbank
            "SAND.ST",  # Sandvik
            "ABB.ST",  # ABB
        ]

    print(f"\nTraining model on {len(tickers)} tickers...")
    print(f"Epochs: {args.epochs}")
    print(f"Walk-forward: {args.walk_forward}")
    print("-" * 50)

    track = not args.no_mlflow
    cli_tags = {
        "cli.epochs": str(args.epochs),
        "cli.batch_size": str(args.batch_size),
        "cli.walk_forward": str(args.walk_forward),
        "cli.focal_loss": str(not args.no_focal_loss),
        "cli.tickers": ",".join(tickers),
    }

    try:
        if args.walk_forward:
            cli_tags["cli.initial_days"] = str(args.initial_days)
            cli_tags["cli.window_size"] = str(args.window_size)
            trainer = WalkForwardTrainer(
                initial_train_days=args.initial_days,
                validation_days=args.window_size,
                step_days=args.window_size,
            )
            results = trainer.run(
                tickers=tickers,
                epochs=args.epochs,
                batch_size=args.batch_size,
                track_with_mlflow=track,
                tags=cli_tags,
            )
            print("\n" + results.summary())
        else:
            # Standard training
            from datetime import datetime as _dt
            paths = ModelConfig.checkpoint_paths(getattr(args, "name", None))
            holdout_date = _dt.strptime(args.holdout_start, "%Y-%m-%d").date() if getattr(args, "holdout_start", None) else None
            trainer = ModelTrainer(holdout_date=holdout_date)
            results = trainer.train(
                tickers=tickers,
                epochs=args.epochs,
                batch_size=args.batch_size,
                use_focal_loss=not args.no_focal_loss,
                track_with_mlflow=track,
                tags=cli_tags,
                model_path=paths["weights"],
            )
            print("\nTraining complete!")
            print(f"Final test accuracy: {results['test_signal_accuracy']:.4f}")

            # Auto-calibrate unless opted out
            if not args.no_calibrate:
                _run_calibration(
                    tickers=results["loaded_tickers"],
                    horizon=5,
                    output_path=paths["calibration"],
                    mlflow_run_id=results.get("mlflow_run_id"),
                    model_name=getattr(args, "name", None),
                )
    except Exception as e:
        print(f"Training error: {e}")
        sys.exit(1)


def cmd_list(args):
    """List available Swedish tickers."""
    tickers = StockDataFetcher.list_swedish_tickers()

    print("\nAvailable Swedish Tickers:")
    print("-" * 40)
    print(f"{'Alias':<20} {'Yahoo Symbol':<20}")
    print("-" * 40)

    for alias, symbol in sorted(tickers.items()):
        print(f"{alias:<20} {symbol:<20}")


def _run_leverage_comparison(args, start_date, end_date, horizons):
    """Run backtest at 1x, 2x, 3x leverage and print a comparison table."""
    paths = ModelConfig.checkpoint_paths(getattr(args, "name", None))
    leverages = [1.0, 2.0, 3.0]
    results = []
    for lev in leverages:
        backtester = Backtester(
            model_path=paths["weights"],
            commission_pct=args.commission,
            strict_holdout=not args.no_strict_holdout,
            leverage=lev,
            enforce_position_cooldown=args.position_cooldown,
        )
        result = backtester.run(
            ticker=args.ticker,
            start_date=start_date,
            end_date=end_date,
            horizons=horizons,
        )
        results.append((lev, result))

    # Print comparison table per horizon
    print(f"\n{'=' * 80}")
    print(f"  LEVERAGE COMPARISON: {args.ticker}")
    print(f"{'=' * 80}")
    for horizon in sorted(horizons):
        print(f"\n  Horizon {horizon}d")
        print(f"  {'Leverage':<12} {'Net Ret%':<12} {'Sharpe':<10} {'Max DD%':<12} {'Win Rate':<12} {'Trades'}")
        print(f"  {'-' * 68}")
        for lev, result in results:
            m = result.horizon_metrics.get(horizon)
            if m is None:
                continue
            low = " ⚠" if m.low_sample else ""
            print(
                f"  {lev:.0f}x{'':<9} {m.net_total_return:+.2f}%{'':<5} "
                f"{m.sharpe_ratio:.2f}{'':<6} {m.max_drawdown:+.1f}%{'':<6} "
                f"{m.win_rate * 100:.1f}%{'':<6} {m.trade_count}{low}"
            )
    print(f"\n{'=' * 80}")


def cmd_backtest(args):
    """Run backtesting on historical data."""
    print(f"\nBacktesting {args.ticker}...")

    # Parse dates
    start_date = None
    end_date = None
    if args.start_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    if args.end_date:
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()

    # Parse horizons
    horizons = args.horizons if args.horizons else [1, 2, 3, 4, 5, 6, 7]

    # --compare-leverage: run 1x/2x/3x and print a comparison table, then exit
    if args.compare_leverage:
        _run_leverage_comparison(args, start_date, end_date, horizons)
        return

    paths = ModelConfig.checkpoint_paths(getattr(args, "name", None))
    backtester = Backtester(
        model_path=paths["weights"],
        commission_pct=args.commission,
        strict_holdout=not args.no_strict_holdout,
        leverage=args.leverage,
        enforce_position_cooldown=args.position_cooldown,
    )
    try:
        result = backtester.run(
            ticker=args.ticker,
            start_date=start_date,
            end_date=end_date,
            horizons=horizons,
        )

        # Print summary
        print(result.summary())

        # Log to MLflow unless opted out
        if not args.no_mlflow:
            setup_mlflow()
            run_name = f"backtest-{args.ticker}"
            tags = {
                "run_type": "backtest",
                "ticker": args.ticker,
            }
            with training_run(run_name=run_name, tags=tags):
                log_hyperparameters(
                    {
                        "ticker": args.ticker,
                        "start_date": str(result.start_date),
                        "end_date": str(result.end_date),
                        "horizons": str(horizons),
                        "commission_pct": str(args.commission),
                    }
                )
                # Result-level metrics (comparable across runs)
                result_metrics: dict[str, float] = {
                    "trading_days": result.trading_days,
                    "buy_hold_return": result.buy_hold_return,
                }
                if result.benchmark_return is not None:
                    result_metrics["benchmark_return"] = result.benchmark_return
                log_metrics(result_metrics)
                # Log per-horizon metrics with horizon prefix
                for h, m in result.horizon_metrics.items():
                    metrics: dict = {
                        f"h{h}.accuracy": m.accuracy,
                        f"h{h}.win_rate": m.win_rate,
                        f"h{h}.net_return": m.net_total_return,
                        f"h{h}.sharpe": m.sharpe_ratio,
                        f"h{h}.sortino": m.sortino_ratio,
                        f"h{h}.max_drawdown": m.max_drawdown,
                        f"h{h}.calmar": m.calmar_ratio,
                        f"h{h}.price_mae": m.price_mae,
                        # Reliability / significance
                        f"h{h}.trade_count": m.trade_count,
                        f"h{h}.win_rate_pvalue": m.win_rate_pvalue,
                    }
                    # Per-class precision and recall
                    for sig in Signal:
                        cm = m.class_metrics.get(sig)
                        if cm:
                            key = sig.name.lower()
                            metrics[f"h{h}.{key}_precision"] = cm.precision
                            metrics[f"h{h}.{key}_recall"] = cm.recall
                            metrics[f"h{h}.{key}_support"] = cm.support
                    log_metrics(metrics)
            print("\nResults logged to MLflow.")

        # Export if requested
        if args.output:
            if args.output.endswith(".csv"):
                result.export_csv(args.output)
                print(f"\nResults exported to {args.output}")
            else:
                # Default to JSON
                output_path = (
                    args.output if args.output.endswith(".json") else args.output + ".json"
                )
                result.export_json(output_path)
                print(f"\nResults exported to {output_path}")

    except Exception as e:
        print(f"Backtest error: {e}")
        sys.exit(1)


def cmd_portfolio(args):
    """Run a portfolio backtest across multiple tickers with shared capital."""
    from datetime import datetime as _dt

    start_date = _dt.strptime(args.start_date, "%Y-%m-%d").date() if args.start_date else None
    end_date = _dt.strptime(args.end_date, "%Y-%m-%d").date() if args.end_date else None

    backtester = PortfolioBacktester(
        model_name=getattr(args, "name", None),
        commission_pct=args.commission,
        initial_capital=args.capital,
        max_positions=args.max_positions,
        strict_holdout=not args.no_strict_holdout,
        leverage=args.leverage,
        use_kelly=args.kelly,
        kelly_max=args.kelly_max,
        adx_filter=getattr(args, "adx_filter", 0.0),
        allow_short=getattr(args, "short", False),
        reversal_exit=getattr(args, "reversal_exit", False),
    )
    result = backtester.run(
        tickers=args.tickers,
        horizon=args.horizon,
        start_date=start_date,
        end_date=end_date,
    )
    print(result.summary())


def cmd_history(args):
    """Show trend of backtest and training results from MLflow."""

    setup_mlflow()

    run_type = args.type
    ticker = args.ticker if hasattr(args, "ticker") else None
    horizon = args.horizon
    n = args.runs

    runs = get_recent_runs(run_type=run_type, ticker=ticker, max_results=n)

    if not runs:
        print("No runs found. Run some backtests or training first.")
        return

    print(f"\nLast {len(runs)} '{run_type}' runs" + (f" for {ticker}" if ticker else "") + ":\n")

    if run_type == "backtest":
        hk = f"h{horizon}"
        header = f"{'Date':<22} {'Ticker':<14} {'Acc':>6} {'WinRate':>8} {'NetRet%':>8} {'Sharpe':>7} {'MaxDD%':>7}"
        print(header)
        print("-" * len(header))
        for r in runs:
            ts = r["start_time"]
            dt = datetime.fromtimestamp(ts / 1000, tz=UTC).strftime("%Y-%m-%d %H:%M")
            t = r["tags"].get("ticker", r["params"].get("ticker", "—"))
            m = r["metrics"]
            acc = f"{m.get(f'{hk}.accuracy', float('nan')):.3f}"
            wr = f"{m.get(f'{hk}.win_rate', float('nan')):.3f}"
            nr = f"{m.get(f'{hk}.net_return', float('nan')):+.2f}"
            sh = f"{m.get(f'{hk}.sharpe', float('nan')):+.2f}"
            dd = f"{m.get(f'{hk}.max_drawdown', float('nan')):+.2f}"
            print(f"{dt:<22} {t:<14} {acc:>6} {wr:>8} {nr:>8} {sh:>7} {dd:>7}")
    else:
        header = f"{'Date':<22} {'Run name':<40} {'TestAcc':>8} {'TestLoss':>9}"
        print(header)
        print("-" * len(header))
        for r in runs:
            ts = r["start_time"]
            dt = datetime.fromtimestamp(ts / 1000, tz=UTC).strftime("%Y-%m-%d %H:%M")
            name = (r["run_name"] or "—")[:39]
            acc = r["metrics"].get("test_signal_accuracy", float("nan"))
            loss = r["metrics"].get("test_loss", float("nan"))
            print(f"{dt:<22} {name:<40} {acc:>8.4f} {loss:>9.4f}")


def _run_calibration(
    tickers: list[str],
    horizon: int,
    output_path: str,
    num_buckets: int = 10,
    mlflow_run_id: str | None = None,
    model_name: str | None = None,
) -> bool:
    """
    Run backtests on tickers and fit a confidence calibrator.

    If mlflow_run_id is provided, calibration metrics are logged as a separate
    MLflow run tagged with the training run_id. This avoids re-opening a
    completed run (which is brittle) while still linking the two in the UI.

    Returns True if calibration succeeded, False otherwise.
    """
    import time

    import mlflow
    import numpy as np

    print(f"\nCalibrating confidence using {len(tickers)} tickers (horizon={horizon}d)...")
    print("-" * 50)

    all_confidences = []
    all_correct = []
    ticker_metrics: dict[str, HorizonMetrics] = {}

    cal_paths = ModelConfig.checkpoint_paths(model_name)
    backtester = Backtester(model_path=cal_paths["weights"])
    total_start = time.monotonic()

    for i, ticker in enumerate(tickers, 1):
        try:
            print(f"  [{i}/{len(tickers)}] Backtesting {ticker}...", flush=True)
            t0 = time.monotonic()
            result = backtester.run(ticker=ticker, horizons=[horizon])
            elapsed = time.monotonic() - t0
            collected = 0
            for daily in result.daily_predictions:
                pred = daily.predictions.get(horizon)
                if pred and pred.is_correct is not None:
                    all_confidences.append(pred.confidence)
                    all_correct.append(pred.is_correct)
                    collected += 1
            m = result.horizon_metrics.get(horizon)
            if m:
                ticker_metrics[ticker] = m
                print(
                    f"         done in {elapsed:.1f}s — {collected} predictions  |  "
                    f"acc {m.accuracy * 100:.1f}%  win {m.win_rate * 100:.1f}%  "
                    f"net {m.net_total_return:+.1f}%  sharpe {m.sharpe_ratio:.2f}"
                )
            else:
                print(f"         done in {elapsed:.1f}s — {collected} predictions collected")
        except Exception as e:
            print(f"  Warning: Could not backtest {ticker}: {e}")

    if len(all_confidences) < 100:
        print(f"  Not enough data ({len(all_confidences)} predictions). Skipping calibration.")
        return False

    total_elapsed = time.monotonic() - total_start
    print(f"\n  Collected {len(all_confidences)} predictions total in {total_elapsed:.1f}s")

    calibrator = ConfidenceCalibrator(num_buckets=num_buckets)
    calibrator.fit(np.array(all_confidences), np.array(all_correct))
    print("\n" + calibrator.get_calibration_table())
    calibrator.save(output_path)
    print(f"  Calibration saved to {output_path}")

    if mlflow_run_id:
        # Log calibration as its own run tagged with the training run_id.
        # This avoids re-opening a completed run (which triggers unsupported
        # API endpoints) while still linking calibration to its training run.
        overall_accuracy = float(np.mean(all_correct))
        tags = {
            "run_type": "calibration",
            "training_run_id": mlflow_run_id,
        }
        metrics: dict[str, float] = {
            "cal_overall_accuracy": overall_accuracy,
            "cal_total_predictions": float(len(all_confidences)),
        }
        for ticker, m in ticker_metrics.items():
            safe = ticker.replace(".", "_").replace("-", "_").replace("^", "")
            metrics[f"cal_{safe}_accuracy"] = m.accuracy
            metrics[f"cal_{safe}_win_rate"] = m.win_rate
            metrics[f"cal_{safe}_sharpe"] = m.sharpe_ratio
            metrics[f"cal_{safe}_net_return"] = m.net_total_return
        try:
            with mlflow.start_run(tags=tags):
                mlflow.log_metrics(metrics)
                mlflow.log_artifact(output_path, artifact_path="calibration")
        except Exception as e:
            print(f"Warning: Could not log calibration to MLflow: {e}")

    return True


def cmd_calibrate(args):
    """Train confidence calibrator from backtest results."""
    name = getattr(args, "name", None)
    tickers = args.tickers or ["VOLV-B.ST", "ERIC-B.ST", "HM-B.ST", "SEB-A.ST", "ATCO-A.ST"]
    output_path = args.output or ModelConfig.checkpoint_paths(name)["calibration"]
    horizon = args.horizon if args.horizon is not None else 5

    ok = _run_calibration(
        tickers=tickers,
        horizon=horizon,
        output_path=output_path,
        num_buckets=args.buckets,
        model_name=name,
    )
    if not ok:
        sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Trading Signal Generator for Swedish Stocks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s analyze ^OMXS30       Analyze OMX Stockholm 30 index
  %(prog)s analyze VOLV-B.ST     Analyze Volvo B shares
  %(prog)s scan                  Scan default Swedish stocks
  %(prog)s scan ERIC-B.ST HM-B.ST   Scan specific tickers
  %(prog)s train --epochs 100    Train model with 100 epochs
  %(prog)s list                  List available tickers
  %(prog)s backtest ^OMXS30      Backtest on OMX Stockholm 30
  %(prog)s backtest VOLV-B.ST --horizons 1 3 5 7  Backtest specific horizons
  %(prog)s calibrate             Calibrate confidence using default tickers
  %(prog)s calibrate --horizon 3 Train calibrator for 3-day predictions
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a single ticker")
    analyze_parser.add_argument("ticker", help="Stock ticker symbol")
    analyze_parser.add_argument(
        "--min-confidence",
        type=float,
        default=None,
        dest="min_confidence",
        help="Minimum confidence %% to act on a signal (e.g. 60); below this shows HOLD",
    )
    analyze_parser.add_argument("--name", default=None, help="Model name (e.g. 'financials')")
    analyze_parser.set_defaults(func=cmd_analyze)

    # scan command
    scan_parser = subparsers.add_parser("scan", help="Scan multiple tickers for signals")
    scan_parser.add_argument("tickers", nargs="*", help="Ticker symbols (default: Swedish stocks)")
    scan_parser.add_argument(
        "--min-confidence",
        type=float,
        default=None,
        dest="min_confidence",
        help="Minimum confidence %% to act on a signal (e.g. 60); below this shows HOLD",
    )
    scan_parser.add_argument("--name", default=None, help="Model name (e.g. 'financials')")
    scan_parser.set_defaults(func=cmd_scan)

    # train command
    train_parser = subparsers.add_parser("train", help="Train the signal model")
    train_parser.add_argument("tickers", nargs="*", help="Ticker symbols for training data")
    train_parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    train_parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    train_parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Use walk-forward training with periodic retraining",
    )
    train_parser.add_argument(
        "--initial-days",
        type=int,
        default=500,
        help="Initial training window size in days (walk-forward only)",
    )
    train_parser.add_argument(
        "--window-size",
        type=int,
        default=60,
        help="Validation window size in days (walk-forward only)",
    )
    train_parser.add_argument(
        "--no-focal-loss",
        action="store_true",
        help="Disable focal loss (use standard cross-entropy)",
    )
    train_parser.add_argument(
        "--no-mlflow", action="store_true", help="Disable MLflow experiment tracking"
    )
    train_parser.add_argument(
        "--no-calibrate",
        action="store_true",
        dest="no_calibrate",
        help="Skip automatic confidence calibration after training.",
    )
    train_parser.add_argument("--name", default=None, help="Model name — saves to checkpoints/<name>/ (e.g. 'financials')")
    train_parser.add_argument("--holdout-start", default=None, dest="holdout_start", help="Fixed holdout boundary YYYY-MM-DD — only data before this date is used for training (e.g. 2025-01-01)")
    train_parser.set_defaults(func=cmd_train)

    # list command
    list_parser = subparsers.add_parser("list", help="List available tickers")
    list_parser.set_defaults(func=cmd_list)

    # backtest command
    backtest_parser = subparsers.add_parser(
        "backtest", help="Backtest the model on historical data"
    )
    backtest_parser.add_argument("ticker", help="Stock ticker symbol")
    backtest_parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        help="Prediction horizons in days (default: 1 2 3 4 5 6 7)",
    )
    backtest_parser.add_argument(
        "--start-date",
        help="Start date for backtest (YYYY-MM-DD)",
    )
    backtest_parser.add_argument(
        "--end-date",
        help="End date for backtest (YYYY-MM-DD)",
    )
    backtest_parser.add_argument(
        "--output",
        help="Output file path for results (JSON or CSV)",
    )
    backtest_parser.add_argument(
        "--commission",
        type=float,
        default=0.001,
        help="One-way commission as decimal (default: 0.001 = 0.1%%)",
    )
    backtest_parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow logging of backtest results",
    )
    backtest_parser.add_argument(
        "--no-strict-holdout",
        action="store_true",
        dest="no_strict_holdout",
        help="Allow backtest to overlap with training data (not recommended — disables the "
             "look-ahead bias guard)",
    )
    backtest_parser.add_argument(
        "--leverage",
        type=float,
        default=1.0,
        help="Leverage multiplier applied to each trade (default: 1.0 = no leverage).",
    )
    backtest_parser.add_argument(
        "--compare-leverage",
        action="store_true",
        dest="compare_leverage",
        help="Run backtest at 1x, 2x, and 3x leverage and print a comparison table.",
    )
    backtest_parser.add_argument(
        "--position-cooldown",
        action="store_true",
        dest="position_cooldown",
        help="Enforce non-overlapping positions: after a trade, skip the next N days "
             "where N = horizon. Prevents the >100%% drawdown artefact from overlapping trades.",
    )
    backtest_parser.add_argument("--name", default=None, help="Model name — loads from checkpoints/<name>/ (e.g. 'financials')")
    backtest_parser.set_defaults(func=cmd_backtest)

    # portfolio command
    portfolio_parser = subparsers.add_parser(
        "portfolio", help="Backtest multiple tickers as a portfolio with shared capital"
    )
    portfolio_parser.add_argument("tickers", nargs="+", help="Ticker symbols")
    portfolio_parser.add_argument("--name", default=None, help="Model name (e.g. 'indexes')")
    portfolio_parser.add_argument("--horizon", type=int, default=5, help="Holding period in trading days (default: 5)")
    portfolio_parser.add_argument("--capital", type=float, default=10_000.0, help="Initial capital (default: 10000)")
    portfolio_parser.add_argument("--max-positions", type=int, default=None, dest="max_positions", help="Max concurrent positions (default: number of tickers)")
    portfolio_parser.add_argument("--commission", type=float, default=0.001, help="One-way commission as decimal (default: 0.001)")
    portfolio_parser.add_argument("--start-date", dest="start_date", help="Start date YYYY-MM-DD")
    portfolio_parser.add_argument("--end-date", dest="end_date", help="End date YYYY-MM-DD")
    portfolio_parser.add_argument("--no-strict-holdout", action="store_true", dest="no_strict_holdout", help="Allow overlap with training data")
    portfolio_parser.add_argument("--leverage", type=float, default=1.0, help="Base leverage multiplier (default: 1.0)")
    portfolio_parser.add_argument("--kelly", action="store_true", help="Use Kelly criterion to size each trade by model confidence")
    portfolio_parser.add_argument("--kelly-max", type=float, default=3.0, dest="kelly_max", help="Max leverage Kelly can assign per trade (default: 3.0)")
    portfolio_parser.add_argument("--adx-filter", type=float, default=0.0, dest="adx_filter", help="Minimum ADX(14) to allow a trade — skips signals in low-trend/ranging markets (default: 0 = disabled)")
    portfolio_parser.add_argument("--short", action="store_true", help="Also open short positions on SELL signals (default: BUY only)")
    portfolio_parser.add_argument("--reversal-exit", action="store_true", dest="reversal_exit", help="Close a position early when the model signals the opposite direction")
    portfolio_parser.set_defaults(func=cmd_portfolio)

    # history command
    history_parser = subparsers.add_parser(
        "history", help="Show trend of backtest/training results from MLflow"
    )
    history_parser.add_argument(
        "--type",
        default="backtest",
        choices=["backtest", "standard", "walk-forward"],
        help="Run type to query (default: backtest)",
    )
    history_parser.add_argument(
        "--ticker",
        default=None,
        help="Filter by ticker symbol",
    )
    history_parser.add_argument(
        "--horizon",
        type=int,
        default=5,
        help="Prediction horizon to display for backtest runs (default: 5)",
    )
    history_parser.add_argument(
        "--runs",
        type=int,
        default=20,
        help="Number of recent runs to show (default: 20)",
    )
    history_parser.set_defaults(func=cmd_history)

    # calibrate command
    calibrate_parser = subparsers.add_parser(
        "calibrate", help="Train confidence calibrator from backtest data"
    )
    calibrate_parser.add_argument("tickers", nargs="*", help="Ticker symbols for calibration data")
    calibrate_parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Prediction horizon in bars (default: 5 for 1d, 35 for 1h — matches training prediction_horizon)",
    )
    calibrate_parser.add_argument(
        "--buckets", type=int, default=10, help="Number of calibration buckets (default: 10)"
    )
    calibrate_parser.add_argument(
        "--output", help="Output path for calibration file (default: checkpoints/calibration.json)"
    )
    calibrate_parser.add_argument("--name", default=None, help="Model name — loads/saves to checkpoints/<name>/")
    calibrate_parser.set_defaults(func=cmd_calibrate)

    # Parse and execute
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
