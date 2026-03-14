#!/usr/bin/env python3
"""CLI entry point for the trading signal generator."""

import argparse
import sys
from datetime import datetime

from backtesting import Backtester
from data.fetcher import StockDataFetcher
from models.training import ModelTrainer
from models.walk_forward import WalkForwardTrainer
from signals.calibration import ConfidenceCalibrator
from signals.generator import SignalGenerator


def cmd_analyze(args):
    """Analyze a single ticker and generate a signal."""
    print(f"\nAnalyzing {args.ticker}...")

    generator = SignalGenerator(min_confidence=args.min_confidence)
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

    generator = SignalGenerator(min_confidence=args.min_confidence)
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
            "^OMX",           # OMX Stockholm 30 Index
            "VOLV-B.ST",      # Volvo
            "ERIC-B.ST",      # Ericsson
            "HM-B.ST",        # H&M
            "SEB-A.ST",       # SEB Bank
            "ATCO-A.ST",      # Atlas Copco
            "INVE-B.ST",      # Investor
            "NDA-SE.ST",      # Nordea
            "SWED-A.ST",      # Swedbank
            "SAND.ST",        # Sandvik
            "ABB.ST",         # ABB
        ]

    print(f"\nTraining model on {len(tickers)} tickers...")
    print(f"Epochs: {args.epochs}")
    print(f"Walk-forward: {args.walk_forward}")
    print("-" * 50)

    try:
        if args.walk_forward:
            # Walk-forward training
            trainer = WalkForwardTrainer(
                initial_train_days=args.initial_days,
                validation_days=args.window_size,
                step_days=args.window_size,
            )
            results = trainer.run(
                tickers=tickers,
                epochs=args.epochs,
                batch_size=args.batch_size,
            )
            print("\n" + results.summary())
        else:
            # Standard training
            trainer = ModelTrainer()
            results = trainer.train(
                tickers=tickers,
                epochs=args.epochs,
                batch_size=args.batch_size,
                use_focal_loss=not args.no_focal_loss,
            )
            print("\nTraining complete!")
            print(f"Final test accuracy: {results['test_signal_accuracy']:.4f}")
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

    backtester = Backtester(commission_pct=args.commission)
    try:
        result = backtester.run(
            ticker=args.ticker,
            start_date=start_date,
            end_date=end_date,
            horizons=horizons,
        )

        # Print summary
        print(result.summary())

        # Export if requested
        if args.output:
            if args.output.endswith(".csv"):
                result.export_csv(args.output)
                print(f"\nResults exported to {args.output}")
            else:
                # Default to JSON
                output_path = args.output if args.output.endswith(".json") else args.output + ".json"
                result.export_json(output_path)
                print(f"\nResults exported to {output_path}")

    except Exception as e:
        print(f"Backtest error: {e}")
        sys.exit(1)


def cmd_calibrate(args):
    """Train confidence calibrator from backtest results."""
    import numpy as np

    tickers = args.tickers or ["VOLV-B.ST", "ERIC-B.ST", "HM-B.ST", "SEB-A.ST", "ATCO-A.ST"]

    print(f"\nCalibrating confidence using {len(tickers)} tickers...")
    print(f"Horizon: {args.horizon} days")
    print("-" * 50)

    # Collect backtest results
    all_confidences = []
    all_correct = []

    backtester = Backtester()

    for ticker in tickers:
        try:
            print(f"Backtesting {ticker}...")
            result = backtester.run(
                ticker=ticker,
                horizons=[args.horizon],
            )

            # Extract confidence and correctness
            for daily in result.daily_predictions:
                pred = daily.predictions.get(args.horizon)
                if pred and pred.is_correct is not None:
                    all_confidences.append(pred.confidence)
                    all_correct.append(pred.is_correct)

        except Exception as e:
            print(f"  Warning: Could not backtest {ticker}: {e}")

    if len(all_confidences) < 100:
        print(f"\nError: Not enough data points ({len(all_confidences)}). Need at least 100.")
        sys.exit(1)

    print(f"\nCollected {len(all_confidences)} predictions")

    # Fit calibrator
    calibrator = ConfidenceCalibrator(num_buckets=args.buckets)
    calibrator.fit(
        np.array(all_confidences),
        np.array(all_correct),
    )

    # Display calibration table
    print("\n" + calibrator.get_calibration_table())

    # Save calibrator
    output_path = args.output or "checkpoints/calibration.json"
    calibrator.save(output_path)
    print(f"\nCalibration saved to {output_path}")


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
    analyze_parser = subparsers.add_parser(
        "analyze", help="Analyze a single ticker"
    )
    analyze_parser.add_argument("ticker", help="Stock ticker symbol")
    analyze_parser.add_argument(
        "--min-confidence", type=float, default=None, dest="min_confidence",
        help="Minimum confidence %% to act on a signal (e.g. 60); below this shows HOLD",
    )
    analyze_parser.set_defaults(func=cmd_analyze)

    # scan command
    scan_parser = subparsers.add_parser(
        "scan", help="Scan multiple tickers for signals"
    )
    scan_parser.add_argument(
        "tickers", nargs="*", help="Ticker symbols (default: Swedish stocks)"
    )
    scan_parser.add_argument(
        "--min-confidence", type=float, default=None, dest="min_confidence",
        help="Minimum confidence %% to act on a signal (e.g. 60); below this shows HOLD",
    )
    scan_parser.set_defaults(func=cmd_scan)

    # train command
    train_parser = subparsers.add_parser("train", help="Train the signal model")
    train_parser.add_argument(
        "tickers", nargs="*", help="Ticker symbols for training data"
    )
    train_parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    train_parser.add_argument(
        "--batch-size", type=int, default=32, help="Training batch size"
    )
    train_parser.add_argument(
        "--walk-forward", action="store_true",
        help="Use walk-forward training with periodic retraining"
    )
    train_parser.add_argument(
        "--initial-days", type=int, default=500,
        help="Initial training window size in days (walk-forward only)"
    )
    train_parser.add_argument(
        "--window-size", type=int, default=60,
        help="Validation window size in days (walk-forward only)"
    )
    train_parser.add_argument(
        "--no-focal-loss", action="store_true",
        help="Disable focal loss (use standard cross-entropy)"
    )
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
        "--commission", type=float, default=0.001,
        help="One-way commission as decimal (default: 0.001 = 0.1%%)",
    )
    backtest_parser.set_defaults(func=cmd_backtest)

    # calibrate command
    calibrate_parser = subparsers.add_parser(
        "calibrate", help="Train confidence calibrator from backtest data"
    )
    calibrate_parser.add_argument(
        "tickers", nargs="*", help="Ticker symbols for calibration data"
    )
    calibrate_parser.add_argument(
        "--horizon", type=int, default=5,
        help="Prediction horizon to use for calibration (default: 5)"
    )
    calibrate_parser.add_argument(
        "--buckets", type=int, default=10,
        help="Number of calibration buckets (default: 10)"
    )
    calibrate_parser.add_argument(
        "--output", help="Output path for calibration file (default: checkpoints/calibration.json)"
    )
    calibrate_parser.set_defaults(func=cmd_calibrate)

    # Parse and execute
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
