#!/usr/bin/env python3
"""CLI entry point for the trading signal generator."""

import argparse
import sys
from datetime import datetime

from data.fetcher import StockDataFetcher
from signals.generator import SignalGenerator
from models.training import ModelTrainer
from backtesting import Backtester


def cmd_analyze(args):
    """Analyze a single ticker and generate a signal."""
    print(f"\nAnalyzing {args.ticker}...")

    generator = SignalGenerator()
    try:
        signal = generator.generate(args.ticker)
        print(signal)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def cmd_scan(args):
    """Scan multiple tickers and generate signals."""
    if args.tickers:
        tickers = args.tickers
    else:
        # Default: scan major Swedish stocks
        tickers = list(StockDataFetcher.list_swedish_tickers().keys())[:10]

    print(f"\nScanning {len(tickers)} tickers...")
    print("-" * 50)

    generator = SignalGenerator()
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
        # Default training tickers
        tickers = ["^OMX", "VOLV-B.ST", "ERIC-B.ST", "HM-B.ST", "SEB-A.ST"]

    print(f"\nTraining model on {len(tickers)} tickers...")
    print(f"Epochs: {args.epochs}")
    print("-" * 50)

    trainer = ModelTrainer()
    try:
        results = trainer.train(
            tickers=tickers,
            epochs=args.epochs,
            batch_size=args.batch_size,
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

    backtester = Backtester()
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
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # analyze command
    analyze_parser = subparsers.add_parser(
        "analyze", help="Analyze a single ticker"
    )
    analyze_parser.add_argument("ticker", help="Stock ticker symbol")
    analyze_parser.set_defaults(func=cmd_analyze)

    # scan command
    scan_parser = subparsers.add_parser(
        "scan", help="Scan multiple tickers for signals"
    )
    scan_parser.add_argument(
        "tickers", nargs="*", help="Ticker symbols (default: Swedish stocks)"
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
    backtest_parser.set_defaults(func=cmd_backtest)

    # Parse and execute
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
