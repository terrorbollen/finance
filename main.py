#!/usr/bin/env python3
"""CLI entry point for the trading signal generator."""

import argparse
import sys

from data.fetcher import StockDataFetcher
from signals.generator import SignalGenerator
from models.training import ModelTrainer


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

    # Parse and execute
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
