# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A CLI-based trading signal generator for Swedish stocks and indexes (e.g., OMX Stockholm 30). Uses TensorFlow for ML-based predictions and yfinance for market data.

### Signal Output
- **Direction**: Buy / Sell / Hold
- **Confidence score**: Probability/confidence level (0-100%)
- **Price targets**: Entry and exit price predictions

## Architecture

```
finance/
├── main.py              # CLI entry point
├── data/
│   ├── fetcher.py       # yfinance data retrieval
│   └── features.py      # Technical indicators
├── models/
│   ├── signal_model.py  # TensorFlow model definition
│   └── training.py      # Training pipeline
└── signals/
    └── generator.py     # Signal generation logic
```

## Development Environment

- Python 3.13 (managed via `.python-version`)
- Package manager: uv (see `uv.lock`)
- Virtual environment: `.venv/`

## Commands

```bash
# Install dependencies
uv sync

# Run the application
uv run python main.py

# Add a new dependency
uv add <package>
```

## Dependencies

- **tensorflow**: Machine learning framework
- **yfinance**: Yahoo Finance market data downloader
