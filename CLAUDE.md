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

# Install with dev dependencies
uv sync --all-extras

# Run the application
uv run python main.py

# Add a new dependency
uv add <package>

# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=. --cov-report=html

# Lint with ruff
uv run ruff check .

# Format with ruff
uv run ruff format .

# Type check with mypy
uv run mypy .

# Start MLflow tracking server (Docker)
docker-compose up -d

# View MLflow UI
# Open http://localhost:5000 in browser
```

## Dependencies

- **tensorflow**: Machine learning framework
- **yfinance**: Yahoo Finance market data downloader
- **mlflow**: Experiment tracking and model registry

### Dev Dependencies

- **pytest**: Testing framework
- **pytest-cov**: Code coverage
- **ruff**: Linting and formatting
- **mypy**: Type checking
- **pandas-stubs**: Type stubs for pandas
