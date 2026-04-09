---
name: explain-signal
description: Generate a signal for a ticker and explain what drove it. Shows which features are at unusual values (high z-score), what the confidence calibration is doing, and whether the ATR stop-loss is meaningful. Use to build trust in a signal or debug unexpected output.
allowed-tools: Read, Grep, Glob, Bash
---

# Explain Signal

Generate the current signal for the specified ticker and break down what is driving it.

**Required:** a trained model and calibrator must exist in `checkpoints/`. Run `/health-check` first if unsure.

---

## Step 1 — Generate the signal

```bash
uv run python main.py generate <TICKER>
```

Record: direction (BUY/SELL/HOLD), raw confidence, calibrated confidence, entry price, stop-loss, take-profit targets for each horizon.

---

## Step 2 — Load feature stats from config

```bash
python3 -c "
import json
c = json.load(open('checkpoints/signal_model_config.json'))
print('Feature columns:', len(c['feature_columns']))
print('Sequence length:', c['sequence_length'])
print('Holdout start:', c['holdout_start_date'])
"
```

---

## Step 3 — Fetch current feature values and compute z-scores

```bash
python3 - <<'EOF'
import json
import pandas as pd
import numpy as np
import sys

sys.argv = ['explain']  # prevent argparse from running

cfg = json.load(open('checkpoints/signal_model_config.json'))
cols = cfg['feature_columns']
mean = np.array(cfg['feature_mean'])
std  = np.array(cfg['feature_std'])

# Import the data pipeline
from data.fetcher import DataFetcher
from data.features import FeatureEngineer

ticker = sys.argv[1] if len(sys.argv) > 1 else 'VOLV-B.ST'

fetcher = DataFetcher()
df_raw = fetcher.fetch(ticker, period='6mo')
fe = FeatureEngineer()
df = fe.compute(df_raw)

# Take the most recent row
row = df[cols].iloc[-1].values
z = (row - mean) / (std + 1e-8)

# Print top 10 features by absolute z-score
order = np.argsort(np.abs(z))[::-1][:10]
print(f"\nTop 10 features by z-score (most unusual values) for {ticker} on {df.index[-1].date()}:\n")
print(f"{'Feature':<35} {'Value':>10} {'Mean':>10} {'Z-score':>8}")
print('-' * 65)
for i in order:
    direction = '↑' if z[i] > 0 else '↓'
    print(f"{cols[i]:<35} {row[i]:>10.3f} {mean[i]:>10.3f} {z[i]:>7.2f}{direction}")
EOF
```

Replace `VOLV-B.ST` with the requested ticker.

---

## Step 4 — Interpret the output

Based on what you find, write a plain-language explanation of the signal covering:

**Signal summary:**
- Direction and calibrated confidence
- Whether confidence is above the default threshold (60%) — signals below this are filtered in live trading

**Feature drivers:**
- Top 3–5 features with |z-score| > 1.5 and what they mean in plain English
  - e.g. `rsi` = 82 (z=+2.1) → RSI is significantly overbought vs training average
  - e.g. `adx` = 31 (z=+1.8) → Strong trending conditions, above the ADX≥25 filter threshold
  - e.g. `macd_hist` = -0.4 (z=-2.3) → Momentum is unusually negative
- Whether the features telling a consistent story (all pointing same direction) or mixed

**ATR context:**
- ATR value and what it means for the stop-loss distance
  - ATR is stored as % of price (e.g. 2.1 = 2.1% of current price)
  - A stop-loss at 1.5× ATR means roughly X% below entry
- Whether the risk/reward ratio of entry vs stop vs take-profit is favourable (target ≥ 2× stop distance)

**Regime context:**
- ADX value and regime classification (trending if ADX ≥ 25, ranging otherwise)
- Whether the signal is in a favourable regime for this model (per STRATEGY.md, ADX ≥ 25 is where the model performs best)

**Caveats:**
- Flag any feature with |z-score| > 3 as out-of-distribution (the model may not have seen inputs like this)
- Note if the signal is HOLD — this usually means low confidence or unfavourable regime, not a strong sell indicator
- Remind the user of the SELL bias limitation from STRATEGY.md if the signal is SELL

---

## Step 5 — Output format

```
=== Signal Explanation: <TICKER> on <DATE> ===

Signal:     BUY / SELL / HOLD
Confidence: XX% raw → XX% calibrated  [above/below 60% threshold]
Entry:      XXX.XX SEK
Stop-loss:  XXX.XX SEK  (X.X% below entry, ~X.X× ATR)
Target:     XXX.XX SEK  (X.X% above entry, R/R = X.X)

Key drivers:
  1. <feature>: value=X, z=+X.X — plain explanation
  2. <feature>: value=X, z=+X.X — plain explanation
  3. <feature>: value=X, z=-X.X — plain explanation

Regime: TRENDING (ADX=XX) / RANGING (ADX=XX)
Story: consistent / mixed — one sentence summary

Caveats: (if any)
```
