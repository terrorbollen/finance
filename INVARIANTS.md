# Invariants

Rules that must always hold. Violating any of these will silently degrade the system or produce misleading results. Read this before modifying any module in the pipeline.

---

## Normalization must use training statistics

**Rule:** When running inference or backtesting, features must be normalized using `feature_mean` and `feature_std` from `checkpoints/signal_model_config.json` — not statistics computed from the current data window.

**Why:** The model's weights were optimized against training-distribution inputs. Using current-window stats changes the input distribution and degrades accuracy. It also introduces a subtle form of lookahead because current stats depend on future data points within the window.

**Where it matters:** `signals/generator.py` (`generate()`), `backtesting/backtester.py`. Both have a fallback that computes stats from current data if the config is missing — this fallback is for development only and must not be relied on in production.

---

## ATR is expressed as percentage of price, not absolute value

**Rule:** The `atr` feature column produced by `FeatureEngineer` is stored as a percentage of the current price (e.g. `2.1` means 2.1%), not as an absolute SEK value.

**Why:** All features are scale-independent so the model generalises across stocks with different price levels. Stop-loss and take-profit calculations in `signals/generator.py` divide by 100 when converting ATR to a price distance.

**Where it matters:** Any code that reads `df["atr"]` and uses it to compute price levels.

---

## Feature columns must match between training and inference

**Rule:** The list of feature columns used at inference time must exactly match `feature_columns` in the model config. The model's `input_dim` is fixed at training time.

**Why:** A mismatch causes a shape error or — worse — silently feeds the wrong features if columns are reordered.

**How it's enforced:** `SignalGenerator` filters to only columns in the config and warns about missing ones, but a partial mismatch still degrades predictions. **Adding or removing features requires retraining and regenerating the config.**

---

## Signal class encoding is fixed: BUY=0, HOLD=1, SELL=2

**Rule:** The integer encoding of signal classes must not change. This mapping is hardcoded in `SignalModel`, `SignalGenerator`, and `Backtester`.

**Why:** Changing it in one place without changing the others swaps BUY and SELL signals throughout the system with no error raised.

---

## Data must not be shuffled

**Rule:** Time-series data must preserve temporal order throughout training and backtesting. Never pass `shuffle=True` to any split or fit operation.

**Why:** Shuffling leaks future information into past-labeled samples, producing inflated training metrics and a model that cannot generalise to live trading.

---

## Holdout period must not overlap training data

**Rule:** The backtest start date must be on or after `holdout_start_date` from the model config. Never pass `strict_holdout=False` in production.

**Why:** Evaluating on data the model has already seen makes backtest metrics meaningless. The strict holdout flag exists only for debugging purposes.

**How it's enforced:** `Backtester` checks `holdout_start_date` at construction time and raises if the requested start date violates it.

---

## Calibrator must be refitted after every retraining

**Rule:** Run `uv run python main.py calibrate` after every `train` run before using confidence scores for live signals. Auto-calibration runs automatically at the end of `train` unless `--no-calibrate` is passed.

**Why:** Calibration maps raw softmax probabilities to real accuracy rates observed in backtest data. After retraining the model, the probability distribution changes and the old calibrator no longer reflects reality.

## Calibration horizon must match the model's training prediction_horizon

**Rule:** The default calibration horizon is 5 bars (5 trading days), matching the model's `prediction_horizon`. Only override `--horizon` if you intentionally changed `prediction_horizon` during training.

**Why:** The `--horizon` in calibration is a number of *bars*. If it doesn't match what the model was trained to predict, the calibration is evaluated on the wrong outcome window and the calibration file is useless for live trading.

---

## Focal loss and sample weights are mutually exclusive

**Rule:** When `use_focal_loss=True`, do not pass sample weights to `model.fit()`. When using standard cross-entropy, use sample weights for class balancing instead.

**Why:** Focal loss handles class imbalance internally by down-weighting easy examples. Combining it with sample weights double-counts the imbalance correction and distorts training.

**Where it's enforced:** `models/training.py` — `fit_kwargs` omits `sample_weight` when focal loss is active.

---

## `signals/` and `backtesting/` must not import from each other

**Rule:** `signals/` must not import from `backtesting/`, and vice versa. Both depend on `data/` and `models/` but are independent of each other.

**Why:** These modules serve different purposes — one generates live signals, the other runs historical simulations. Coupling them creates circular dependency risk and makes it harder to test either in isolation.

---

## Reference data must share the stock DataFrame's DatetimeIndex

**Rule:** Cross-asset reference DataFrames passed to `FeatureEngineer` (OMXS30, USD/SEK, EUR/SEK) must have a DatetimeIndex that aligns with the stock DataFrame's index.

**Why:** Features are computed by direct index alignment. A mismatch introduces NaN values that propagate through all downstream calculations.

**How it's enforced:** `FeatureEngineer` uses `.reindex()` to align indexes, filling gaps with forward-fill. A large number of gaps is a sign the reference data is misaligned.
