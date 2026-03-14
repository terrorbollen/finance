# Future Improvements

Track potential improvements for the trading signal generator.

## Confidence Calibration (Implemented)

- [x] Implement `ConfidenceCalibrator` using isotonic regression
- [x] Integrate calibrator into `SignalGenerator`
- [x] Add `calibrate` CLI command to train from backtest data
- [x] Display both raw and calibrated confidence in signal output
- [ ] **Confidence threshold filtering**: Only trade above X% calibrated confidence
- [ ] **Per-direction calibration**: Separate calibrators for BUY/SELL/HOLD

## Model Improvements

- [ ] **Multi-horizon model**: Train separate output heads for 1-7 day predictions instead of using a single 5-day trained model for all horizons
- [ ] **Ensemble models**: Combine multiple model predictions (LSTM + GRU + Transformer) for more robust signals
- [ ] **Walk-forward optimization**: Retrain model periodically during backtest to simulate realistic conditions
- [ ] **Hyperparameter tuning**: Systematic search for optimal learning rate, hidden units, dropout rates
- [ ] **Cross-validation strategies**: Time-series cross-validation with multiple train/val/test splits

## Risk Management

- [ ] **Dynamic stop-loss based on ATR**: Use Average True Range (already calculated) to set volatility-adjusted stops (e.g., `stop = price - 2*ATR`)
- [ ] **Position sizing based on confidence**: Higher confidence = larger position (Kelly criterion or fixed-fractional)
- [ ] **Portfolio-level risk limits**: Maximum drawdown limits, sector exposure limits
- [ ] **Dynamic take-profit targets**: Use ATR multiples instead of fixed percentage
- [ ] **Trailing stops**: Based on recent highs/lows

## Feature Engineering

- [ ] **Sentiment analysis**: Incorporate news sentiment from financial news APIs
- [ ] **Market regime indicators**: Add features for bull/bear market detection (ADX, volatility regime)
- [ ] **Sector correlation**: Include sector ETF movements as features
- [ ] **Volatility features**: Add VIX or VSTOXX correlation
- [ ] **Calendar effects**: Day of week, month effects, earnings season indicators
- [ ] **Cross-asset features**: Stock correlation with OMXS30, currency impact (USD/SEK, EUR/SEK)
- [ ] **Macro indicators**: RIKSBANK interest rates, oil prices (Brent crude)

## Signal Generation Logic

- [ ] **Signal filtering**: Don't trade during earnings, avoid low-volume periods, filter low-confidence
- [ ] **Market regime awareness**: Different thresholds for trending vs ranging markets
- [ ] **Multi-timeframe confirmation**: Daily signal confirmed by weekly trend

## Backtesting Enhancements

- [ ] **Transaction costs**: Include realistic trading costs (~0.05-0.1% per trade for Swedish stocks)
- [ ] **Slippage modeling**: Volume-based slippage estimation
- [ ] **Position sizing**: Implement Kelly criterion or other position sizing methods
- [ ] **Risk-adjusted metrics**: Add Sharpe ratio, Sortino ratio, max drawdown, Calmar ratio
- [ ] **Benchmark comparison**: Compare against multiple benchmarks (index, sector ETF)
- [ ] **Monte Carlo simulation**: Estimate confidence intervals for backtest results

## Execution Strategy

- [ ] **Entry timing**: Limit orders vs market orders, wait for pullback?
- [ ] **Partial exits**: Scale out as target approaches
- [ ] **Time-based exits**: Exit if signal hasn't worked within N days
- [ ] **Order management**: Order types (stop-limit, OCO), position averaging

## Infrastructure

- [ ] **Model versioning**: Track model versions and compare performance over time
- [ ] **Automated retraining**: Schedule periodic model retraining with new data
- [ ] **Real-time alerts**: Set up notifications for high-confidence signals
- [ ] **Dashboard**: Web interface for monitoring signals and backtest results
- [ ] **Logging & monitoring**: Prediction tracking, performance dashboards

## Data Sources

- [ ] **Alternative data**: Consider alternative data sources (social media, insider trading from FI)
- [ ] **Multi-exchange**: Support additional exchanges beyond Swedish stocks
- [ ] **Intraday data**: Support for intraday predictions with minute/hourly data

---

## Priority Order (Suggested)

1. ~~Confidence calibration~~ - Done! Run `uv run python main.py calibrate`
2. **Risk management** - Dynamic stop-loss and position sizing
3. **Backtesting realism** - Add transaction costs
4. **Feature enhancements** - Cross-asset features and regime detection
5. **Signal filtering** - Avoid low-quality setups
6. **Execution strategy** - Better entry/exit timing
