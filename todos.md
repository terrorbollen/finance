# Future Improvements

Track potential improvements for the trading signal generator.

## Model Improvements

- [ ] **Multi-horizon model**: Train separate output heads for 1-7 day predictions instead of using a single 5-day trained model for all horizons
- [ ] **Ensemble models**: Combine multiple model predictions (LSTM + GRU + Transformer) for more robust signals
- [ ] **Walk-forward optimization**: Retrain model periodically during backtest to simulate realistic conditions
- [ ] **Hyperparameter tuning**: Systematic search for optimal learning rate, hidden units, dropout rates
- [ ] **Cross-validation strategies**: Time-series cross-validation with multiple train/val/test splits

## Feature Engineering

- [ ] **Sentiment analysis**: Incorporate news sentiment from financial news APIs
- [ ] **Market regime indicators**: Add features for bull/bear market detection
- [ ] **Sector correlation**: Include sector ETF movements as features
- [ ] **Volatility features**: Add VIX or local volatility index correlation
- [ ] **Calendar effects**: Day of week, month effects, earnings season indicators

## Backtesting Enhancements

- [ ] **Transaction costs**: Include realistic trading costs in return calculations
- [ ] **Position sizing**: Implement Kelly criterion or other position sizing methods
- [ ] **Risk-adjusted metrics**: Add Sharpe ratio, Sortino ratio, max drawdown
- [ ] **Benchmark comparison**: Compare against multiple benchmarks (index, sector ETF)
- [ ] **Monte Carlo simulation**: Estimate confidence intervals for backtest results

## Infrastructure

- [ ] **Model versioning**: Track model versions and compare performance over time
- [ ] **Automated retraining**: Schedule periodic model retraining with new data
- [ ] **Real-time alerts**: Set up notifications for high-confidence signals
- [ ] **Dashboard**: Web interface for monitoring signals and backtest results

## Data Sources

- [ ] **Alternative data**: Consider alternative data sources (social media, satellite data)
- [ ] **Multi-exchange**: Support additional exchanges beyond Swedish stocks
- [ ] **Intraday data**: Support for intraday predictions with minute/hourly data
