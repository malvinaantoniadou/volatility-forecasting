# Predicting short-term volatility spikes: S&P 500, then FX

Can past returns and volatility flag unusually volatile weeks better than a simple 20-day-volatility rule? This project tests that on the S&P 500 and then on three currency pairs.

A "spike" is a 5-day-ahead realised volatility above the 75th percentile of the training data.

## Summary of findings
- **S&P 500:** a five-feature logistic regression beat the 20-day rule in all four walk-forward folds (mean ROC-AUC 0.815 vs 0.773) and on a later test period (0.711 vs 0.611; 95% block-bootstrap interval for the difference +0.04 to +0.17).
- The three-feature version did not beat the rule, so the gain comes from the 5-day return and the short/long volatility ratio.
- Gradient boosting (test AUC 0.721) and GARCH(1,1) added nothing reliable on top of logistic regression.
- At a 0.25 cutoff the model flags 159 days and catches 63 of 122 spike days (recall 0.52, precision 0.40).
- **FX (GBP/USD, EUR/GBP, USD/JPY):** no model clearly beat the 20-day rule on any pair, so the equity result did not carry over.
- Predicted probabilities overstate risk in the test period, so the scores are useful for ranking, not as literal probabilities.

## Method
- Forward-looking target: the standard deviation of daily returns over the next five trading days, using only information available at each date's close.
- Features: 5-, 20- and 60-day rolling volatility, 5-day return, and the 5-day/60-day volatility ratio.
- Expanding-window validation (4 folds, 5-day gap) so overlapping labels do not leak between training and validation.
- The spike threshold is learned from training data only.
- Benchmarks: a 20-day-volatility rule with no fitted model, logistic regression (3 and 5 features), gradient boosting, and GARCH(1,1) as a ranking comparison.
- Differences in AUC on the test period are assessed with a moving-block bootstrap, which keeps the overlap between neighbouring daily labels.
- The notebook is a research record, so its narrative follows the order in which the experiments were run.

## Limitations
- The S&P 500 test period was inspected during development, so it is a chronological evaluation, not an untouched holdout.
- Daily labels overlap, so counts of spike days are not independent events.
- One label definition (top quartile) and one horizon (5 days).
- A threshold fixed on training data breaks when the volatility regime changes (for example, EUR/GBP's test period had only about 1.5% spike days).
- The 0.25 cutoff was chosen on S&P 500 data and does not transfer to FX.

## Next steps
Recalibrate probabilities, use rolling or regime-aware thresholds, and test a HAR-style model and longer horizons.

## Files
- `market_volatility_analysis.ipynb`: the full analysis, with a summary at the top and the benchmark and FX work in Sections 22 and 23.
- `requirements.txt`: Python packages used.

## Run it
1. `pip install -r requirements.txt`
2. Open `market_volatility_analysis.ipynb` and run all cells.

Data are downloaded from Yahoo Finance through `yfinance` when the notebook runs. Package versions are not pinned, so a future run may give slightly different numbers.
