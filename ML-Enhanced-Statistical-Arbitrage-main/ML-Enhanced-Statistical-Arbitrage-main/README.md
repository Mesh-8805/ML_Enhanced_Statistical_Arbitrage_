# ML-Enhanced Statistical Arbitrage on NSE Stocks

Course project for MLEE (EEE G513): Machine Learning, BITS Pilani, First Semester 2026–27.
Team: Reyansh Verma, Mihir Godawat, Somesh Jaiswal. Instructor: Ritesh Vyas.

A walk-forward pairs-trading system on ~225 NSE stocks that trade in the
futures (F&O) segment, with daily data from 2014 to 2026. Machine learning does
four jobs that classical pairs trading does by hand. Every ML part is measured
against a classical baseline, after Indian futures costs, on data the model
never saw:

| Classical step | ML replacement | Notebook |
|---|---|---|
| Test every pair (brute force) | PCA embeddings + k-means / GMM (EM) / OPTICS clusters | Part 3.4 |
| Static OLS hedge ratio | Kalman-filter hedge ratio (β drifts daily) | Part 4.1 |
| Take every z-score signal | Meta-labeling classifier (logistic, GDA, naive Bayes, SVM, random forest, LightGBM, MLP) | Part 7 |
| OU forecast of reversion | TCN forecaster | Part 8 |

All the code is in one notebook, `statarb_ml_simple.ipynb`, in reading order,
with an explanation before each part and the results saved inside it.

## Run it

Python 3.12:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/jupyter lab statarb_ml_simple.ipynb
```

(On Windows use `.venv\Scripts\pip` and `.venv\Scripts\jupyter`.) Then choose
*Run All*. The first run takes about 20 minutes on a laptop CPU, because it
runs every backtest and trains the TCN in every window. It saves those results
in `simple_outputs/cache/`, so later runs take a few minutes. A saved result is
only reused if it was made with exactly the same settings.

Every setting is in one dictionary, `config`, in Part 1.2 of the notebook.
Seeds are fixed.

**Data.** `data/raw/` holds the daily prices used for every number below:
adjusted closes and volumes from Yahoo Finance, plus NIFTY 50 and India VIX.
If that folder is missing, the notebook downloads the prices again into
`simple_outputs/raw/`. A new download can give slightly different results,
because Yahoo re-adjusts past prices whenever a company pays a dividend.

## Repository layout

```
statarb_ml_simple.ipynb   all the code, explanations and results
requirements.txt          Python packages
data/raw/                 daily prices (close, volume, NIFTY 50, India VIX)
simple_outputs/tables/    result tables (CSV) written by the notebook
simple_outputs/figures/   figures written by the notebook
simple_outputs/cache/     saved backtests, made on the first run (not in the repo)
```

## What the notebook does

1. **Setup:** libraries, settings and small helpers.
2. **Data and trading costs:** the stock universe, download, cleaning
   (at most 2 days forward-filled, one-day moves above 25% flagged) and the
   Zerodha cost model.
3. **Finding pairs:** OU fit, Engle–Granger test with Benjamini–Hochberg
   false-discovery control, distance and random baselines, PCA + clustering.
4. **The spread and the trading rule:** static, rolling and Kalman hedge
   ratios; enter at |z| > 2, exit at 0, stop at |z| > 4, time stop.
5. **The backtest engine:** 23 walk-forward windows (learn 252 days, trade
   the next 126), next-day execution, costs and rolls.
6. **Experiments:** classical baseline, sanity tests, pair selectors, hedge
   ratios.
7. **Meta-labeling:** seven classifiers with purged, embargoed cross-validation.
8. **Deep learning:** TCN against zero-change, AR(1) and OU forecasts, and a
   cost-hurdle trading test.
9. **Final evaluation:** full system against benchmarks, ablation,
   sensitivity, block bootstrap, deflated Sharpe ratio, error analysis.
10. **Conclusions.**

## How look-ahead bias is avoided

- Pairs, β, μ, σ, the Kalman δ and R, cluster counts and every model are fitted
  on the formation window only; the trading window is never used for tuning.
- Signals use the close of day t; orders fill at the close of day t+1.
- The meta-labeler for window w is trained only on trades from earlier windows.
  Its settings and threshold come from purged, embargoed cross-validation
  inside that training data.
- The TCN trains on formation-window samples. Its validation split is
  time-ordered with a gap, and no date is in both training and test.
- Checks: one extra day of delay, and shuffled trading-window returns.

## Cost model

Zerodha's published F&O rates, with STT at 0.05% on the sell side (the rate
from 1 April 2026, applied to the whole backtest), brokerage of 0.03% or ₹20,
NSE transaction charge, stamp duty, GST and SEBI fee, plus 4 bps slippage per
side. One leg of ₹10 lakh costs ₹612.7 in charges (6.13 bps). With slippage,
one pair round trip costs 28.3 bps of one leg's value. Holding a position
through a monthly expiry adds a roll (one buy and one sell per leg). Setting
`cost_multiplier` to 2.0 gives the double-cost check.

## Results

All numbers are out of sample and after costs. The 23 walk-forward windows run
from 2015 to November 2026. Tables comparing the final system start in
February 2018, the first window in which the meta-labeler has enough past
trades to learn from.

**Bottom line: the ML layers clearly improve on the classical baseline, but
the full system earns about zero after costs, and its Sharpe ratio cannot be
told apart from luck.**

### Headline (Feb 2018 – Nov 2026)

| Strategy | Net Sharpe | Gross Sharpe | Max drawdown | Trades | Hit rate | Break-even costs |
|---|---|---|---|---|---|---|
| Full ML system (k-means pairs, Kalman, meta-labeler, TCN gate) | 0.06 | 0.38 | −4.9% | 499 | 56.5% | 1.2× today's |
| Classical baseline (brute force + FDR, static OLS) | −0.29 | 0.09 | −11.9% | 816 | 46.0% | 0.24× |
| Random pairs (null) | −0.70 | −0.40 | −18.9% | 532 | 47.2% | — |
| NIFTY 50 buy and hold (context) | 0.59 | — | −38.4% | — | — | — |

- 95% block-bootstrap interval for the full system's Sharpe: [−0.62, 0.77].
- Deflated Sharpe ratio after the 50 strategy variants tried in the notebook:
  **0.03**. Luck alone would be expected to produce a Sharpe of about 0.69.
- Correlation with NIFTY: 0.19.

### RQ1: pair discovery (static hedge, all windows)

| Selector | Tests / window | Still cointegrated in trading window | Sharpe | Max DD |
|---|---|---|---|---|
| Brute force + FDR | 19,883 | 7.6% | −0.20 | −11.9% |
| Same sector | 1,466 | 8.3% | −0.56 | −15.8% |
| Distance (Gatev) | — | 7.0% | −0.47 | −11.5% |
| PCA + k-means | 2,362 | 10.2% | −0.16 | −11.1% |
| PCA + GMM (EM) | 7,075 | 7.8% | −0.46 | −15.1% |
| PCA + OPTICS | 146 | 10.9% | 0.10 | −3.5% (137 pairs only) |
| Random pairs | — | 8.0% | −0.61 | −22.5% |

- k-means cuts the number of tests by 88% and raises the share of pairs that
  stay cointegrated from 7.6% to 10.2%.
- The answer is mixed: in the full-system ablation, brute force + FDR scored
  higher (0.37 vs 0.06).
- Benjamini–Hochberg at 5% leaves almost nothing: 180 survivors out of 457,303
  tests, and none in 15 of the 23 windows. 7.4% of pairs pass the raw
  p < 0.05 test, close to the 5% expected by chance. So pairs are ranked by
  their corrected p-value and need a raw p < 0.05, instead of having to
  survive the correction.

### RQ2: meta-labeling (label = profit after costs > 0)

- Out-of-sample AUC is 0.50–0.56, so the classifiers have almost no skill;
  logistic regression scores 0.537.
- Even so, the logistic filter with sizing by p lifts Sharpe from −0.16 to
  0.06 and cuts the max drawdown from −5.6% to −4.9%. In the ablation,
  removing the meta-labeler is the costliest single change.
- The "spread reached its mean" label is nearly constant with a Kalman hedge:
  93% of trades reach z = 0. LightGBM's AUC of 0.62 on that label brought no
  economic gain.

### RQ3: hedge ratio (k-means pairs, no meta-labeler)

| Hedge | Sharpe | Max DD | Turnover / yr |
|---|---|---|---|
| Static OLS | −0.16 | −11.1% | 10.0× |
| Rolling 60-day OLS | −0.69 | −26.2% | 11.9× |
| Kalman filter | 0.00 | −6.6% | 7.7× |

Kalman beats static OLS on every column. The 60-day rolling regression is the
worst: 60 days is too short, so its β is noisy.

### RQ4: deep learning (forecast of the 5-day z-score change)

| Forecaster | RMSE | RMSE when \|z\| > 2 |
|---|---|---|
| Zero change | 1.156 | 3.102 |
| AR(1) | **0.820** | **1.262** |
| OU | 0.941 | 2.032 |
| TCN | 0.846 | 1.427 |

- The TCN beats the OU and zero-change forecasts, but a plain AR(1)
  regression beats it.
- The trading test ("take a trade only if the forecast reversion beats the
  cost") never blocks a trade, so removing the TCN gate changes nothing.

### Ablation (Sharpe; full system = 0.06)

| Removed | Sharpe | Max DD |
|---|---|---|
| − clustering (brute force + FDR) | 0.37 | −1.5% |
| − Kalman (static OLS) | −0.01 | −5.5% |
| − regime gate | not built | |
| − meta-labeler | −0.16 | −5.6% |
| − DL forecaster | 0.06 | −4.9% |

### Sensitivity (Sharpe: baseline → ML system without the TCN gate)

| Setting | Baseline | ML system |
|---|---|---|
| Base | −0.29 | 0.06 |
| Costs × 2 | −0.68 | −0.16 |
| Entry 1.5 / 2.5 | −0.70 / −0.50 | 0.12 / −0.01 |
| Formation 126 / 504 days | −0.96 / −0.58 | −0.05 / 0.05 |
| Top 10 / 40 pairs | 0.12 / −0.47 | 0.57 / −0.16 |

The ML system beats the baseline in all 8 settings. Its own Sharpe stays near
zero and turns negative at 2× costs.

### Checks and error analysis

- **Delay test:** one extra day of delay moved the baseline from −0.20 to
  −0.11. It did not fall, because the baseline has no edge for a delay to
  destroy.
- **Shuffle test:** shuffled spreads earned −0.06 (about zero minus costs).
- **Classical baseline losers:** 76% are relationship breaks (stop-loss or
  time stop).
- **Full system losers:** 58% are hedge drift (z returns to 0 because the
  Kalman filter adapts, not because the prices converge), 25% market shocks,
  5% company events.
- **Worst year:** 2020 (148 trades, 14 stop-outs).
- **Sectors:** banks and auto pairs lose; energy, financials and IT pairs make
  money.

## Limitations

- **Survivorship bias:** the universe is today's F&O list, not a point-in-time
  list.
- **Futures proxy:** spot prices stand in for futures prices; the roll is
  modelled as a cost only.
- **Not built:** the HMM regime gate (so the ablation's "− regime gate" row is
  empty) and the plan's other extensions.
- **Speed shortcut:** the Engle–Granger test uses a fixed lag of 1, so that
  ~20,000 pairs per window run in seconds.
- **Johansen's test** is computed for the selected pairs as a check only; it is
  not used as a filter.
