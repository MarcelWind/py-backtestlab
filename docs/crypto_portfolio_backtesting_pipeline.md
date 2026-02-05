# Crypto Portfolio Backtesting Pipeline

This document outlines a full, statistically sound pipeline for backtesting and evaluating **multi-asset crypto portfolio strategies** with minimal user input and proper out-of-sample (OOS) testing.

---

## 0. Big Picture Architecture

```
Universe selection
        ↓
Data ingestion & cleaning
        ↓
Strategy definition (multi-asset)
        ↓
Parameter optimization (IS)
        ↓
Walk-forward / OOS evaluation
        ↓
Reporting & diagnostics
```

Each stage should be modular, deterministic, and reusable.

---

## 1. Coin Universe Selection

### Hard Constraints
For statistical validity, restrict the universe to assets with:
- **≥ 4–5 years of daily data**
- **High liquidity** (to reduce slippage and noise)
- **Still actively traded** (avoid dead or delisted coins)

### Practical Starting Universe
Start with a **static universe** to avoid selection bias:
- Top **10–20 coins by market cap**
- Exclude stablecoins
- Exclude wrapped assets (e.g., WBTC)

Example universe:
```
BTC, ETH
BNB, ADA, SOL, XRP
LTC, BCH
DOT, LINK
AVAX, MATIC
```

> Important: Fix the universe *before* testing strategies to avoid leaking alpha via selection.

Later improvements may include dynamic universes (e.g., rolling top-N by market cap).

---

## 2. Data Sourcing & Downloading

### Recommended Data Sources
- **Binance API (spot)** – long history, high liquidity
- **CCXT** – exchange-agnostic, good for long-term maintenance

Avoid:
- CoinGecko OHLCV (rate-limited, inconsistent)
- Yahoo Crypto (spotty data quality)

### Data Storage Layout
Use a single canonical format:

```
data/
  BTCUSDT/
    1d.parquet
  ETHUSDT/
    1d.parquet
```

Each file contains:
- timestamp (UTC)
- open, high, low, close
- volume

### Data Hygiene Checklist
You must:
- Align timestamps across all assets
- Handle missing candles (drop globally or forward-fill explicitly)
- Normalize everything to UTC
- Ensure no lookahead bias (no partial candles)

> This step causes most bugs. Do it once, correctly.

---

## 3. Multi-Asset Strategies with `backtesting.py`

### Core Limitation
`backtesting.py` is **single-asset by design**:
- One instrument
- One position at a time

### Recommended Approach: Meta-Portfolio Wrapper
Instead of forcing portfolio logic into the engine:

1. Backtest each asset independently
2. Extract daily returns per asset
3. Combine returns at the portfolio level

```
portfolio_return = Σ (weight_i × return_i)
```

Benefits:
- Clean separation of concerns
- Easier optimization
- Statistically sound

This mirrors how most academic and professional crypto research is done.

### Strategy Abstraction
Define strategies in a reusable specification:

```python
class StrategySpec:
    name: str
    param_space: dict
    strategy_class: Strategy
```

Each backtest run should output:
- Daily returns
- Position sizes
- Turnover

---

## 4. Parameter Optimization

### Optimization Rule
**Optimize on portfolio-level metrics**, not per-asset metrics.

Bad:
- Maximizing Sharpe per coin

Good:
- Maximizing portfolio Sharpe
- Penalizing drawdown and turnover

### Optimization Methods
Start simple:
- Grid search
- Random search

Later extensions:
- Bayesian optimization (Optuna)
- Evolutionary methods (CMA-ES)

### Example Objective Function
```python
score = sharpe - 0.1 * max_drawdown - 0.01 * turnover
```

Keep objectives explicit and interpretable.

---

## 5. Out-of-Sample (OOS) Testing

### Walk-Forward Validation
Example schedule:
```
2018–2020 → optimize (IS)
2021       → test (OOS)
2022–2023 → re-optimize (IS)
2024       → test (OOS)
```

This process should be fully automated.

### What to Store Per Run
- Strategy parameters
- In-sample performance
- Out-of-sample performance
- Parameter stability metrics

> Red flag: strong IS performance with poor OOS results indicates overfitting.

---

## 6. Minimal User-Input Pipeline

### User-Facing API Example
```python
run_backtest(
    universe=["BTC", "ETH", "SOL"],
    strategy=MomentumStrategy,
    param_space={
        "lookback": (20, 200),
        "threshold": (0.0, 0.05)
    },
    timeframe="1d"
)
```

### Fully Automated Internals
- Data loading & alignment
- Parameter optimization
- Walk-forward testing
- Portfolio aggregation
- Performance reporting

The user supplies *what* to test — the system handles *how*.

---

## 7. Project Naming

Avoid novelty names — this is infrastructure.

Strong candidates:
- **AlphaPipeline**
- **QuantForge**
- **StratLab**
- **EdgeLab**
- **AlphaValidator**

Python-style package names:
- `alphapipe`
- `quantpipe`
- `stratlab`

---

## 8. Final Notes

- `backtesting.py` is a good starting point
- You may eventually outgrow it for fully vectorized or event-driven engines
- This architecture ports cleanly to NumPy-based or Zipline-style systems

The key is **modularity, statistical discipline, and automation**.

