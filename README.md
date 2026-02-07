# backtestlab

A lightweight crypto portfolio backtesting pipeline focused on reproducible, multi-asset strategy evaluation, parameter optimization, walk-forward testing, and reporting.

See the full pipeline design in docs/crypto_portfolio_backtesting_pipeline.md.

## Features

- Multi-asset portfolio backtesting and reporting
- Monte Carlo / random-search parameter optimization (using `optimize`) 
- Walk-forward / OOS evaluation support
- Exported metrics and plots saved under `results/`

## Quickstart (Windows)

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # PowerShell
# or
.\.venv\Scripts\activate.bat   # CMD
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the main simulation (this runs optimization and backtests, then saves results):

```powershell
python run.py
```

The script will create a timestamped folder under `results/` (e.g. `results/20260205_222826/`) containing `metrics.csv` and PNG plots.

## Project layout

- `run.py` — example entrypoint: loads prices, runs optimization and backtests, saves plots/metrics.
- `stratlab/` — main package
  - `backtest/` — backtester and engine
  - `data/` — data loading, cleaning, storage
  - `strategy/` — strategy implementations (e.g., `MomentumStrategy`, `SharpeStrategy`, `BuyAndHoldStrategy`)
  - `optimize/` — optimization utilities
  - `report/` — plotting and metrics
  - `universe/` — asset selector
  - `validation/` — walk-forward evaluation
- `data/` — per-asset OHLCV data folders (e.g., `BTCUSDT/`)
- `results/` — output from runs
- `docs/` — design notes and pipeline doc

## Data

Place per-symbol OHLCV files under `data/<SYMBOL>/` in the canonical format used by the data loaders (see `stratlab/data/cleaner.py`). The project expects preprocessed daily data aligned across symbols.

## Running & customizing

- To change the tested universe or strategies, edit `stratlab/universe/selector.py` and the strategy classes in `stratlab/strategy/`.
- `run.py` demonstrates a full flow: load prices, run `optimize`, then run optimized and benchmark backtests and save outputs.

## Results

Each run creates a timestamped folder under `results/` that contains:

- `metrics.csv` — comparison metrics for strategies
- `backtest.png`, `comparison.png`, `distribution.png`, `correlation.png` — diagnostic plots

## Development notes

- See `docs/crypto_portfolio_backtesting_pipeline.md` for design decisions and recommended workflows.
- Dependencies are listed in `requirements.txt`.

## Contributing

Contributions are welcome. Open issues or submit PRs. Keep changes focused and add tests or examples where possible.

## License

See `LICENCE` at the project root.
