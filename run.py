"""Generic backtest entrypoint utilities.

Use this file as a strategy-agnostic launcher. Concrete strategy workflows
should live in dedicated runner modules (for example under `strategies/`).
"""

import argparse
import importlib
from typing import Any

import pandas as pd

from stratlab.backtest.backtester import Backtester


def run_backtest(strategy: Any, prices: pd.DataFrame, rebalance_freq: int = 1) -> dict[str, Any]:
    """Run a strategy on a price matrix and return full backtester output."""
    backtester = Backtester(strategy=strategy, rebalance_freq=rebalance_freq)
    return backtester.run(prices)


def summarize_backtest(result: dict[str, Any]) -> pd.DataFrame:
    """Convert backtester metrics into a one-row DataFrame."""
    metrics = result.get("metrics", {})
    returns = result.get("returns", pd.Series(dtype=float))
    return pd.DataFrame(
        [
            {
                "n_returns": int(len(returns)),
                "mean_bar_return": float(returns.mean()) if len(returns) else 0.0,
                **metrics,
            }
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generic backtest launcher")
    parser.add_argument(
        "--runner",
        default="",
        help="Optional module to execute, e.g. 'strategies.run_weather_imbalance_smoke'",
    )
    args = parser.parse_args()

    if not args.runner:
        print("No runner selected.")
        print("Example:")
        print("  python run.py --runner strategies.run_weather_imbalance_smoke")
        return

    module = importlib.import_module(args.runner)
    if not hasattr(module, "main"):
        raise AttributeError(f"Runner module {args.runner!r} has no main()")

    module.main()


if __name__ == "__main__":
    main()
