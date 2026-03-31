"""Walk-forward validation and outsample testing."""

from .bar_permute import get_permutation, permute_event_bars
from .mcpt import (
    EventLevelMCPTStrategy,
    MCPTResult,
    MCPTStrategy,
    SubmarketMCPT,
    aggregate_market_returns,
    concat_returns_in_order,
    make_scoring_fn,
    next_log_returns,
    profit_factor,
    run_insample_mcpt,
    run_oos_mcpt,
)
from .partition import EventPartition, compute_insample_outsample_event_counts, split_events_in_sample_out_of_sample
from .significance import empirical_one_tailed_pvalue, walk_forward_pvalue_from_event_order

__all__ = [
    "EventLevelMCPTStrategy",
    "EventPartition",
    "MCPTResult",
    "MCPTStrategy",
    "SubmarketMCPT",
    "aggregate_market_returns",
    "compute_insample_outsample_event_counts",
    "concat_returns_in_order",
    "empirical_one_tailed_pvalue",
    "get_permutation",
    "make_scoring_fn",
    "next_log_returns",
    "permute_event_bars",
    "profit_factor",
    "run_insample_mcpt",
    "run_oos_mcpt",
    "split_events_in_sample_out_of_sample",
    "walk_forward_pvalue_from_event_order",
]
