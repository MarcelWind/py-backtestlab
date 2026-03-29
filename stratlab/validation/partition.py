"""Reusable event partition helpers for in-sample/out-of-sample validation."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class EventPartition:
    """Container for deterministic event-level insample/outsample splits."""

    method: str
    seed: int
    n_events_total: int
    n_events_in_sample: int
    n_events_out_of_sample: int
    in_sample_ratio: float
    out_of_sample_ratio: float
    in_sample_event_slugs: list[str]
    out_of_sample_event_slugs: list[str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def compute_insample_outsample_event_counts(n_events: int, out_of_sample_ratio: float = 0.4) -> tuple[int, int]:
    """Compute (insample_count, outsample_count) with outsample using ceil(out_of_sample_ratio * n)."""
    if n_events < 2:
        raise ValueError("Need at least 2 events to create both in-sample and out-of-sample partitions")
    if not (0.0 < out_of_sample_ratio < 1.0):
        raise ValueError("out_of_sample_ratio must be between 0 and 1")

    outsample_count = int(math.ceil(out_of_sample_ratio * n_events))
    outsample_count = max(1, min(outsample_count, n_events - 1))
    insample_count = int(n_events - outsample_count)
    return insample_count, outsample_count


def split_events_in_sample_out_of_sample(
    event_slugs: list[str],
    *,
    out_of_sample_ratio: float = 0.4,
    seed: int = 42,
) -> EventPartition:
    """Split events into insample/outsample partitions once using deterministic random sampling."""
    insample_count, outsample_count = compute_insample_outsample_event_counts(
        n_events=len(event_slugs),
        out_of_sample_ratio=out_of_sample_ratio,
    )

    rng = np.random.default_rng(seed)
    permuted = rng.permutation(np.array(event_slugs, dtype=object)).tolist()
    oos_slugs = [str(s) for s in permuted[:outsample_count]]
    is_slugs = [str(s) for s in permuted[outsample_count : outsample_count + insample_count]]

    return EventPartition(
        method="random_event_split",
        seed=int(seed),
        n_events_total=len(event_slugs),
        n_events_in_sample=is_count,
        n_events_out_of_sample=oos_count,
        in_sample_ratio=float(is_count / len(event_slugs)),
        out_of_sample_ratio=float(oos_count / len(event_slugs)),
        in_sample_event_slugs=is_slugs,
        out_of_sample_event_slugs=oos_slugs,
    )
