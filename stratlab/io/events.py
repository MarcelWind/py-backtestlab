"""Event list loading helpers."""

from __future__ import annotations

import json
from pathlib import Path


def load_event_slugs_from_file(events_file: Path) -> list[str]:
    """Load event slugs from a JSON file.

    Supported formats:
    - ["slug-a", "slug-b"]
    - {"event_slugs": ["slug-a", "slug-b"]}
    - {"events": ["slug-a", {"slug": "slug-b"}, {"event_slug": "slug-c"}]}
    """
    try:
        raw = json.loads(events_file.read_text())
    except FileNotFoundError as exc:
        raise ValueError(f"Events file not found: {events_file}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Events file is not valid JSON: {events_file} ({exc})") from exc

    if isinstance(raw, list):
        candidates = raw
    elif isinstance(raw, dict):
        if "event_slugs" in raw:
            candidates = raw["event_slugs"]
        elif "events" in raw:
            candidates = raw["events"]
        else:
            raise ValueError(
                f"Invalid events file {events_file}: expected top-level list, 'event_slugs', or 'events'"
            )
    else:
        raise ValueError(f"Invalid events file {events_file}: expected JSON list or object")

    if not isinstance(candidates, list):
        raise ValueError(f"Invalid events file {events_file}: events payload must be a list")

    slugs: list[str] = []
    for idx, item in enumerate(candidates):
        if isinstance(item, str):
            slug = item.strip()
        elif isinstance(item, dict):
            raw_slug = item.get("event_slug", item.get("slug", ""))
            slug = str(raw_slug).strip()
        else:
            raise ValueError(f"Invalid event entry at index {idx} in {events_file}: expected string or object")

        if not slug:
            raise ValueError(f"Invalid event entry at index {idx} in {events_file}: empty slug")
        slugs.append(slug)

    return slugs
