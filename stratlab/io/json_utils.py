"""JSON parsing helper utilities."""

from __future__ import annotations


def json_error_context(text: str, lineno: int, radius: int = 2) -> str:
    """Build a compact line-context snippet for JSON decode errors."""
    lines = text.splitlines()
    if not lines:
        return ""
    idx = max(0, lineno - 1)
    start = max(0, idx - radius)
    end = min(len(lines), idx + radius + 1)
    snippet = []
    for i in range(start, end):
        prefix = ">" if i == idx else " "
        snippet.append(f"{prefix} {i + 1:4d}: {lines[i]}")
    return "\n".join(snippet)
