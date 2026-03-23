from __future__ import annotations

import gzip
import hashlib
import json
import pickle
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SNAPSHOT_CACHE_SCHEMA_VERSION = 1


def _normalize_for_json(value: Any) -> Any:
    """Normalize values into a deterministic JSON-serializable shape."""
    if isinstance(value, dict):
        return {str(k): _normalize_for_json(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple, set)):
        return [_normalize_for_json(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def stable_json_dumps(payload: Any) -> str:
    return json.dumps(_normalize_for_json(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def hash_payload(payload: Any) -> str:
    return hashlib.sha256(stable_json_dumps(payload).encode("utf-8")).hexdigest()


def compute_file_fingerprint(file_path: Path) -> str:
    """Hash file contents to invalidate cache when indicator logic changes."""
    if not file_path.exists():
        return "missing"
    return hashlib.sha256(file_path.read_bytes()).hexdigest()


def build_cache_key_material(
    *,
    event_slug: str,
    resample_rule: str,
    prefer_outcome: str,
    rebalance_freq: int,
    profile: str,
    base_params_hash: str,
    indicator_names_hash: str,
    indicator_code_fingerprint: str,
    source_data_fingerprint: str,
) -> dict[str, Any]:
    return {
        "schema_version": SNAPSHOT_CACHE_SCHEMA_VERSION,
        "event_slug": str(event_slug),
        "resample_rule": str(resample_rule),
        "prefer_outcome": str(prefer_outcome),
        "rebalance_freq": int(rebalance_freq),
        "profile": str(profile),
        "base_params_hash": str(base_params_hash),
        "indicator_names_hash": str(indicator_names_hash),
        "indicator_code_fingerprint": str(indicator_code_fingerprint),
        "source_data_fingerprint": str(source_data_fingerprint),
    }


def cache_file_path(cache_dir: Path, event_slug: str, cache_key: str) -> Path:
    safe_slug = event_slug.strip().replace("/", "_")
    return cache_dir / safe_slug / f"{cache_key}.pkl.gz"


def clear_cache(cache_dir: Path) -> None:
    if cache_dir.exists():
        shutil.rmtree(cache_dir)


def load_indicator_snapshots(
    *,
    cache_dir: Path,
    event_slug: str,
    cache_key: str,
) -> dict[str, dict] | None:
    path = cache_file_path(cache_dir, event_slug, cache_key)
    if not path.exists():
        return None
    try:
        with gzip.open(path, "rb") as f:
            payload = pickle.load(f)
    except (OSError, pickle.PickleError, EOFError):
        return None

    if not isinstance(payload, dict):
        return None
    meta = payload.get("meta")
    snapshots = payload.get("snapshots")
    if not isinstance(meta, dict) or not isinstance(snapshots, dict):
        return None
    if int(meta.get("schema_version", -1)) != SNAPSHOT_CACHE_SCHEMA_VERSION:
        return None
    if str(meta.get("cache_key", "")) != cache_key:
        return None

    out: dict[str, dict] = {}
    for name, snap in snapshots.items():
        if isinstance(name, str) and isinstance(snap, dict):
            out[name] = snap
    return out if out else None


def save_indicator_snapshots(
    *,
    cache_dir: Path,
    event_slug: str,
    cache_key: str,
    meta: dict[str, Any],
    snapshots: dict[str, dict],
) -> Path:
    path = cache_file_path(cache_dir, event_slug, cache_key)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "meta": {
            **meta,
            "schema_version": SNAPSHOT_CACHE_SCHEMA_VERSION,
            "cache_key": cache_key,
            "created_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        },
        "snapshots": snapshots,
    }

    tmp_path = Path(f"{path}.tmp")
    with gzip.open(tmp_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    if not tmp_path.exists():
        raise OSError(f"Temporary cache file was not created: {tmp_path}")
    tmp_path.replace(path)
    return path
