"""Fetch data.zip from a remote server and load all OHLCV data into a DataFrame."""

import io
import subprocess
import zipfile

import pandas as pd


def fetch_zip(host: str, remote_path: str, local_path: str = "data.zip", port: int | None = None):
    """SCP the data archive from a remote server."""
    cmd = ["scp"]
    if port is not None:
        cmd += ["-P", str(port)]
    cmd.append(f"{host}:{remote_path}")
    cmd.append(local_path)
    print(f"Fetching: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"Saved to {local_path}")


def load_zip(zip_path: str = "data.zip") -> pd.DataFrame:
    """Load all parquet files from a zip into a single DataFrame.

    Adds 'event_slug' and 'market' columns derived from the file paths.
    """
    frames = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            parts = name.replace("\\", "/").split("/")
            if len(parts) != 3 or not parts[2].endswith(".parquet") or parts[1] == "unknown":
                continue
            event_slug = parts[1]
            market = parts[2].removesuffix(".parquet")
            df = pd.read_parquet(io.BytesIO(zf.read(name)))
            df["event_slug"] = event_slug
            df["market"] = market
            frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No parquet data found in {zip_path}")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["event_slug", "market", "timestamp"]).reset_index(drop=True)
    return combined


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch data.zip from a remote server and inspect OHLCV data.")
    parser.add_argument("host", nargs="?", help="SCP host (e.g. user@server)")
    parser.add_argument("remote_path", nargs="?", help="Remote path to data.zip")
    parser.add_argument("--local-path", default="data.zip", help="Local destination (default: data.zip)")
    parser.add_argument("-P", "--port", type=int, default=None, help="SCP port (default: 22)")
    parser.add_argument("--local", action="store_true", help="Skip fetch, just load local data.zip")
    args = parser.parse_args()

    if not args.local:
        if not args.host or not args.remote_path:
            parser.error("host and remote_path are required unless --local is specified")
        fetch_zip(args.host, args.remote_path, args.local_path, port=args.port)

    df = load_zip(args.local_path)
    print(f"Loaded {len(df)} candles across {df['event_slug'].nunique()} events, "
          f"{df['market'].nunique()} markets")
    print(f"Time range: {df['datetime'].min()} -> {df['datetime'].max()}")
    print(f"Total volume: {df['volume'].sum():.2f} ({df['trade_count'].sum()} trades)")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nSample:\n{df.head()}")
