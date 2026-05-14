"""Smoke test for the kronos forecast endpoint.

Sends a real 90-bar history + 30-bar future request and validates the response.
Usage: python test_client.py [URL]   # default http://127.0.0.1:18000
"""
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests


URL = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("KRONOS_URL", "http://127.0.0.1:18000")


def real_bars(symbol="TQQQ", n_hist=90, n_fut=30):
    """Pull real 1-min bars from our local backtest cache and build a payload."""
    csv = "C:/Users/ferna/OneDrive/Escritorio/keys/kronos/backtest/data/windows/TQQQ_Apr26.csv"
    if not os.path.exists(csv):
        print(f"!!  no bars CSV at {csv} — falling back to synthetic"); return synthetic(n_hist, n_fut)
    df = pd.read_csv(csv)
    df["timestamps"] = pd.to_datetime(df["timestamps"], utc=True)
    hist = df.iloc[-(n_hist + n_fut + 1) : -(n_fut + 1)].copy()
    fut  = df.iloc[-(n_fut + 1) : -1]["timestamps"].dt.strftime("%Y-%m-%dT%H:%M:%SZ").tolist()
    hist_rows = [
        {
            "timestamps": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "open": float(o), "high": float(h), "low": float(l),
            "close": float(c), "volume": float(v), "amount": float(a),
        }
        for ts, o, h, l, c, v, a in zip(
            hist["timestamps"], hist["open"], hist["high"], hist["low"],
            hist["close"], hist["volume"], hist["amount"],
        )
    ]
    return {"history": hist_rows, "future_timestamps": fut, "sample_count": 10}


def synthetic(n_hist=90, n_fut=30):
    base = datetime(2026, 5, 14, 13, 30, tzinfo=timezone.utc)
    hist = [{
        "timestamps": (base + timedelta(minutes=i)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "open": 100.0 + 0.01 * i, "high": 100.05 + 0.01 * i,
        "low":  99.95 + 0.01 * i, "close":100.02 + 0.01 * i,
        "volume": 1000.0, "amount": 100020.0,
    } for i in range(n_hist)]
    fut = [(base + timedelta(minutes=n_hist + i)).strftime("%Y-%m-%dT%H:%M:%SZ")
           for i in range(n_fut)]
    return {"history": hist, "future_timestamps": fut, "sample_count": 10}


def main():
    print(f"target: {URL}")
    r0 = requests.get(f"{URL}/health", timeout=10)
    print(f"GET /health -> {r0.status_code} {r0.text}")
    if r0.status_code != 200:
        sys.exit(1)

    payload = real_bars()
    print(f"\nposting /forecast with {len(payload['history'])} hist + "
          f"{len(payload['future_timestamps'])} future bars...")
    t0 = time.time()
    r = requests.post(f"{URL}/forecast", json=payload, timeout=60)
    wall_ms = int((time.time() - t0) * 1000)
    print(f"POST /forecast -> {r.status_code}  wall_ms={wall_ms}")
    print(f"body: {r.text[:400]}")
    if r.status_code != 200:
        sys.exit(1)
    data = r.json()
    print(f"\nvol_range: ${data['vol_range']:.4f}")
    print(f"inference_ms: {data['inference_ms']}")
    print(f"wall vs inference: {wall_ms}ms vs {data['inference_ms']}ms")


if __name__ == "__main__":
    main()
