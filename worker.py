"""Vast.ai PyWorker for the Kronos vol-forecast model server.

The model server (kronos_server.py) listens on 127.0.0.1:18000 and exposes:
  - GET  /health    -> {"status": "ready" | "loading" | "error"}
  - POST /forecast  -> {"vol_range": <float>, ...}

This file uses vastai-sdk's Worker/WorkerConfig/HandlerConfig to:
  - Detect readiness by tailing the model log for "Application startup complete."
  - Route incoming requests to the local model server
  - Compute per-request workload (constant: one Kronos forecast)
  - Run a small benchmark so the autoscaler can estimate throughput
"""
import os
from pathlib import Path

from vastai import (
    Worker,
    WorkerConfig,
    HandlerConfig,
    BenchmarkConfig,
    LogActionConfig,
)

MODEL_LOG = os.environ.get("MODEL_LOG_FILE", "/var/log/model/server.log")
PORT      = int(os.environ.get("MODEL_SERVER_PORT", "18000"))
HOST      = os.environ.get("MODEL_SERVER_URL", "http://127.0.0.1")


def _benchmark_payload():
    """Build a small but real Kronos request to benchmark the worker.

    We send 90 fake 1-min bars and ask for a 30-bar forecast — close to
    actual production traffic shape.
    """
    import datetime as _dt
    import json
    base = _dt.datetime(2026, 5, 14, 13, 30, tzinfo=_dt.timezone.utc)
    hist = []
    for i in range(90):
        ts = (base + _dt.timedelta(minutes=i)).isoformat()
        hist.append({
            "timestamps": ts,
            "open": 100.0 + 0.01 * i,
            "high": 100.05 + 0.01 * i,
            "low":  99.95 + 0.01 * i,
            "close":100.02 + 0.01 * i,
            "volume":1000.0,
            "amount":100020.0,
        })
    fut = [(base + _dt.timedelta(minutes=90 + i)).isoformat() for i in range(30)]
    return {"history": hist, "future_timestamps": fut, "sample_count": 5}


worker_config = WorkerConfig(
    model_server_url=HOST,
    model_server_port=PORT,
    model_log_file=MODEL_LOG,
    handlers=[
        HandlerConfig(
            route="/forecast",
            allow_parallel_requests=False,  # Kronos predict() is GPU-bound; one at a time
            max_queue_time=120.0,           # generous: cold-start can take ~30s
            workload_calculator=lambda payload: 1.0,  # one forecast = one unit
            benchmark_config=BenchmarkConfig(
                generator=_benchmark_payload,
                runs=3,
                concurrency=1,
            ),
        ),
        HandlerConfig(
            route="/health",
            allow_parallel_requests=True,
            max_queue_time=10.0,
            workload_calculator=lambda payload: 0.0,  # health checks are free
        ),
    ],
    log_action_config=LogActionConfig(
        on_load=["Application startup complete."],
        on_error=[
            "Traceback (most recent call last):",
            "RuntimeError:",
            "CUDA error:",
            "Kronos load failed",
        ],
        on_info=['"message":"Download'],
    ),
)


if __name__ == "__main__":
    Worker(worker_config).run()
