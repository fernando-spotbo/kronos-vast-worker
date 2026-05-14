"""Kronos vol-forecast Flask server.

Listens on localhost:18000. POST /forecast with a JSON body:
{
  "history": [                       # last LOOKBACK 1-min bars (OHLCV + amount)
    {"timestamps": "...", "open": ..., "high": ..., "low": ..., "close": ..., "volume": ..., "amount": ...},
    ...
  ],
  "future_timestamps": ["...", ...]  # PRED_LEN target timestamps
}

Returns:
{
  "vol_range": <float>,              # predicted high-low range over the future window
  "inference_ms": <int>,
  "n_history": <int>,
  "n_future": <int>
}

Health: GET /health -> {"status": "ready"} once Kronos has loaded.

Designed to run as the model-server behind a Vast.ai PyWorker.
"""
from __future__ import annotations

import os
import sys
import time
import json
import logging
import threading
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

# -- Kronos repo lives next to this file or under /workspace/kronos_src
KRONOS_CANDIDATES = [
    os.environ.get("KRONOS_DIR", ""),
    "/workspace/kronos_src",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "kronos_src"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Kronos"),
]
for c in KRONOS_CANDIDATES:
    if c and os.path.isdir(os.path.join(c, "model")):
        sys.path.insert(0, os.path.abspath(c))
        break
else:
    raise RuntimeError(f"Kronos model dir not found. Tried: {KRONOS_CANDIDATES}")

from model import Kronos, KronosTokenizer, KronosPredictor

PRICE_COLS = ["open", "high", "low", "close", "volume", "amount"]
DEFAULT_PRED_LEN  = int(os.environ.get("PRED_LEN", "30"))
DEFAULT_T         = float(os.environ.get("T_SAMPLE", "0.6"))
DEFAULT_TOP_P     = float(os.environ.get("TOP_P", "0.9"))
DEFAULT_SAMPLES   = int(os.environ.get("SAMPLE_CT", "10"))
DEVICE            = os.environ.get("DEVICE", "cuda:0")

PORT = int(os.environ.get("MODEL_SERVER_PORT", "18000"))
HOST = os.environ.get("MODEL_SERVER_HOST", "0.0.0.0")

LOG_FILE = os.environ.get("MODEL_LOG_FILE", "/var/log/model/server.log")

# -- logging: stderr + file (Vast PyWorker tails the file for readiness signals)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stderr),
        logging.FileHandler(LOG_FILE, mode="a"),
    ],
)
log = logging.getLogger("kronos_server")


# ---------- model load ----------

class _Predictor:
    instance: KronosPredictor | None = None
    ready: bool = False
    error: str | None = None
    lock: threading.Lock = threading.Lock()

    @classmethod
    def load(cls):
        with cls.lock:
            if cls.instance is not None:
                return
            try:
                log.info("loading Kronos tokenizer + base model...")
                t0 = time.time()
                tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
                mdl = Kronos.from_pretrained("NeoQuasar/Kronos-base")
                cls.instance = KronosPredictor(mdl, tok, device=DEVICE, max_context=512)
                cls.ready = True
                log.info(f"Application startup complete. (load took {time.time() - t0:.1f}s)")
            except Exception as e:
                cls.error = str(e)
                log.exception("Kronos load failed")


def _warmup_kronos_async():
    threading.Thread(target=_Predictor.load, daemon=True).start()


# ---------- Flask app ----------

app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    if _Predictor.error:
        return jsonify({"status": "error", "error": _Predictor.error}), 500
    if _Predictor.ready:
        return jsonify({"status": "ready"}), 200
    return jsonify({"status": "loading"}), 503


@app.route("/forecast", methods=["POST"])
def forecast():
    if not _Predictor.ready or _Predictor.instance is None:
        return jsonify({"error": "model not ready"}), 503
    try:
        data = request.get_json(force=True)
        hist = pd.DataFrame(data["history"])
        if "timestamps" not in hist.columns:
            return jsonify({"error": "history rows must include 'timestamps'"}), 400
        hist["timestamps"] = pd.to_datetime(hist["timestamps"], utc=True)
        for c in PRICE_COLS:
            if c not in hist.columns:
                return jsonify({"error": f"history rows must include '{c}'"}), 400
        fut_ts = pd.to_datetime(data["future_timestamps"], utc=True)
        if not hasattr(fut_ts, "__len__"):
            return jsonify({"error": "future_timestamps must be a list"}), 400
        pred_len = len(fut_ts)
        T        = float(data.get("T", DEFAULT_T))
        top_p    = float(data.get("top_p", DEFAULT_TOP_P))
        samples  = int(data.get("sample_count", DEFAULT_SAMPLES))

        t0 = time.time()
        pred = _Predictor.instance.predict(
            df=hist[PRICE_COLS].reset_index(drop=True),
            x_timestamp=hist["timestamps"].reset_index(drop=True),
            y_timestamp=pd.Series(fut_ts).reset_index(drop=True),
            pred_len=pred_len, T=T, top_p=top_p,
            sample_count=samples, verbose=False,
        )
        elapsed_ms = int((time.time() - t0) * 1000)
        vol_range = float(pred["high"].max() - pred["low"].min())
        return jsonify({
            "vol_range": vol_range,
            "inference_ms": elapsed_ms,
            "n_history": len(hist),
            "n_future": int(pred_len),
        })
    except Exception as e:
        log.exception("forecast error")
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "service": "kronos-vol-forecast",
        "model": "NeoQuasar/Kronos-base",
        "device": DEVICE,
        "endpoints": ["GET /health", "POST /forecast"],
        "ready": _Predictor.ready,
    })


# ---------- entrypoint ----------

if __name__ == "__main__":
    log.info(f"kronos_server starting on {HOST}:{PORT} device={DEVICE}")
    _warmup_kronos_async()
    # Flask's dev server is fine here: PyWorker is the public-facing proxy.
    app.run(host=HOST, port=PORT, threaded=True, use_reloader=False)
