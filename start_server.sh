#!/bin/bash
# Vast.ai serverless template startup: fetch Kronos sources, install deps,
# launch the kronos_server.py model server, then start the PyWorker.
#
# Triggered by the template's onstart-cmd:
#   wget -O - https://raw.githubusercontent.com/<USER>/kronos-vast-worker/main/start_server.sh | bash

set -euo pipefail

LOGFILE=/var/log/start_server.sh.log
exec > >(tee -a "$LOGFILE") 2>&1
echo "[start_server] $(date -u +'%FT%TZ') starting"

# Vast workers usually have /workspace pre-mounted
cd /workspace

# 1. Pull our worker code (this repo) - contains the Kronos `model/` Python package
#    bundled as `kronos_model/`, plus the Flask server, PyWorker, and start script.
WORKER_DIR="${PYWORKER_DIR:-/workspace/pyworker}"
WORKER_REPO="${PYWORKER_REPO:-https://github.com/fernando-spotbo/kronos-vast-worker.git}"
if [ ! -f "$WORKER_DIR/kronos_server.py" ]; then
    git clone --depth 1 "$WORKER_REPO" "$WORKER_DIR" || { echo "clone $WORKER_REPO failed"; exit 1; }
fi

# 2. The Kronos `model` python package lives inside our repo under `kronos_model/`.
#    Make it importable as `model` by symlinking.
if [ -d "$WORKER_DIR/kronos_model" ] && [ ! -e /workspace/kronos_src ]; then
    mkdir -p /workspace/kronos_src
    ln -sf "$WORKER_DIR/kronos_model" /workspace/kronos_src/model
fi

# 3. Install Python deps. Most are already in the pytorch image; this is fast.
pip install --quiet --no-cache-dir \
    pandas einops 'huggingface_hub==0.33.1' safetensors flask requests vastai \
    || { echo "pip install failed"; exit 1; }

cd "$WORKER_DIR"

# 4. Pre-warm the Kronos model cache while server starts (saves first-call latency)
export KRONOS_DIR=/workspace/kronos_src
export MODEL_SERVER_PORT=18000
export MODEL_LOG_FILE=/var/log/model/server.log
mkdir -p /var/log/model

echo "[start_server] launching kronos_server.py..."
nohup python kronos_server.py >> /var/log/model/server.stdout 2>&1 &
SERVER_PID=$!
echo "[start_server] kronos_server.py pid=$SERVER_PID"

# 5. Wait for the model server to be ready before starting PyWorker.
# PyWorker also tails the log for "Application startup complete.", but a
# pre-check here is more robust.
for i in $(seq 1 120); do
    if curl -fs http://127.0.0.1:${MODEL_SERVER_PORT}/health 2>/dev/null | grep -q ready; then
        echo "[start_server] model server ready after ${i}s"
        break
    fi
    sleep 1
done

# 6. Run the PyWorker (foreground, this becomes pid 1 of the container)
echo "[start_server] launching PyWorker..."
exec python worker.py
