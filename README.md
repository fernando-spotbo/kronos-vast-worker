# Kronos Vast Worker

A Vast.ai Serverless PyWorker that serves [Kronos-base](https://huggingface.co/NeoQuasar/Kronos-base)
volatility forecasts over HTTP.

## What it does

- Loads the `Kronos-base` time-series model on a GPU
- Exposes `POST /forecast` accepting recent OHLCV bars + future timestamps
- Returns the predicted high-low *range* over the future window (used as a
  volatility proxy)

## Files

| File | Purpose |
|---|---|
| `kronos_server.py` | Local Flask server (127.0.0.1:18000) wrapping the model |
| `worker.py` | Vast.ai PyWorker HTTP proxy that routes traffic to the model server |
| `start_server.sh` | Vast template startup: install deps, launch model server, start worker |
| `requirements.txt` | Python deps installed inside the worker container |

## Deploying on Vast.ai

1. Push this repo to a public GitHub URL.
2. Create a Vast template:
   ```bash
   vastai create template \
     --name kronos-vol-forecast \
     --image pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime \
     --env "-p 18000:18000 -e PYWORKER_REPO=https://github.com/<user>/kronos-vast-worker -e PYWORKER_DIR=/workspace/pyworker" \
     --onstart-cmd "wget -O - https://raw.githubusercontent.com/<user>/kronos-vast-worker/main/start_server.sh | bash" \
     --search_params "gpu_ram>=12 cpu_ghz>=4 pcie_bw>=15 num_gpus=1 rentable=true verified=true" \
     --disk_space 25.0 --ssh --direct
   ```
3. Create an endpoint:
   ```bash
   vastai create endpoint --endpoint_name kronos-fc \
       --target_util 0.9 --cold_workers 1 --max_workers 2
   ```
4. Create a workergroup linking the template + endpoint:
   ```bash
   vastai create workergroup --template_hash <hash from step 2> \
       --endpoint_name kronos-fc --cold_workers 1 --test_workers 1
   ```
5. The endpoint URL is `https://run.vast.ai/route/<endpoint-id>/forecast`.

## Calling the endpoint

```python
import requests
r = requests.post(
    "https://run.vast.ai/route/<endpoint-id>/forecast",
    headers={"Authorization": "Bearer <api-key>"},
    json={
        "history": [{"timestamps": "...", "open": ..., ...}, ...],   # ~80-100 bars
        "future_timestamps": ["...", ...],                            # 30 bars
        "sample_count": 10,
    },
    timeout=30,
)
print(r.json()["vol_range"])
```
