# GLM-5.3-Flash EXL3 on 2x DGX Spark (MTP-best recipe)

MIT recipe serving `turboderp/GLM-5.3-Flash-exl3` (4.05bpw) with vLLM TP=2 and
native MTP3 speculative decoding plus packed decode sidecars.

## Images

- `alcoholmajuu/glm53-mtp-spark:exl3v15` — clean build (`45a7e480…`).
  Verified to 128k context.
- `alcoholmajuu/glm53-mtp-spark:exl3v15-topkfix` (`0267fbe6…`) — same plus
  the upstream vLLM #54110 backport (`build/topk-54110-backport.diff`).
  Required for 512k–1M context; identical speed at short context.

Pull: `docker pull alcoholmajuu/glm53-mtp-spark:exl3v15-topkfix`
(manifest `sha256:677527b4…`, arm64, ~10.4GB compressed)

## Measured (single stream, 1024 output tokens)

| input | code | JSON | Japanese prose |
|---|---|---|---|
| 4k | 42.9–45.2 | 41.3–44.6 | 36.2–37.3 |
| 16k | 42.2–47.0 | 41.9–43.2 | 35.9–36.4 |

Same-config runs vary ±5–10%; ranges above are repeated measurements.

## Limits

- 512k needs the topkfix image (GB10 `persistent_topk` grid limit).
  Build it with `docker/Dockerfile.topkfix` (needs the patched `_C` .so;
  patch: `build/topk-54110-backport.diff`, upstream vLLM PR #54110).
- 1M needs `gpu-memory-utilization 0.89` (8.83 GiB KV) and leaves thin host
  headroom. Verified boot + short requests; long-prefill stress is ongoing.
- Agent use: launch with `--prefix-cache` (reuses conversation history
  across turns). Opencode example: `examples/opencode-vllm.jsonc`
  (tunnel `ssh -L 8000:127.0.0.1:8000 <rank0>` first, API binds loopback).

## Run

1. `cp .env.example .env` and fill in both nodes, sync this tree to both.
2. Weights: `Terra3312/GLM-5.3-Flash-EXL3-MTP` (converted, ready to serve)
   or regenerate from `turboderp/GLM-5.3-Flash-exl3@2a30229e` with
   `scripts/convert_nonexperts.py` plus `scripts/extract_packed_*.py`
   (hashes in `PINS.json`).
3. Launch: `scripts/launch_mtp.py --image <digest> --receipt receipts/launch.json --execute`.
4. Smoke: on-node `bench/run_stream.py --output /tmp/smoke` (API binds loopback).

## Licenses

Recipe code is MIT (`LICENSE`). `build/context/exl3.py` is Apache-2.0,
vLLM/FlashInfer are Apache-2.0 (not bundled). The topk backport is upstream
vLLM PR #54110 (Apache-2.0). See `THIRD_PARTY_NOTICES.md` and `PINS.json`.
Do not relabel Apache material as MIT.
