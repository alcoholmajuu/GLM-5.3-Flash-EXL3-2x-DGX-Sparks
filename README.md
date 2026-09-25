# GLM-5.3-Flash EXL3 on 2x DGX Spark (MTP-best recipe)

MIT recipe serving `turboderp/GLM-5.3-Flash-exl3` (4.05bpw) with vLLM TP=2 and
native MTP3 speculative decoding plus packed decode sidecars.

Measured (single stream, 1024 output tokens): 16k input code 46.1 / JSON 47.8 /
Japanese prose 37.8 tok/s. Decode holds to 64k (44.4 / 40.7 / 39.2); TTFT ~80s
at 64k dominates end-to-end. 512k serving fails at engine start
(`persistent_topk` oversubscription in FULL graph profiling); 128k starts.

## Requirements

- Two NVIDIA DGX Spark (128GB unified) with IB/RoCE between them, Docker.
- ~175GB for weights: converted BF16 checkpoint (164GB) + packed sidecars (~6GB).
- Weights: `<HF repo>` (converted, ready to serve) or regenerate from
  `turboderp/GLM-5.3-Flash-exl3@2a30229e` with `scripts/convert_nonexperts.py`
  plus `scripts/extract_packed_*.py` (hashes in `PINS.json`).

## Run

1. `cp .env.example .env` and fill in both nodes, sync this tree to both.
2. Build: `docker build -f docker/Dockerfile.phase1 ...` then
   `docker build -f docker/Dockerfile.mtp ...` (rebuild, then re-smoke).
3. Convert + packed sidecars (or download), verify with
   `scripts/finalize_checkpoint.py`.
4. Launch: `scripts/launch_mtp.py --image <digest> --receipt receipts/launch.json --execute`.
5. Smoke: `bench/run_stream.py --output /tmp/smoke`.

## Licenses

Recipe code is MIT (`LICENSE`). `build/context/exl3.py` is Apache-2.0,
vLLM/FlashInfer are Apache-2.0 (not bundled). See `THIRD_PARTY_NOTICES.md`
and `PINS.json`. Do not relabel Apache material as MIT.
