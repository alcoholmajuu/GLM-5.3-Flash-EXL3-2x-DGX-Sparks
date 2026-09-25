# Third-party notices

Original recipe files are MIT. External materials retain their own licenses.
Do not describe Apache-licensed dependencies or their copied material as MIT.

| File | Source | Revision | License |
|---|---|---|---|
| `build/context/exl3.py` | gitcommit90/glm-5.3-one-spark `overlay/exl3.py` | `ddc28b11e10c0f40112506944b99fd1a79448379` | Apache-2.0 (file SPDX) |
| `build/context/patch_exl3_ext_aarch64.py` | gitcommit90/glm-5.3-one-spark | `ddc28b11e10c0f40112506944b99fd1a79448379` | MIT |
| `build/context/mla_0.py`, `mla_1.py` | MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks Dockerfile RUN blocks | `eb9bd4805605bd66653ed3c887db637e4d5980b6` | MIT |
| `build/context/LICENSES/mia.txt` | MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks LICENSE | `eb9bd4805605bd66653ed3c887db637e4d5980b6` | MIT |
| `build/context/LICENSES/one-spark.txt` | gitcommit90/glm-5.3-one-spark LICENSE | `ddc28b11e10c0f40112506944b99fd1a79448379` | MIT |
| `build/exllamav3-1.5.0.tar.gz` | turboderp-org/exllamav3 v1.5.0 | `0740edc2da569fb99174023c1d2988b1e98cb41e` | MIT |
| vLLM (base image + pinned APIs) | vllm-project/vllm | `487ecf187d3dfe74d2cf6119a92881dba403c219` | Apache-2.0 |
| Target weights | turboderp/GLM-5.3-Flash-exl3, branch `4.05bpw` | `2a30229e67012798ba9f0cd832bb78abf4c363d5` | MIT |
| Packed sidecars | Derived from the MIT target weights by `scripts/extract_packed_*.py` | see `PINS.json` | MIT |
