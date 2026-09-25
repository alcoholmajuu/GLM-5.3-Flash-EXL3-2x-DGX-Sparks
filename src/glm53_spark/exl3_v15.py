"""MIT: ExLlamaV3 v1.5 ABI and decode MoE integration for the TP-only recipe."""
import torch

_scratch = {}


def install(upstream, cooperative_max_rows=2):
    assert cooperative_max_rows in (1,2,4,8)
    if getattr(upstream, '_glm53_v15_installed', False):
        return
    import exllamav3_ext as ext
    assert hasattr(ext, 'exl3_moe_coop')
    previous_apply = upstream.apply_exl3_experts

    def launch(fn, xh, out, counts, tokens, weights, temps, ptrs, k, flags, limit, active):
        fn(xh, out, counts, tokens, weights, *temps, upstream.MOE_ACT_SILU,
           k, k, k, *(ptrs[p + '_' + s] for p in ('gate', 'up', 'down')
                     for s in ('trellis', 'suh', 'svh')),
           *flags, float(limit), active if active is not None else -1,
           None, None, 1, temps[0].shape[1], 16)

    def apply(x, ids, weights, layer, *, limit=10.0, fused=None):
        rows, hidden = x.shape[-2:]
        # The existing bulk kernel remains the prefill path. This recipe uses
        # tensor parallelism; any expert-parallel placement stays on that path.
        if (fused is False or rows > cooperative_max_rows or getattr(layer, 'expert_map', None) is not None
                or not getattr(layer, '_exl3_ptrs', None)):
            return previous_apply(x, ids, weights, layer, limit=limit, fused=fused)
        flags = layer._exl3_codebook_flags
        assert flags == (False, True, False, True, False, True), flags
        intermediate = layer._exl3_intermediate_local
        key = (str(x.device), hidden, intermediate)
        if key not in _scratch:
            assert not torch.cuda.is_current_stream_capturing(), 'Warm up MoE before graph capture'
            # 256 slots allow four split-K slices for 8 tokens x top8.
            slots = 256
            def alloc(width, dtype):
                return torch.empty((slots, width), device=x.device, dtype=dtype)
            half = torch.float16
            buffers = (alloc(hidden, half), alloc(hidden, half),
                       alloc(intermediate, torch.float32), alloc(intermediate, torch.float32),
                       alloc(intermediate, half), alloc(hidden, torch.float32))
            counters = torch.zeros(slots * (intermediate // 128) + 8 * (hidden // 128)
                                   + 2 * slots + 3, device=x.device, dtype=torch.int32)
            output = torch.empty((8, hidden), device=x.device, dtype=torch.float32)
            _scratch[key] = (*buffers, counters, output)
            print(f'GLM53 EXL3 v1.5 cooperative MoE enabled: rows<={cooperative_max_rows}, mul1, TP-local '
                  f'H={hidden} I={intermediate}', flush=True)
        *buffers, counters, output = _scratch[key]
        ptrs = layer._exl3_ptrs
        k = layer._exl3_k
        ext.exl3_moe_coop(x.reshape(rows, hidden).half(), ids.reshape(rows, -1).long().contiguous(),
            weights.reshape(rows, -1).half().contiguous(), 0, len(layer._exl3_inners), hidden,
            *(ptrs[p + '_' + s] for p in ('gate', 'up', 'down') for s in ('trellis', 'suh', 'svh')),
            None, None, None, k, k, k, False, True, upstream.MOE_ACT_SILU, float(limit), True,
            *buffers, counters, output[:rows], None, None)
        layer._exl3_last_apply = 'v15_cooperative'
        return output[:rows].to(dtype=x.dtype)

    upstream._exl3_moe_launch = launch
    upstream.apply_exl3_experts = apply
    upstream._glm53_v15_installed = True
