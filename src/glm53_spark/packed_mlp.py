"""MIT: use original mul1 dense/shared MLP matrices for short decode shapes.

The regular BF16 weight remains the prefill path. No weights are requantized.
INT8 GEMV dispatch is supplied by the pinned ExLlamaV3 library for eligible rows.
"""
import os
from pathlib import Path
import re
import torch
from safetensors import safe_open


def install(upstream):
    if getattr(upstream, '_glm53_packed_mlp_installed', False):
        return
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    path = Path(os.environ['GLM53_PACKED_MLP'])
    assert path.is_file(), path
    original = upstream.Exl3Config.get_quant_method

    class PackedMLP(UnquantizedLinearMethod):
        def __init__(self, prefix):
            self.prefix = prefix

        def process_weights_after_loading(self, layer):
            super().process_weights_after_loading(layer)
            match = re.search(r'layers\.(\d+)\.mlp\.(shared_experts\.)?(gate_up_proj|down_proj)$', self.prefix)
            assert match
            stem = 'model.language_model.layers.' + match[1] + '.mlp.' + (match[2] or '')
            names = ['gate_proj', 'up_proj'] if match[3] == 'gate_up_proj' else ['down_proj']
            shard = upstream.shard_exl3_col if len(names) == 2 else upstream.shard_exl3_row
            inners = []
            source = path if int(match[1]) < 45 else Path(os.environ['GLM53_PACKED_MTP'])
            with safe_open(str(source), framework='pt', device='cpu') as f:
                for name in names:
                    tensors = {}
                    for suffix in ('trellis', 'suh', 'svh', 'mul1'):
                        value = f.get_tensor(stem + name + '.' + suffix)
                        if value.ndim == 0:
                            value = value.reshape(1)
                        value = shard(value, suffix, layer.tp_rank, layer.tp_size)
                        tensors[suffix] = value.contiguous().to(layer.weight.device)
                    inner = upstream.make_linear_exl3(tensors['trellis'], tensors['suh'],
                                                     tensors['svh'], tensors['mul1'])
                    assert inner.in_features == layer.weight.shape[1]
                    inners.append(inner)
            assert sum(i.out_features for i in inners) == layer.weight.shape[0]
            layer._glm53_packed_mlp = inners
            print('GLM53 original mul1 MLP decode enabled: ' + self.prefix, flush=True)

        def apply(self, layer, x, bias=None):
            if x.numel() // x.shape[-1] > 8:
                return super().apply(layer, x, bias)
            packed_input = x.contiguous().half()
            ys = [i.forward(packed_input, {}, out_dtype=torch.float16)
                  for i in layer._glm53_packed_mlp]
            output = (ys[0] if len(ys) == 1 else torch.cat(ys, dim=-1)).to(x.dtype)
            return output if bias is None else output + bias

    def method(self, layer, prefix):
        match = re.search(r'layers\.(\d+)\.mlp\.(?:shared_experts\.)?(?:gate_up_proj|down_proj)$', prefix)
        # The sidecar covers target layers 0..44. MTP layer 45 retains its
        # native weights rather than looking up a nonexistent sidecar entry.
        if match and (int(match[1]) < 45 or os.environ.get('GLM53_PACKED_MTP')):
            return PackedMLP(prefix)
        return original(self, layer, prefix)

    upstream.Exl3Config.get_quant_method = method
    upstream._glm53_packed_mlp_installed = True
