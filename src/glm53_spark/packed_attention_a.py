"""MIT: original packed MLA q_a/kv_a, indexer wq_b and MTP eh_proj for short decode.

These layers are replicated (not TP-sharded), so every rank reads them in full on
each decode step. The original approved EXL3 tensors are loaded unchanged from an
exact-copy sidecar; prefill and larger batches keep the loaded BF16 weights.
"""
import os
import re
import torch
from safetensors import safe_open


def _inner(upstream, handle, stem, device):
    values = {}
    for suffix in ('trellis', 'suh', 'svh', 'mul1'):
        value = handle.get_tensor(stem+'.'+suffix)
        if value.ndim == 0:
            value = value.reshape(1)
        values[suffix] = value.contiguous().to(device)
    return upstream.make_linear_exl3(values['trellis'], values['suh'], values['svh'], values['mul1'])


def _packed_forward(inners, x):
    half = x.contiguous().half()
    outputs = [i.forward(half, {}, out_dtype=torch.float16).to(x.dtype) for i in inners]
    return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=-1)


def install(upstream):
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    cls = UnquantizedLinearMethod
    if getattr(cls, '_glm53_attention_a_installed', False):
        return
    path = os.environ['GLM53_PACKED_ATTENTION_A']
    previous_process = cls.process_weights_after_loading
    previous_apply = cls.apply
    pattern = re.compile(r'^(?:language_model\.)?model\.layers\.(\d+)\.self_attn\.(fused_qkv_a_proj|indexer\.wq_b)$')

    def process(self, layer):
        previous_process(self, layer)
        prefix = getattr(layer, 'prefix', '')
        match = pattern.fullmatch(prefix)
        if not match:
            return
        index = int(match[1])
        if index < 45 and not prefix.startswith('language_model.'):
            return
        stem = f'model.language_model.layers.{index}.self_attn.'
        stems = [stem+'q_a_proj', stem+'kv_a_proj_with_mqa'] if match[2] == 'fused_qkv_a_proj' else [stem+'indexer.wq_b']
        # ReplicatedLinear keeps tp_size=world size but stores the full matrix;
        # the exact full-shape match below proves the weight is unsharded.
        with safe_open(path, framework='pt', device='cpu') as handle:
            inners = [_inner(upstream, handle, s, layer.weight.device) for s in stems]
        for inner in inners:
            assert inner.in_features == layer.weight.shape[1], (prefix, inner.in_features)
        assert sum(i.out_features for i in inners) == layer.weight.shape[0], prefix
        layer._glm53_attention_a_inners = inners
        print(f'GLM53 original packed replicated decode: {prefix}; rows={layer.weight.shape[0]}', flush=True)

    def apply(self, layer, x, bias=None):
        inners = getattr(layer, '_glm53_attention_a_inners', None)
        if inners is None or x.numel()//x.shape[-1] > 8:
            return previous_apply(self, layer, x, bias)
        result = _packed_forward(inners, x)
        return result if bias is None else result+bias

    cls.process_weights_after_loading = process
    cls.apply = apply
    cls._glm53_attention_a_installed = True

    # MTP eh_proj is a plain nn.Linear; attach the packed matrix after MTP weights load.
    from vllm.models.glm5next.nvidia import mtp
    load = mtp.Glm5NextMTP.load_weights

    def load_weights(self, weights):
        loaded = load(self, weights)
        for name, module in self.named_modules():
            if not name.endswith('eh_proj') or not isinstance(module, torch.nn.Linear):
                continue
            index = re.search(r'layers\.(\d+)\.eh_proj$', name)
            assert index and int(index[1]) >= 45, name
            with safe_open(path, framework='pt', device='cpu') as handle:
                inner = _inner(upstream, handle, f'model.language_model.layers.{index[1]}.eh_proj', module.weight.device)
            assert (inner.in_features, inner.out_features) == (module.in_features, module.out_features), name
            original = module.forward

            def forward(x, _inner=inner, _original=original):
                if x.numel()//x.shape[-1] > 8:
                    return _original(x)
                return _packed_forward([_inner], x)

            module.forward = forward
            print(f'GLM53 original packed MTP eh_proj decode: {name}', flush=True)
        return loaded

    mtp.Glm5NextMTP.load_weights = load_weights
