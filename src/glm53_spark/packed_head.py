"""MIT: original approved mul1 lm_head for decode, native BF16 for prefill."""
import os
import torch
from safetensors import safe_open


def install(upstream):
    if getattr(upstream, '_glm53_packed_head_installed', False): return
    from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod
    original = upstream.Exl3Config.get_quant_method

    class PackedHead(UnquantizedEmbeddingMethod):
        def process_weights_after_loading(self, layer):
            super().process_weights_after_loading(layer)
            tensors = {}
            with safe_open(os.environ['GLM53_PACKED_HEAD'], framework='pt', device='cpu') as f:
                for suffix in ('trellis', 'suh', 'svh', 'mul1'):
                    value = f.get_tensor('lm_head.'+suffix)
                    if value.ndim == 0: value = value.reshape(1)
                    tensors[suffix] = upstream.shard_exl3_col(
                        value, suffix, layer.tp_rank, layer.tp_size).contiguous().to(layer.weight.device)
            inner = upstream.make_linear_exl3(tensors['trellis'],tensors['suh'],tensors['svh'],tensors['mul1'])
            assert tuple(layer.weight.shape) == (inner.out_features,inner.in_features), 'Unsupported padded vocabulary'
            layer._glm53_packed_head = inner
            print(f'GLM53 original mul1 lm_head active: rank={layer.tp_rank}, rows<=8', flush=True)

        def apply(self, layer, x, bias=None):
            if x.numel()//x.shape[-1] > 8:
                return super().apply(layer,x,bias)
            output = layer._glm53_packed_head.forward(x.contiguous().half(),{},out_dtype=torch.float16).to(x.dtype)
            return output if bias is None else output+bias

    def method(self,layer,prefix):
        if prefix in ('lm_head','language_model.lm_head'): return PackedHead()
        return original(self,layer,prefix)

    upstream.Exl3Config.get_quant_method = method
    upstream._glm53_packed_head_installed = True
