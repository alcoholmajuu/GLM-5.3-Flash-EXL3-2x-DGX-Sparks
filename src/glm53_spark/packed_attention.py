"""MIT: original packed attention projections for short decode, BF16 prefill.

The target's KDA merged matrix contains three TP-sharded large projections plus
small mixed-sharding outputs. Preserve the already-loaded small tail verbatim.
This module does not change attention or KV-cache kernels.
"""
import os
import re
import torch
import torch.nn.functional as F
from safetensors import safe_open


def install(upstream):
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    cls=UnquantizedLinearMethod
    if getattr(cls,'_glm53_attention_installed',False):return
    path=os.environ['GLM53_PACKED_ATTENTION']
    original_process=cls.process_weights_after_loading
    original_apply=cls.apply
    pattern=re.compile(r'^(?:language_model\.)?model\.layers\.(\d+)\.self_attn\.(o_proj|q_b_proj|in_proj_qkvbfg_a)$')

    def process(self,layer):
        original_process(self,layer)
        match=pattern.fullmatch(getattr(layer,'prefix',''))
        if not match:return
        layer_index=int(match[1]);kind=match[2]
        source=path
        if layer_index < 45 and not layer.prefix.startswith('language_model.'):
            return  # Unprefixed names are accepted only for the MTP layer.
        if layer_index >= 45:
            # MTP layer 45 uses a separate exact-copy sidecar only when explicitly enabled.
            source=os.environ.get('GLM53_PACKED_MTP')
            if not source or kind=='in_proj_qkvbfg_a':
                return
        stem=f'model.language_model.layers.{layer_index}.self_attn.'
        stem+='qkv_proj' if kind=='in_proj_qkvbfg_a' else kind
        inners=[]
        count=3 if kind=='in_proj_qkvbfg_a' else 1
        with safe_open(source,framework='pt',device='cpu') as handle:
            for part in range(count):
                values={}
                for suffix in ('trellis','suh','svh','mul1'):
                    value=handle.get_tensor(stem+'.'+suffix)
                    if value.ndim==0:value=value.reshape(1)
                    shard=upstream.shard_exl3_row if kind=='o_proj' else upstream.shard_exl3_col
                    # Q, K and V are contiguous equal-sized output sections.
                    rank=part*layer.tp_size+layer.tp_rank
                    value=shard(value,suffix,rank,layer.tp_size*count)
                    values[suffix]=value.contiguous().to(layer.weight.device)
                inner=upstream.make_linear_exl3(values['trellis'],values['suh'],values['svh'],values['mul1'])
                assert inner.in_features==layer.weight.shape[1]
                inners.append(inner)
        rows=sum(i.out_features for i in inners)
        if count==3:
            assert len({i.out_features for i in inners})==1
            assert 0<layer.weight.shape[0]-rows<1024
        else:
            assert rows==layer.weight.shape[0]
        layer._glm53_attention_inners=inners
        layer._glm53_attention_tail=rows
        print(f'GLM53 original packed attention decode: {layer.prefix}; packed_rows={rows}, native_tail={layer.weight.shape[0]-rows}',flush=True)

    def apply(self,layer,x,bias=None):
        inners=getattr(layer,'_glm53_attention_inners',None)
        if inners is None or x.numel()//x.shape[-1]>8:
            return original_apply(self,layer,x,bias)
        half=x.contiguous().half()
        outputs=[i.forward(half,{},out_dtype=torch.float16).to(x.dtype) for i in inners]
        tail=layer._glm53_attention_tail
        if tail<layer.weight.shape[0]:
            outputs.append(F.linear(x,layer.weight[tail:]))
        result=outputs[0] if len(outputs)==1 else torch.cat(outputs,dim=-1)
        return result if bias is None else result+bias

    cls.process_weights_after_loading=process
    cls.apply=apply
    cls._glm53_attention_installed=True
