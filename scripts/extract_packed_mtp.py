"""MIT: copy the original packed MTP layer-45 attention and shared-expert tensors, no requantization.

Source is the approved turboderp 4.05bpw mtp.safetensors. Only o_proj, q_b_proj and the
shared-expert gate/up/down (trellis/suh/svh/mul1) are copied, for decode-only packed paths.
"""
import argparse
import hashlib
import json
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file

p = argparse.ArgumentParser()
p.add_argument('source', type=Path)
p.add_argument('output', type=Path)
a = p.parse_args()
assert not a.output.exists()
source = a.source/'mtp.safetensors'
with source.open('rb') as f:
    source_sha = hashlib.file_digest(f, 'sha256').hexdigest()
stems = ['model.language_model.layers.45.self_attn.o_proj', 'model.language_model.layers.45.self_attn.q_b_proj']
stems += ['model.language_model.layers.45.mlp.shared_experts.'+n for n in ('gate_proj','up_proj','down_proj')]
payload = {}
with safe_open(str(source), framework='pt', device='cpu') as f:
    for stem in stems:
        for suffix in ('trellis','suh','svh','mul1'):
            payload[stem+'.'+suffix] = f.get_tensor(stem+'.'+suffix).clone()
a.output.parent.mkdir(parents=True, exist_ok=True)
save_file(payload, str(a.output), metadata={'source_revision': '2a30229e67012798ba9f0cd832bb78abf4c363d5',
                                          'operation': 'exact tensor copy; no requantization'})
a.output.chmod(0o644)
with a.output.open('rb') as f:
    digest = hashlib.file_digest(f, 'sha256').hexdigest()
receipt = dict(source_revision='2a30229e67012798ba9f0cd832bb78abf4c363d5', source_file='mtp.safetensors',
               source_sha256=source_sha, output_sha256=digest, tensor_count=len(payload), modules=stems,
               tensor_bytes=sum(t.numel()*t.element_size() for t in payload.values()),
               operation='exact original tensor copy')
a.output.with_suffix('.json').write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps({k: v for k, v in receipt.items() if k != 'modules'}), flush=True)
