# SPDX-License-Identifier: MIT
"""Verify converted shard hashes, tensor inventory and packed-expert boundary."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from safetensors import safe_open

parser = argparse.ArgumentParser()
parser.add_argument('checkpoint', type=Path)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--source-metadata', type=Path, required=True)
args = parser.parse_args()
root = args.checkpoint
complete = json.loads((root/'conversion_receipts/complete.json').read_text())
assert complete['source_revision'] == '2a30229e67012798ba9f0cd832bb78abf4c363d5'
assert len(complete['shards']) == 20 and 'mtp.safetensors' in complete['shards']
index = json.loads((root/'model.safetensors.index.json').read_text())
metadata = json.loads(args.source_metadata.read_text())['data']
assert metadata['sha'] == complete['source_revision']
source_hashes = {row['rfilename']: row['lfs']['sha256'] for row in metadata['siblings']
                 if row['rfilename'].endswith('.safetensors')}
actual_map = {}
shard_results = []
packed_counts = {}
converted_counts = {}
mtp_layers = set()
started = time.monotonic()
for shard in complete['shards']:
    path = root/shard
    receipt = json.loads((root/'conversion_receipts'/(shard+'.json')).read_text())
    assert receipt['input_sha256'] == source_hashes[shard], 'Source revision drift: '+shard
    with path.open('rb') as handle:
        actual_sha = hashlib.file_digest(handle, 'sha256').hexdigest()
    assert actual_sha == receipt['output_sha256'], shard
    with safe_open(str(path), framework='pt', device='cpu') as handle:
        keys = sorted(handle.keys())
        assert keys == receipt['output_keys'], shard
        for key in keys:
            assert key not in actual_map, key
            actual_map[key] = shard
            if shard == 'mtp.safetensors' and '.layers.' in key:
                mtp_layers.add(int(key.split('.layers.')[1].split('.')[0]))
            if key.endswith('.trellis'):
                assert '.mlp.experts.' in key, key
                shape = handle.get_slice(key).get_shape()
                k = shape[-1] // 16
                assert k == 4, (key, shape)
                packed_counts[shard] = packed_counts.get(shard, 0) + 1
    converted_counts[shard] = len(receipt['modules'])
    shard_results.append({'file': shard, 'sha256': actual_sha, 'tensor_count': len(keys)})
    print(json.dumps(shard_results[-1]), flush=True)
assert actual_map == index['weight_map']
assert len(actual_map) == complete['tensor_count']
assert mtp_layers == {45}, mtp_layers
quant = json.loads((root/'quantization_config.json').read_text())
assert quant['bits'] == 4 and quant['codebook'] == 'mul1'
assert all('.mlp.experts.' in name for name in quant['tensor_storage'])
result = {'complete': True, 'source_revision': complete['source_revision'],
          'elapsed_seconds': time.monotonic()-started, 'shards': shard_results,
          'tensor_count': len(actual_map), 'mtp_layers_zero_based': sorted(mtp_layers),
          'packed_expert_matrices_by_shard': packed_counts,
          'converted_nonexpert_matrices_by_shard': converted_counts}
args.output.write_text(json.dumps(result, indent=2)+'\n')
