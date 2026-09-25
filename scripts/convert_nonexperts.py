"""Expand only non-routed EXL3 matrices to BF16, preserving packed experts.

Uses the pinned MIT ExLlamaV3 decoder as an external library. Input is immutable;
outputs and per-shard receipts are atomic and resumable. No foreign scales/head.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import time
import types

import torch
from safetensors import safe_open
from safetensors.torch import save_file

def digest(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()

def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)

def native_names(name, value, layers):
    match = re.search(r'\.layers\.(\d+)\.self_attn\.(qkv_proj|conv1d)\.weight$', name)
    if match and (match[2] == 'conv1d' or layers[int(match[1])] == 'linear_attention'):
        if value.shape[0] % 3:
            raise ValueError(f'Cannot split QKV rows: {name} {value.shape}')
        tail = 'proj' if match[2] == 'qkv_proj' else 'conv1d'
        return {name.replace(match[2]+'.weight', letter+'_'+tail+'.weight'): chunk.contiguous()
                for letter, chunk in zip('qkv', value.chunk(3, dim=0))}
    if '.visual.' in name:
        for letter in 'qkv':
            name = name.replace(f'.attn.{letter}_proj.', f'.attn.{letter}.')
    return {name: value.contiguous()}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--overlay', default='/opt/recipe/exl3.py')
    parser.add_argument('--max-shards', type=int)
    parser.add_argument('--wait-for-download', action='store_true')
    args = parser.parse_args()
    if args.source.resolve() == args.destination.resolve():
        raise ValueError('Source and destination must differ')
    args.destination.mkdir(parents=True, exist_ok=True)
    receipts = args.destination/'conversion_receipts'
    receipts.mkdir(exist_ok=True)
    def require_file(path):
        while not path.is_file():
            if not args.wait_for_download:
                raise FileNotFoundError(path)
            print(json.dumps({'waiting_for':str(path)}),flush=True)
            time.sleep(30)
    for file, expected in {
        'quantization_config.json':'d18c35bf6184a67a3b612b15c216bdfdf9e81505c0e02b646152925b13fafef3',
        'model.safetensors.index.json':'ee1f2dbea800dea0b4225c38193f7ef41180ed42dd51f80a0cee58daf44cf606',
    }.items():
        require_file(args.source/file)
        if digest(args.source/file) != expected:
            raise ValueError('Input metadata does not match locked target: '+file)
    config = json.loads((args.source/'config.json').read_text())
    layers = config['text_config']['layer_types']
    quant = json.loads((args.source/'quantization_config.json').read_text())
    if quant['codebook'] != 'mul1':
        raise ValueError('Only the approved mul1 checkpoint is supported')
    spec = importlib.util.spec_from_file_location('exl3_reference', args.overlay)
    overlay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(overlay)
    linear_cls = overlay.load_linear_exl3_cls()
    index = json.loads((args.source/'model.safetensors.index.json').read_text())
    shards = sorted(set(index['weight_map'].values()))
    if 'mtp.safetensors' not in shards:
        shards.append('mtp.safetensors')
    output_map = {}
    total_size = 0
    for shard_number, shard in enumerate(shards):
        if args.max_shards and shard_number >= args.max_shards:
            print('Partial conversion only; no final index emitted', flush=True)
            return
        source = args.source/shard
        require_file(source)
        destination = args.destination/shard
        receipt_path = receipts/(shard+'.json')
        input_hash = digest(source)
        if receipt_path.exists() and destination.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt['input_sha256'] != input_hash or receipt['output_sha256'] != digest(destination):
                raise RuntimeError('Resume hash mismatch: '+shard)
            output_map.update({key:shard for key in receipt['output_keys']})
            total_size += receipt['tensor_bytes']
            continue
        started = time.monotonic()
        output = {}
        modules = []
        with safe_open(str(source), framework='pt', device='cpu') as handle:
            keys = set(handle.keys())
            converted = {key[:-len('.trellis')] for key in keys
                         if key.endswith('.trellis') and '.mlp.experts.' not in key}
            consumed = set()
            for prefix in sorted(converted):
                names = {suffix:prefix+'.'+suffix for suffix in ('trellis','suh','svh','mul1')}
                if not set(names.values()) <= keys:
                    raise RuntimeError('EXL3 group crosses shards or lacks a component: '+prefix)
                tensors = {suffix:handle.get_tensor(name).to('cuda') for suffix,name in names.items()}
                marker = int(tensors['mul1'].item()) & 0xffffffff
                if marker != 2212286765:
                    raise ValueError(f'Unexpected mul1 multiplier {prefix}: {marker}')
                inner = linear_cls(config=types.SimpleNamespace(infer_params=types.SimpleNamespace(no_reconstruct=False)), in_features=tensors['suh'].numel(),
                                   out_features=tensors['svh'].numel(), transformers_fix=True,
                                   out_dtype=torch.float16, **tensors)
                # get_weight_tensor includes both Hadamards and both sign/scale vectors;
                # upstream matrix orientation is input x output, HF Linear is output x input.
                weight = inner.get_weight_tensor().T.contiguous().to(torch.bfloat16).cpu()
                for key,value in native_names(prefix+'.weight',weight,layers).items():
                    if key in output:
                        raise ValueError('Output collision: '+key)
                    output[key] = value
                modules.append({'name':prefix,'K':inner.K,'shape':list(weight.shape),
                                'mul1_multiplier':marker,'output_dtype':'BF16'})
                consumed.update(names.values())
                del tensors, inner, weight
            for name in sorted(keys-consumed):
                value = handle.get_tensor(name)
                # Native vision has a fused qkv bias as well as split copies.
                # Retain fused bias and verify any redundant split against its slice.
                m = re.search(r'(.*\.visual\..*\.attn\.)(([qkv])_proj)\.bias$', name)
                if m and m[1]+'qkv.bias' in keys:
                    fused = handle.get_tensor(m[1]+'qkv.bias')
                    expected = fused.chunk(3,dim=0)['qkv'.index(m[3])]
                    if not torch.equal(value,expected):
                        raise ValueError('Split/fused vision bias mismatch: '+name)
                    continue
                for key, tensor in native_names(name,value,layers).items():
                    if key in output:
                        raise ValueError('Duplicate output tensor: '+key)
                    output[key] = tensor
        temp = destination.with_suffix('.safetensors.tmp')
        save_file(output,str(temp))
        # safetensors creates mode 0600; checkpoints must be readable by the
        # host user for verified node-to-node transfer from a root container.
        temp.chmod(0o644)
        temp.replace(destination)
        tensor_bytes = sum(t.numel()*t.element_size() for t in output.values())
        receipt = {'input_sha256':input_hash,'output_sha256':digest(destination),
                   'output_keys':sorted(output),'modules':modules,'tensor_bytes':tensor_bytes,
                   'elapsed_seconds':time.monotonic()-started,'routed_experts_modified':False}
        write_json(receipt_path,receipt)
        output_map.update({key:shard for key in output})
        total_size += tensor_bytes
        print(json.dumps({'shard':shard,'converted_modules':len(modules),
                          'elapsed_seconds':receipt['elapsed_seconds']}),flush=True)
        del output
        gc.collect()
        torch.cuda.empty_cache()
    # Converted non-expert ledger entries must not activate the TP1 EXL3 path.
    quant['tensor_storage'] = {k:v for k,v in quant['tensor_storage'].items() if '.mlp.experts.' in k}
    quant.update(bits=4, scope='glm53_routed_experts_only', original_declared_bits=4.05)
    config['quantization_config'] = {k:v for k,v in quant.items() if k != 'tensor_storage'}
    config['text_config']['quantization_config'] = config['quantization_config']
    write_json(args.destination/'quantization_config.json',quant)
    write_json(args.destination/'config.json',config)
    for filename in ('LICENSE','README.md','chat_template.jinja','generation_config.json',
                     'processor_config.json','tokenizer_config.json','tokenizer.json'):
        file = args.source/filename
        require_file(file)
        shutil.copyfile(file,args.destination/file.name)
    write_json(args.destination/'model.safetensors.index.json',
               {'metadata':{'total_size':total_size},'weight_map':output_map})
    write_json(receipts/'complete.json',{'source_revision':'2a30229e67012798ba9f0cd832bb78abf4c363d5',
                                        'shards':shards,'tensor_count':len(output_map)})

if __name__ == '__main__':
    main()
