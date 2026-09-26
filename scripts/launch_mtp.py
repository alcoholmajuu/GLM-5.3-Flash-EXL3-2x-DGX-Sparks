# SPDX-License-Identifier: MIT
"""Launch the MTP-best recipe pair on two DGX Spark nodes.

Configuration is environment-driven; no hostnames are hardcoded.
See ../.env.example. Requires the built image (docker/Dockerfile.mtp),
the converted checkpoint and packed sidecars (see README.md).
"""
import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PINS = json.loads((ROOT/'PINS.json').read_text())

parser = argparse.ArgumentParser()
parser.add_argument('--image', required=True, help='Immutable image sha256 ID from the build receipt')
parser.add_argument('--run', default='mtp', help='Unique run name [a-z][a-z0-9_]{0,63}')
parser.add_argument('--max-model-len', type=int, default=18432)
parser.add_argument('--gpu-memory-utilization', type=float, default=0.85)
parser.add_argument('--receipt', type=Path, required=True)
parser.add_argument('--vision', action='store_true',
                    help='Enable image input: mount chat_template/chat_template_vision.jinja '
                         'and pass --chat-template plus --limit-mm-per-prompt {"image":1}. '
                         'Weights are untouched.')
parser.add_argument('--execute', action='store_true')
args = parser.parse_args()

assert re.fullmatch(r'sha256:[0-9a-f]{64}', args.image)
assert re.fullmatch(r'[a-z][a-z0-9_]{0,63}', args.run)
assert 0.80 <= args.gpu_memory_utilization <= 0.85

RANK0 = os.environ['RANK0_HOST']
RANK1 = os.environ['RANK1_HOST']
MASTER_ADDR = os.environ.get('MASTER_ADDR', RANK0)
MASTER_PORT = os.environ.get('MASTER_PORT', '29692')
REMOTE = os.environ.get('REMOTE_ROOT', '/opt/glm53-spark')
MODEL = REMOTE+'/checkpoints/target-bf16-nonexperts'

common = ['--served-model-name','glm53-exl3','--tensor-parallel-size','2',
          '--nnodes','2','--distributed-executor-backend','mp',
          '--master-addr',MASTER_ADDR,'--master-port',MASTER_PORT,
          '--quantization','exl3','--dtype','bfloat16','--kv-cache-dtype','fp8_ds_mla',
          '--gpu-memory-utilization',str(args.gpu_memory_utilization),
          '--max-model-len',str(args.max_model_len),
          '--max-num-seqs','1','--max-num-batched-tokens','1024',
          '--no-enable-flashinfer-autotune',
          '--compilation-config',json.dumps({'cudagraph_mode':'FULL_AND_PIECEWISE',
                                             'cudagraph_capture_sizes':[1,2,4,8]}),
          '--no-enable-prefix-caching',
          '--speculative-config',json.dumps({'method':'mtp','num_speculative_tokens':3,
              'draft_tensor_parallel_size':2,'use_local_argmax_reduction':True,
              'draft_sample_method':'greedy'})]
if args.vision:
    common += ['--limit-mm-per-prompt',json.dumps({'image':1}),
               '--chat-template','/vision-test/chat_template_vision.jinja']
plans = []
for host,rank,ip in [(RANK1,1,os.environ.get('RANK1_IP',RANK1)),(RANK0,0,os.environ.get('RANK0_IP',RANK0))]:
    name = 'glm53-'+args.run+'-'+('rank1' if rank else 'rank0')
    env = {'NCCL_SOCKET_IFNAME':os.environ.get('NCCL_SOCKET_IFNAME','=enP2p1s0f0np0'),
           'NCCL_IB_HCA':os.environ.get('NCCL_IB_HCA','=roceP2p1s0f0:1'),
           'NCCL_IB_DISABLE':'0','NCCL_NET':'IB','NCCL_DEBUG':'INFO',
           'VLLM_HOST_IP':ip,
           'VLLM_NCCL_SO_PATH':'/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib/libnccl.so.2',
           'EXL3_FUSED_MOE':'1','EXL3_FAT_KERNEL':'0','EXL3_FAT_BATCHED':'1',
           'EXL3_FAT_EXPERT_LOG':'0','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
           'VLLM_NO_USAGE_STATS':'1','DO_NOT_TRACK':'1',
           'GLM53_MOE_COOP_ROWS':'8','NCCL_MAX_CTAS':'4',
           'VLLM_USE_BREAKABLE_CUDAGRAPH':'0',
           'GLM53_PACKED_MLP':'/packed-mlp/model.safetensors',
           'GLM53_PACKED_HEAD':'/packed-head/model.safetensors',
           'GLM53_PACKED_ATTENTION':'/packed-attention/model.safetensors',
           'GLM53_PACKED_ATTENTION_A':'/packed-attention-a/model.safetensors',
           'GLM53_PACKED_MTP':'/packed-mtp/model.safetensors'}
    command = ['docker','run','-d','--name',name,'--gpus','all','--network','host','--ipc','host',
               '--device','/dev/infiniband','--ulimit','memlock=-1','--ulimit','stack=67108864',
               '-v',MODEL+':/model:ro','-v',REMOTE+'/runtime-cache:/root/.cache',
               '--entrypoint','vllm',
               '-v',REMOTE+'/checkpoints/packed-mlp-original:/packed-mlp:ro',
               '-v',REMOTE+'/checkpoints/packed-head-original:/packed-head:ro',
               '-v',REMOTE+'/checkpoints/packed-attention-original:/packed-attention:ro',
               '-v',REMOTE+'/checkpoints/packed-attention-a-original:/packed-attention-a:ro',
               '-v',REMOTE+'/checkpoints/packed-mtp-original:/packed-mtp:ro',
               '-v',REMOTE+'/src/glm53_spark:/usr/local/lib/python3.12/dist-packages/glm53_spark:ro']
    if args.vision:
        command += ['-v',REMOTE+'/chat_template/chat_template_vision.jinja:/vision-test/chat_template_vision.jinja:ro']
    for key,value in env.items():
        command += ['-e',key+'='+value]
    command += [args.image,'serve','/model',*common,'--node-rank',str(rank)]
    command += ['--headless'] if rank else ['--host','127.0.0.1','--port','8000']
    plans.append({'host':host,'rank':rank,'argv':command,'container':name})

receipt = {'utc':datetime.now(timezone.utc).isoformat(),'executed':args.execute,
           'run':args.run,'plans':plans,'vision':bool(args.vision),
           'plugin_sources':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted((ROOT/'src/glm53_spark').glob('*.py'))}}
if args.vision:
    receipt['chat_template_vision'] = {
        'sha256':hashlib.sha256((ROOT/'chat_template/chat_template_vision.jinja').read_bytes()).hexdigest()}
if args.execute:
    for plan in plans:
        live = subprocess.check_output(['ssh',plan['host'],'docker ps --format '+shlex.quote('{{.Names}}')],
                                       text=True).splitlines()
        if any(n.startswith('glm53-') for n in live):
            raise RuntimeError('Recipe container still running on '+plan['host'])
        probe = 'test -f '+shlex.quote(MODEL+'/conversion_receipts/complete.json')
        subprocess.run(['ssh',plan['host'],probe],check=True)
        actual = subprocess.check_output(
            ['ssh',plan['host'],'docker image inspect --format '+shlex.quote('{{.Id}}')+' '+shlex.quote(args.image)],
            text=True).strip()
        if actual != args.image:
            raise RuntimeError('Image identity mismatch on '+plan['host'])
        plan['verified_image_id'] = actual
        for dirname,key in [('packed-mlp-original','packed_mlp'),('packed-head-original','packed_head'),
                            ('packed-attention-original','packed_attention'),
                            ('packed-attention-a-original','packed_attention_a'),
                            ('packed-mtp-original','packed_mtp')]:
            actual = subprocess.check_output(
                ['ssh',plan['host'],'sha256sum '+shlex.quote(REMOTE+'/checkpoints/'+dirname+'/model.safetensors')],
                text=True).split()[0]
            if actual != PINS['packed'][key]:
                raise RuntimeError(f'Packed sidecar mismatch on {plan["host"]}: {dirname}')
            plan[dirname.replace('-','_')+'_sha256'] = actual
        for path,digest in receipt['plugin_sources'].items():
            actual = subprocess.check_output(
                ['ssh',plan['host'],'sha256sum '+shlex.quote(REMOTE+'/'+path)],text=True).split()[0]
            assert actual == digest,(plan['host'],path)
        if args.vision:
            actual = subprocess.check_output(
                ['ssh',plan['host'],'sha256sum '+shlex.quote(REMOTE+'/chat_template/chat_template_vision.jinja')],
                text=True).split()[0]
            assert actual == receipt['chat_template_vision']['sha256'],(plan['host'],actual)
            plan['chat_template_vision_sha256'] = actual
        existing = subprocess.run(['ssh',plan['host'],'docker container inspect '+plan['container']],
                                  capture_output=True)
        if existing.returncode == 0:
            raise RuntimeError('Existing container requires inspection: '+plan['container'])
    for plan in plans:
        result = subprocess.run(['ssh',plan['host'],shlex.join(plan['argv'])],capture_output=True,text=True)
        plan.update(exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr)
        args.receipt.write_text(json.dumps(receipt,indent=2)+'\n')
        if result.returncode:
            raise RuntimeError('Launch failed; inspect receipt and already started container')
args.receipt.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'receipt':str(args.receipt),'containers':[p['container'] for p in plans]},indent=2))
