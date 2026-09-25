# SPDX-License-Identifier: MIT
"""Wait for conversion, verify both nodes and transfer only this recipe's output.

Does not launch inference, delete files, change source weights or overwrite logs.
Use run_recorded.py to retain the command stream through context compaction.
"""
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
REMOTE = os.environ.get('REMOTE_ROOT', '/opt/glm53-spark')
RANK0 = os.environ['RANK0_HOST']
RANK1 = os.environ['RANK1_HOST']
RANK1_IP = os.environ.get('RANK1_IP', os.environ['RANK1_HOST'])
MODEL = REMOTE+'/checkpoints/target-bf16-nonexperts'
IMAGE = 'sha256:cc141151db8ea05a3a51d87ffa82fd4560381256d8f8c068f8d0fb86c734b3ec'


def run(argv):
    print(json.dumps({'argv': argv}), flush=True)
    subprocess.run(argv, check=True)


while True:
    done = subprocess.run(['ssh', RANK0, 'test -f '+MODEL+'/conversion_receipts/complete.json'])
    if done.returncode == 0:
        break
    active = subprocess.run(['ssh', RANK0, 'docker inspect -f "{{.State.Running}}" glm53-phase1-convert'],
                            capture_output=True, text=True)
    if active.returncode or active.stdout.strip() != 'true':
        raise RuntimeError('Conversion is not complete and converter is not running; inspect logs')
    print(json.dumps({'waiting': 'conversion complete.json', 'utc_epoch': time.time()}), flush=True)
    time.sleep(30)

# The in-flight conversion predates the output-mode fix. Limit chmod to the
# generated final safetensors directly in this single, validated checkpoint.
run(['ssh', RANK0, shlex.join(['docker', 'run', '--rm', '-v', REMOTE+':/work',
     '--entrypoint', 'find', IMAGE, '/work/checkpoints/target-bf16-nonexperts',
     '-maxdepth', '1', '-type', 'f', '-name', '*.safetensors', '-exec', 'chmod', '644', '{}', '+'])])

for host in (RANK0, RANK1):
    run(['scp', str(ROOT/'scripts/verify_converted_checkpoint.py'), host+':'+REMOTE+'/'])
    if host == RANK1:
        run(['ssh', RANK0, shlex.join(['rsync', '-a', '--info=stats2', '--exclude=*.tmp',
             MODEL+'/', RANK1_IP+':'+MODEL+'/'])])
    run(['ssh', host, shlex.join(['docker', 'run', '--rm', '-v', REMOTE+':/work',
         '--entrypoint', 'python3', IMAGE, '/work/verify_converted_checkpoint.py',
         '/work/checkpoints/target-bf16-nonexperts', '--source-metadata', '/work/target_weight_metadata.json',
         '--output', '/work/checkpoint_verified_'+host+'.json'])])
    run(['scp', host+':'+REMOTE+'/checkpoint_verified_'+host+'.json',
         str(ROOT/'receipts'/('checkpoint_verified_'+host+'.json'))])
alice = json.loads((ROOT/'receipts/checkpoint_verified_'+RANK0+'.json').read_text())
bella = json.loads((ROOT/'receipts/checkpoint_verified_'+RANK1+'.json').read_text())
assert alice['shards'] == bella['shards'], 'Node checkpoint mismatch'
print(json.dumps({'complete': True, 'both_nodes_identical': True, 'serving_started': False}), flush=True)
