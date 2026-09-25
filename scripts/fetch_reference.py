# SPDX-License-Identifier: MIT
"""Retrieve individual allowlisted files at locked commits and record provenance."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('repository')
parser.add_argument('paths', nargs='+')
args = parser.parse_args()
lock = json.loads((ROOT / 'versions.lock').read_text())
component = next(x for x in lock['components'] if x['repository'] == args.repository)
sha = component['revision']
if args.repository.startswith('MiaAI-Lab/'):
    assert sha == 'eb9bd4805605bd66653ed3c887db637e4d5980b6'
manifest_path = ROOT / 'results/phase1/reference_files.json'
manifest_path.parent.mkdir(parents=True, exist_ok=True)
manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
for path in args.paths:
    if Path(path).is_absolute() or '..' in Path(path).parts or path.endswith(('.pt', '.safetensors')):
        raise ValueError(path)
    url = f'https://raw.githubusercontent.com/{args.repository}/{sha}/{path}'
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read()
    target = ROOT / 'vendor' / args.repository / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    manifest[str(target.relative_to(ROOT))] = {'repository': args.repository, 'revision': sha,
                                              'path': path, 'url': url, 'license': component['license'],
                                              'sha256': hashlib.sha256(data).hexdigest()}
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(path, len(data))
manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
