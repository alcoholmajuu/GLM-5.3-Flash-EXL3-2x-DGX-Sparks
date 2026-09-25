"""Importable namespace for the separately licensed, pinned runtime adapter.

Spawn unpickling imports this installed module before general plugins run.
The upstream payload stays external and retains its Apache-2.0 notice.
"""
from pathlib import Path

_adapter_path = Path('/opt/recipe/exl3.py')
if not _adapter_path.is_file():
    raise RuntimeError('Pinned EXL3 adapter missing: '+str(_adapter_path))
exec(compile(_adapter_path.read_bytes(), str(_adapter_path), 'exec'), globals())
