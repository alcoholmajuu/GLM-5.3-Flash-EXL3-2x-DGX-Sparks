"""MIT integration entry point; upstream EXL3 adapter retains its Apache notice."""
import importlib
import os

def register():
    upstream = importlib.import_module('glm53_spark.upstream')
    if os.environ.get('GLM53_EXL3_V15') == '1':
        from .exl3_v15 import install
        install(upstream,cooperative_max_rows=int(os.environ.get('GLM53_MOE_COOP_ROWS','2')))
    if os.environ.get('GLM53_PACKED_MLP'):
        from .packed_mlp import install
        install(upstream)
    if os.environ.get('GLM53_PACKED_HEAD'):
        from .packed_head import install
        install(upstream)
    if os.environ.get('GLM53_PACKED_ATTENTION'):
        from .packed_attention import install
        install(upstream)
    if os.environ.get('GLM53_PACKED_ATTENTION_A'):
        from .packed_attention_a import install
        install(upstream)
