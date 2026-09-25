"""MIT: complete the disabled CPU-offload ABI on ARM for fixed ExLlamaV3 v1.5."""
from pathlib import Path
import sys

path = Path(sys.argv[1]) / 'cpu/moe_mul1.cpp'
source = path.read_text()
anchor = 'bool exl3_moe_cpu_has_avx2() { return false; }'
assert source.count(anchor) == 1
source = source.replace(anchor, anchor + '''
bool exl3_moe_cpu_has_avx512_bw() { return false; }
int64_t exl3_moe_cpu_pool_stress(int, int, int, int) {
    TORCH_CHECK(false, "CPU MoE unavailable on aarch64");
}
''')
path.write_text(source)
