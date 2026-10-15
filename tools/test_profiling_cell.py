#!/usr/bin/env python3
"""Test the pure analysis helpers of notebook cell 4 (the profiling ladder).

Those helpers run blind inside Kaggle, against tool output we cannot reproduce
here (no GPU, no nvcc), so they are extracted from the notebook cell itself,
AST-filtered down to the dependency-free functions, and run against fixtures that
copy the real output formats of ptxas -v, cuobjdump, nvdisasm/SASS and
`nvidia-smi --query-gpu --format=csv`.

Run:  python tools/test_profiling_cell.py     (stdlib only, no GPU needed)
"""
import ast
import json
import pathlib
import re
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "kaggle_gbm_montecarlo.ipynb"

KERNEL = "gbm_terminal_price_kernel"
WANTED = {"_num", "_smi_ts", "parse_ptxas_v", "parse_res_usage", "census_sass",
          "parse_smi_trace", "occupancy"}

# --------------------------------------------------------------------- fixture
PTXAS_LOG = """
$ /usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_75 -lineinfo -Xptxas -v -o build/gbm src/gbm_montecarlo.cu -lcurand

ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function '_Z26gbm_terminal_price_kernelPfxillm' for 'sm_75'
ptxas info    : Function properties for _Z26gbm_terminal_price_kernelPfxillm
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 26 registers, 384 bytes cmem[0], 0 bytes smem, 380 bytes cmem[1]
"""

RES_USAGE = """
Resource usage report for build/gbm

Function gbm_terminal_price_kernel:
  REG:26  STACK:0  SHARED:0  LOCAL:0  CONSTANT[0]:384  TEXTURE:0  SURFACE:0  SAMPLER:0

Function _Z26gbm_terminal_price_kernelPfxillm:
  REG:26 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:384
"""

# A slice of the shape cuobjdump -sass emits, with a predicate, a label, an
# explicit braced operand and a second function that must NOT be counted.
SASS_DUMP = """
        Function : _ZN9SomeOtherKernelEi
        /*0000*/                   LDG.E R1, [R2.64] ;
        /*0010*/                   F2F.F32.F64 R2, R4 ;
        /*0020*/                   EXIT ;

        Function : _Z26gbm_terminal_price_kernelPfxillm
        /*0000*/                   IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28] ;
        /*0010*/                   S2R R0, SR_TID.X ;
        /*0020*/                   IMAD R0, R2, c[0x0][0x178], R0 ;
        /*0030*/                   ISETP.GE.U32.AND P0, PT, R0, c[0x0][0x160], PT ;
        /*0040*/              @P0   EXIT ;
        /*0050*/                   MOV R5, RZ ;
        /*0060*/                   MUFU.RCP R6, R4 ;
.LL_x_1:
        /*0070*/                   FFMA R7, R4, R6, -1 ;
        /*0080*/                   FFMA R8, R7, R6, R6 ;
        /*0090*/                   MUFU.EX2 R9, R8 ;
        /*00a0*/                   FMUL R10, R9, 0.69314718 ;
        /*00b0*/                   MUFU.EX2 R11, R10 ;
        /*00c0*/              @!P0  BRA 0x70 ;
        /*00d0*/                   STG.E [R2.64], R11 ;
        /*00e0*/                   EXIT ;
"""

# `nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,clocks.sm,\
#  clocks.mem,power.draw,temperature.gpu --format=csv -lms 50`
SMI_TRACE = """timestamp, utilization.gpu [%], utilization.memory [%], clocks.sm [MHz], clocks.mem [MHz], power.draw [W], temperature.gpu
2026/10/15 09:12:00.100, 0 %, 0 %, 585 MHz, 405 MHz, 9.83 W, 38
2026/10/15 09:12:00.150, 3 %, 1 %, 585 MHz, 405 MHz, 9.90 W, 38
2026/10/15 09:12:00.200, 97 %, 41 %, 1590 MHz, 5001 MHz, 68.42 W, 46
2026/10/15 09:12:00.250, 99 %, 55 %, 1590 MHz, 5001 MHz, [Not Supported], 47
2026/10/15 09:12:00.300, 96 %, 38 %, 1590 MHz, 5001 MHz, [N/A], 47
"""

failures = []


def check(label, ok, detail=""):
    print(("  PASS  " if ok else "  FAIL  ") + label + (("  " + detail) if detail else ""))
    if not ok:
        failures.append(label)


def load_helpers():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    cell_src = "".join([c for c in nb["cells"] if c["cell_type"] == "code"][3]["source"])
    if "gbm_terminal_price_kernel" not in cell_src or "census_sass" not in cell_src:
        sys.exit("cell 4 is not the profiling cell - check the cell order")
    tree = ast.parse(cell_src)
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in WANTED]
    found = {f.name for f in funcs}
    if found != WANTED:
        sys.exit("helpers missing from the cell: %s" % sorted(WANTED - found))
    ns = {"re": re, "time": time, "KERNEL": KERNEL}
    exec(compile(ast.fix_missing_locations(ast.Module(body=funcs, type_ignores=[])),
                 "<notebook-cell-4-helpers>", "exec"), ns)
    return ns


ns = load_helpers()
check("all %d helpers extracted from the notebook cell" % len(WANTED), True)

# -------------------------------------------------------------- ptxas -v log
info = ns["parse_ptxas_v"](PTXAS_LOG)
check("ptxas -v: registers per thread", info.get("registers_per_thread") == 26, str(info))
check("ptxas -v: zero spill stores/loads detected",
      info.get("spill_stores_bytes") == 0 and info.get("spill_loads_bytes") == 0, str(info))
check("ptxas -v: stack frame is 0", info.get("stack_bytes") == 0, str(info))
check("ptxas -v: static smem read from the 'bytes smem' field",
      info.get("static_smem_bytes") == 0, str(info))
cmem_only = ns["parse_ptxas_v"]("ptxas info : Used 30 registers, 384 bytes cmem[0]")
check("ptxas -v: 'cmem' is not mistaken for 'smem'",
      "static_smem_bytes" not in cmem_only, str(cmem_only))
check("ptxas -v: empty log yields no claims", ns["parse_ptxas_v"]("") == {})

# ------------------------------------------------------- cuobjdump -res-usage
usage = ns["parse_res_usage"](RES_USAGE)
check("res-usage: REG parsed", usage.get("REG") == 26, str(usage))
check("res-usage: only numeric fields kept (no key with empty value)",
      all(isinstance(v, int) for v in usage.values()), str(usage))
check("res-usage: absent report yields {}", ns["parse_res_usage"]("nothing here") == {})

# ------------------------------------------------------------- SASS census
total, counts = ns["census_sass"](SASS_DUMP)
check("SASS: every instruction of the target function is counted", total == 15,
      "%d instructions" % total)
check("SASS: the other function in the dump is excluded",
      "F2F.F32.F64" not in counts and counts.get("LDG.E", 0) == 0, str(counts))
check("SASS: both EXIT paths of the target function are counted", counts.get("EXIT") == 2,
      str(counts.get("EXIT")))
check("SASS: predicated and labelled lines are counted",
      counts.get("BRA") == 1 and counts.get("MUFU.EX2", 0) == 2, str(counts))
check("SASS: FFMA/FMUL/STG counted", (counts.get("FFMA"), counts.get("FMUL"),
                                      counts.get("STG.E")) == (2, 1, 1), str(counts))
check("SASS: a dump with no target function yields nothing",
      ns["census_sass"]("Function : _Z9otherKernelv\n/*0000*/ EXIT ;") == (0, {}))

# ---------------------------------------------------------- nvidia-smi trace
rows = ns["parse_smi_trace"](SMI_TRACE)
check("smi: header skipped, 5 samples parsed", len(rows) == 5, str(len(rows)))
check("smi: units stripped from utilisation",
      rows[2]["gpu_util"] == 97 and rows[2]["mem_util"] == 41, str(rows[2]))
check("smi: clocks parsed", rows[2]["sm_mhz"] == 1590 and rows[2]["mem_mhz"] == 5001, str(rows[2]))
check("smi: unsupported power reported as None, not 0",
      rows[3]["power_w"] is None and rows[4]["power_w"] is None, str([r["power_w"] for r in rows]))
check("smi: timestamp parsed to epoch seconds",
      rows[0]["ts_epoch"] is not None and rows[2]["ts_epoch"] > rows[0]["ts_epoch"],
      str(rows[0]["ts_epoch"]))
check("smi: idle vs busy split is separable",
      [r["gpu_util"] for r in rows] == [0, 3, 97, 99, 96], str([r["gpu_util"] for r in rows]))
check("smi: garbage input yields no rows",
      ns["parse_smi_trace"]("nvidia-smi: command not found\n") == [])

# -------------------------------------------------------------- occupancy
PROPS = {"device": "Tesla T4", "cc": 7.5, "sms": 40, "max_threads_per_sm": 1024,
         "max_threads_per_block": 1024, "smem_per_block": 49152, "warp_size": 32,
         "regs_per_sm": 65536}
occ = ns["occupancy"](26, 256, PROPS)
# 26 regs -> 4 reg-warps of 256 per warp, 8 warps per block -> 8192 regs/block
# -> 65536/8192 = 8 blocks by registers, 1024/256 = 4 blocks by threads -> 4 blocks
check("occupancy: registers per block rounded to the warp granularity",
      occ.get("regs_per_block") == 8192, str(occ))
check("occupancy: register limit computed", occ["limits"]["registers"] == 8, str(occ))
check("occupancy: thread limit computed", occ["limits"]["threads"] == 4, str(occ))
check("occupancy: resident blocks = min of the limits",
      occ["resident_blocks_per_sm"] == 4, str(occ))
check("occupancy: limiter identified", occ["limiter"] == "threads", str(occ))
check("occupancy: 100% of the SM's warp slots", abs(occ["theoretical_occupancy_pct"] - 100.0) < 1e-9,
      "%.1f%%" % occ["theoretical_occupancy_pct"])
heavy = ns["occupancy"](80, 256, PROPS)
# 80 regs -> 10 reg-warps of 256, 8 warps per block -> 20480 regs/block
# -> 65536/20480 = 3 blocks by registers, 4 by threads -> register limited, 24/32 warps
check("occupancy: a register-heavy kernel is register-limited",
      heavy.get("regs_per_block") == 20480 and heavy["limiter"] == "registers"
      and heavy["resident_blocks_per_sm"] == 3
      and abs(heavy["theoretical_occupancy_pct"] - 75.0) < 1e-9, str(heavy))
check("occupancy: missing facts produce no claim",
      ns["occupancy"](0, 256, PROPS) == {} and ns["occupancy"](26, 256, {}) == {})
check("occupancy: zero registers never divides by zero",
      ns["occupancy"](0, 256, {"sms": 40}) == {})

print("")
print("%d checks failed" % len(failures) if failures else "all checks passed")
for f in failures:
    print("  -", f)
sys.exit(1 if failures else 0)
