#!/usr/bin/env python3
"""Run notebook cell 4 (plots + profiling ladder + summary) end to end off-device.

The cell is executed with fake `cuobjdump` and `nvidia-smi` shims on PATH and
fixtures for the cell-3 artifacts, so the profiling ladder goes through its real
code paths on a machine that has neither a GPU nor nvcc. What this proves is that
the ladder *writes a well-formed record instead of crashing*: every section of
results/profiling.md exists, the measured/static/derived labels are honest, the
raw artifacts land on disk, and the benchmark summary still gets written.

What it cannot prove: the numbers a real T4 produces, and the "everything worked"
branch of the representative run (that needs the compiled binary).

Run:  python tools/test_profiling_ladder.py     (needs pandas + matplotlib)
"""
import contextlib
import io
import json
import os
import pathlib
import shutil
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "kaggle_gbm_montecarlo.ipynb"

SIZES_CSV = """paths,steps,block,cpu_ms,gpu_kernel_ms,gpu_e2e_ms,speedup_kernel,speedup_e2e,gpu_paths_per_sec,cpu_reps,gpu_reps
10000,252,256,86.29,0.082,0.196,1050.1,441.0,1.217e+08,3,5
100000,252,256,851.96,0.650,0.870,1310.5,979.1,1.538e+08,3,5
1000000,252,256,8572.62,6.168,7.267,1389.9,1179.6,1.621e+08,3,5
"""

BLOCKS_CSV = """paths,steps,block,gpu_kernel_ms,gpu_e2e_ms,gpu_paths_per_sec,paths_per_sec_per_step,reps
10000,252,64,0.081,0.207,1.236e+08,4.9,5
10000,252,256,0.082,0.193,1.219e+08,4.8,5
100000,252,256,0.650,0.899,1.539e+08,6.1,5
1000000,252,256,3.003,4.088,3.330e+08,13.2,5
"""

COMPILE_LOG = """$ /usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_75 -lineinfo -Xptxas -v -o build/gbm src/gbm_montecarlo.cu -lcurand

ptxas info    : Compiling entry function '_Z26gbm_terminal_price_kernelPfxillm' for 'sm_75'
ptxas info    : Function properties for _Z26gbm_terminal_price_kernelPfxillm
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 26 registers, 384 bytes cmem[0], 0 bytes smem
"""

ENV_CUDA = """cudaGetDeviceCount -> 1 (no error)
device                        : Tesla T4
compute capability            : 7.5
multiprocessors (SMs)         : 40
max threads per SM            : 1024
max threads per block         : 1024
shared memory per block       : 49152 bytes
warp size                     : 32
registers per SM              : 65536
global memory                 : 14.76 GiB
memory bus width              : 256 bit
memory clock rate             : 5001 MHz
L2 cache size                 : 4194304 bytes
concurrent kernels            : 1
cudaRuntimeGetVersion         : 12050
cudaDriverGetVersion          : 12050
compiled with nvcc            : 12.5
"""

RNG_FIT = """OLS fit over the step sweep (kernel-only CUDA-event timings)
  t_ms(steps) = a + b*steps
  a = 0.029774 ms   fixed per-launch cost
  b = 0.00898685 ms   marginal cost of one extra step
  R^2 = 0.999994  (points = 5)
"""

CUOBJDUMP_RES = """
Resource usage report for build/gbm

Function gbm_terminal_price_kernel:
  REG:26  STACK:0  SHARED:0  LOCAL:0  CONSTANT[0]:384  TEXTURE:0  SURFACE:0  SAMPLER:0
"""

CUOBJDUMP_SASS = """
        Function : _Z26gbm_terminal_price_kernelPfxillm
        /*0000*/                   IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28] ;
        /*0010*/                   S2R R0, SR_TID.X ;
        /*0020*/                   ISETP.GE.U32.AND P0, PT, R0, c[0x0][0x160], PT ;
        /*0030*/              @P0   EXIT ;
        /*0040*/                   MUFU.RCP R6, R4 ;
        /*0050*/                   FFMA R7, R4, R6, -1 ;
        /*0060*/                   MUFU.EX2 R9, R8 ;
        /*0070*/                   FMUL R10, R9, 0.69314718 ;
        /*0080*/                   MUFU.EX2 R11, R10 ;
        /*0090*/                   STG.E [R2.64], R11 ;
        /*00a0*/                   EXIT ;
"""

# The shim body: role is argv[1], the rest is the real argv. Placeholders use
# .replace() rather than %-formatting, so the inner print keeps its own % signs.
HELPER = '''#!/usr/bin/env python3
import datetime, sys, time

role, argv = sys.argv[1], sys.argv[2:]
if role == "cuobjdump":
    if "-res-usage" in argv:
        sys.stdout.write("""__RES__""")
    elif "-sass" in argv:
        sys.stdout.write("""__SASS__""")
elif role == "nvidia-smi":
    print("timestamp, utilization.gpu [%], utilization.memory [%], clocks.sm [MHz], "
          "clocks.mem [MHz], power.draw [W], temperature.gpu")
    for n in range(100):
        busy = n >= 20
        now = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S.%f")[:-3]
        print("%s, %d %%, %d %%, %d MHz, 5001 MHz, %s W, %d"
              % (now, 97 if busy else 0, 41 if busy else 0, 1590 if busy else 585,
                 "68.42" if busy else "9.83", 46 if busy else 38))
        sys.stdout.flush()
        time.sleep(0.05)
else:
    sys.exit(2)
'''

failures = []


def check(label, ok, detail=""):
    print(("  PASS  " if ok else "  FAIL  ") + label + (("  " + detail) if detail else ""))
    if not ok:
        failures.append(label)


def write_shims(bindir):
    """Create `cuobjdump` and `nvidia-smi` that answer with fixture output."""
    helper = bindir / "fake_tools.py"
    helper.write_text(HELPER.replace("__RES__", CUOBJDUMP_RES).replace("__SASS__", CUOBJDUMP_SASS),
                      encoding="utf-8")
    for tool in ("cuobjdump", "nvidia-smi"):
        if os.name == "nt":
            shim = bindir / (tool + ".bat")
            shim.write_text('@echo off\r\n"%s" "%s" %s %%*\r\n'
                            % (sys.executable, helper, tool), encoding="utf-8")
        else:
            shim = bindir / tool
            body = '#!/bin/sh\nexec "%s" "%s" %s "$@"\n' % (sys.executable, helper, tool)
            shim.write_text(body, encoding="utf-8")
            shim.chmod(0o755)


def make_workspace():
    ws = pathlib.Path(tempfile.mkdtemp(prefix="gbm_prof_test_"))
    res = ws / "results"
    res.mkdir()
    (ws / "build").mkdir()
    # Carry the real README so the cell's RESULTS-marker patching is exercised too.
    shutil.copy(ROOT / "README.md", ws / "README.md")
    (res / "benchmark_sizes.csv").write_text(SIZES_CSV, encoding="utf-8")
    (res / "benchmark_blocks.csv").write_text(BLOCKS_CSV, encoding="utf-8")
    (res / "compile_log.txt").write_text(COMPILE_LOG, encoding="utf-8")
    (res / "environment_cuda.txt").write_text(ENV_CUDA, encoding="utf-8")
    (res / "rng_overhead_fit.txt").write_text(RNG_FIT, encoding="utf-8")
    return ws


def run_cell4(ws, bindir):
    nb = json.loads(NB.read_text(encoding="utf-8"))
    cell = "".join([c for c in nb["cells"] if c["cell_type"] == "code"][3]["source"])
    if "census_sass" not in cell:
        sys.exit("cell 4 is not the profiling cell")

    cwd, path = os.getcwd(), os.environ["PATH"]
    os.environ["PATH"] = str(bindir) + os.pathsep + path
    try:
        os.chdir(ws)
        # cell 1 normally supplies these; sys is re-imported here because the cell
        # reads sys.version for the record
        ns = {"GPU_NAME": "Tesla T4 (shimmed)", "ARCH": "sm_75",
              "NVCC": "/usr/local/cuda/bin/nvcc", "cpu_model": "shim CPU", "n_cpu": 4,
              "sys": sys, "banner": lambda t: None, "run": lambda *a, **k: None}
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            try:
                exec(compile(cell, "<notebook-cell-4>", "exec"), ns)
                return None, out.getvalue()
            except Exception as exc:  # noqa: BLE001 - escaping the cell is the thing under test
                import traceback
                return traceback.format_exc(), out.getvalue()
    finally:
        os.chdir(cwd)
        os.environ["PATH"] = path


ws = make_workspace()
bindir = pathlib.Path(tempfile.mkdtemp(prefix="gbm_prof_shims_"))
try:
    write_shims(bindir)
    print("shims:", sorted(p.name for p in bindir.iterdir()))
    before = (ROOT / "README.md").read_text(encoding="utf-8")
    before_block = before[before.index("<!-- RESULTS:START -->"):before.index("<!-- RESULTS:END -->")]
    err, cell_out = run_cell4(ws, bindir)

    check("cell 4 completes without an exception escaping", err is None,
          (err or "").strip().splitlines()[-1] if err else "")
    prof = ws / "results" / "profiling.md"
    check("results/profiling.md was written", prof.exists())
    text = prof.read_text(encoding="utf-8") if prof.exists() else ""
    check("profiling.md is substantial", len(text) > 1500, "%d bytes" % len(text))
    check("the ladder did not fall into its error branch",
          "The Kaggle-compatible analysis failed" not in text)

    for heading in ("## 1. Tool discovery", "## 2. Nsight Systems", "## 3. Nsight Compute",
                    "## 4. Kaggle-compatible kernel profile", "### 4a. Representative run",
                    "### 4b. Kernel resource usage", "### 4c. SASS instruction census",
                    "### 4d. Occupancy and launch geometry", "### 4e. Sampled GPU telemetry",
                    "### 4f. Static-vs-measured cross-checks", "## 5. Limitations", "## 6. Getting"):
        check("profiling.md contains %r" % heading, heading in text)

    check("honesty labels are present", text.count("**measured**") >= 2
          and "**static**" in text and "**derived**" in text)
    check("no CUPTI counter result is claimed", "CUPTI hardware-counter" in text)
    check("an absent Nsight is reported as absent, not as a result",
          "Nsight Systems produced a usable timeline" not in text
          and "not in this image" in text)
    check("the ncu counter permission problem is documented as a limitation",
          "ERR_NVGPUCTRPERM" in text or "not in this image" in text)

    # --- the static path really consumed the tool output ---------------------
    check("4b used the register count from ptxas/cuobjdump", "| 26 |" in text, "")
    check("4c censused the SASS from the shimmed cuobjdump",
          "SASS instructions in" in text and "MUFU.EX2" in text)
    check("4c counts the shim's instructions", "`MUFU.EX2` | 2" in text, "")
    check("4d computed occupancy from the device properties and regs",
          "theoretical occupancy" in text and "% |" in text)
    check("4e split the trace into before/soak windows",
          "before the soak (1 s)" in text and "soak window" in text)
    check("4e reports distributions, not just peaks",
          "SM util mean" in text and "SM util median" in text and "SM util peak" in text)
    soak_row = [ln for ln in text.splitlines() if ln.startswith("| soak window")]
    before_row = [ln for ln in text.splitlines() if ln.startswith("| before the soak")]
    check("4e soak window shows the shim's busy utilisation",
          bool(soak_row) and "| 97% |" in soak_row[0], soak_row[0][:90] if soak_row else "")
    check("4e before window shows the shim's quiet utilisation",
          bool(before_row) and "| 0% | 0% |" in before_row[0],
          before_row[0][:90] if before_row else "")
    check("4e does not call the first window idle",
          "idle baseline" not in text and "guaranteed idle" in text)
    check("4f cross-checks expf against the census", "`expf` is a hardware instruction" in text)

    # --- the findings must be valid markdown tables, not just correct numbers -
    def tables(md):
        blocks, cur = [], []
        for line in md.splitlines():
            if line.strip().startswith("|") and line.strip().endswith("|"):
                cur.append(line)
            elif cur:
                blocks.append(cur)
                cur = []
        if cur:
            blocks.append(cur)
        return blocks

    found = tables(text)
    ragged = [b[0][:50] for b in found if len({ln.count("|") for ln in b}) != 1]
    check("every markdown table in profiling.md is rectangular (%d tables)" % len(found),
          not ragged and len(found) >= 6, "ragged: %s" % ragged[:2])
    check("every table has a header separator row",
          all(len(b) >= 2 and set(b[1].replace("|", "").replace(" ", "")) <= {"-", ":"}
              for b in found))

    # --- raw artifacts + the deliverables that follow the ladder -------------
    for art in ("cuobjdump_res_usage.txt", "sass_census.csv", "nvidia_smi_trace.csv"):
        check("raw artifact results/%s written" % art, (ws / "results" / art).exists())
    check("sass_census.csv is a real census",
          (ws / "results" / "sass_census.csv").exists()
          and "opcode,count" in (ws / "results" / "sass_census.csv").read_text(encoding="utf-8"))
    summary = ws / "results" / "summary.md"
    check("the summary after the ladder was still written", summary.exists())
    listing = summary.read_text(encoding="utf-8").split("## Files")[-1] if summary.exists() else ""
    check("the summary's file listing names files, not directories",
          " bytes" in listing and "profile  " not in listing, listing.strip()[:80])
    check("cell 3's benchmark CSVs are untouched",
          (ws / "results" / "benchmark_sizes.csv").read_text(encoding="utf-8") == SIZES_CSV)

    # --- the README's RESULTS block is patched, not mangled ------------------
    check("the cell reported patching the README", "patched README.md" in cell_out)
    after = (ws / "README.md").read_text(encoding="utf-8")
    check("the README still has exactly one marker pair",
          after.count("<!-- RESULTS:START -->") == 1 and after.count("<!-- RESULTS:END -->") == 1)
    after_block = after[after.index("<!-- RESULTS:START -->"):after.index("<!-- RESULTS:END -->")]
    check("the marked block was replaced with this run's tables",
          after_block != before_block and "Tesla T4 (shimmed)" in after_block)
    check("the patched block carries the measured tables",
          "### CPU vs GPU over simulation size" in after_block
          and "### RNG setup / launch-overhead fit" in after_block)
    check("prose outside the markers survived",
          all(h in after for h in ("## Profiling", "## Limitations",
                                   "## Reproducing on Kaggle", "## What this demonstrates")))

    print("\n" + "=" * 74 + "\nprofiling.md written by the ladder (for review)\n" + "=" * 74)
    print("\n".join(text.splitlines()[:70]))
finally:
    shutil.rmtree(ws, ignore_errors=True)
    shutil.rmtree(bindir, ignore_errors=True)

print("")
print("%d checks failed" % len(failures) if failures else "all checks passed")
for f in failures:
    print("  -", f)
sys.exit(1 if failures else 0)
