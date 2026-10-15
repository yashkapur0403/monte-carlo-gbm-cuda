#!/usr/bin/env python3
"""Generate notebooks/kaggle_gbm_montecarlo.ipynb from src/gbm_montecarlo.cu.

The notebook embeds the exact bytes of the CUDA source file, so a clean Kaggle
session can run the whole project from one artifact with no upload step.

Run:  python tools/build_notebook.py
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "gbm_montecarlo.cu"
OUT = ROOT / "notebooks" / "kaggle_gbm_montecarlo.ipynb"

src_text = SRC.read_text(encoding="utf-8")
# The notebook embeds the source inside a Python raw triple-quoted string.
for bad in ("'''", '"""'):
    if bad in src_text:
        sys.exit("source contains %r which breaks embedding; use base64 instead" % bad)
if src_text.endswith("\\"):
    sys.exit("source ends with a backslash; not embeddable in a raw string")

cells = []


def md(text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": text.strip("\n").splitlines(keepends=True),
    })


def code(text):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    })


# ---------------------------------------------------------------------------
md(r"""
# CUDA Monte Carlo GBM stock-price simulator (CPU baseline, custom kernel, benchmarks)

Self-contained Kaggle workflow for a GPU-computing portfolio project.

**What it does**

1. Environment check: GPU model, compute capability, `nvcc`, driver and runtime versions.
2. Compiles `src/gbm_montecarlo.cu` (CPU baseline + a custom `__global__` CUDA kernel),
   then runs `smoke` and the statistical correctness suite.
3. Soaks the GPU for ~25 s of sustained kernel load (a shared T4 starts below its steady SM
   clock, which distorts the first heavy measurement), then benchmarks CPU vs GPU across
   10k / 100k / 1M paths and CUDA block sizes 64 / 128 / 256 / 512, writing raw CSV. Rows
   that cannot be physical (end-to-end time below the kernel time, or two sweeps disagreeing
   on the same geometry) trigger one re-soak and one repeat of the sweep.
4. Plots the results, then profiles: it uses NVIDIA Nsight Systems / Compute if this image has
   them and the container permits them, and otherwise falls back to whatever works here - a
   measured CUDA-event run of the real kernel, static kernel statistics read out of the built
   cubin (`cuobjdump`, `ptxas`), the occupancy derived from them, and a sampled `nvidia-smi`
   telemetry trace. Every finding is labelled measured / static / estimated in
   `results/profiling.md`, together with what is *not* available and why.
5. Verifies the whole result set (every artifact present, valid CSV, no failed check) and only
   then packages it into `gbm_montecarlo_results.zip` and offers the files for download.

**Model** (only the terminal price is stored; one CUDA thread per independent path):

    S(t+dt) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z),   Z ~ N(0,1),  dt = T/steps

**Requirements:** a CUDA-capable GPU (Kaggle GPU session: T4 = `sm_75`, P100 = `sm_60`) and the
CUDA toolkit (`nvcc`). No extra `pip install` is needed on a Kaggle GPU image.

Everything is written under the current working directory: `src/`, `build/`, `results/`.

Run the cells in order (**Run All**). Cell 5 refuses to download anything unless cells 1-4
produced a complete, passing result set.
""")

# ---------------------------------------------------------------------------
code(r"""
# ============================== 1. Environment ==============================
import json, os, pathlib, re, shutil, subprocess, sys, time

RUN_START = time.time()  # cell 5 compares artifact timestamps against this

ROOT = pathlib.Path.cwd()
for d in ("src", "results", "build"):
    (ROOT / d).mkdir(parents=True, exist_ok=True)
print("working dir :", ROOT)
print("python      :", sys.version.split()[0])

ENV_LOG = []


def run(cmd, quiet=False, timeout=1800):
    '''Run a command, echo it, return CompletedProcess (stdout+stderr captured).'''
    printable = cmd if isinstance(cmd, str) else " ".join(cmd)
    if not quiet:
        print("$", printable, flush=True)
    p = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True,
                       timeout=timeout)
    if not quiet:
        if p.stdout.strip():
            print(p.stdout.rstrip())
        if p.stderr.strip():
            print(p.stderr.rstrip()[:3000])
    return p


def banner(t):
    print("\n" + "=" * 74 + "\n" + t + "\n" + "=" * 74)


# ---- GPU -------------------------------------------------------------------
smi = run(["nvidia-smi"], quiet=False)
ENV_LOG.append("$ nvidia-smi\n" + (smi.stdout or "") + (smi.stderr or ""))

GPU_NAME, ARCH = None, None
q = run(["nvidia-smi", "--query-gpu=name,compute_cap,memory.total,driver_version",
         "--format=csv,noheader"], quiet=False)
ENV_LOG.append("$ nvidia-smi --query-gpu=...\n" + (q.stdout or "") + (q.stderr or ""))
try:
    first = [r for r in q.stdout.strip().splitlines() if r.strip()][0]
    parts = [x.strip() for x in first.split(",")]
    GPU_NAME = parts[0]
    if len(parts) > 1 and "." in parts[1]:
        major, minor = parts[1].split(".")
        ARCH = "sm_%s%s" % (major, minor)
except Exception as exc:  # pragma: no cover - driver dependent
    print("compute_cap query failed:", exc)

if ARCH is None:
    try:
        import torch
        cap = torch.cuda.get_device_capability(0)
        ARCH = "sm_%d%d" % (cap[0], cap[1])
        GPU_NAME = torch.cuda.get_device_name(0)
        print("arch via torch:", ARCH)
    except Exception as exc:
        print("torch fallback failed:", exc)

# ---- CPU -------------------------------------------------------------------
cpu_model, n_cpu = "", str(os.cpu_count())
try:
    for line in open("/proc/cpuinfo"):
        if line.lower().startswith("model name"):
            cpu_model = line.split(":", 1)[1].strip()
            break
except OSError:
    pass
n_cpu = os.cpu_count()
print("\nGPU        :", GPU_NAME, "| arch:", ARCH)
print("CPU        :", cpu_model, "| logical cores:", n_cpu)

# ---- CUDA toolkit ----------------------------------------------------------
NVCC = shutil.which("nvcc")
if NVCC is None and pathlib.Path("/usr/local/cuda/bin/nvcc").exists():
    NVCC = "/usr/local/cuda/bin/nvcc"
EXTRA_FLAGS = []

if NVCC is None:
    print("\nnvcc not found in the image; trying the pip CUDA toolchain fallback ...")
    run([sys.executable, "-m", "pip", "install", "-q", "nvidia-cuda-nvcc-cu12",
         "nvidia-cuda-runtime-cu12", "nvidia-curand-cu12"])
    import site
    search = list(site.getsitepackages()) + [site.getusersitepackages()] + [ROOT]
    for sp in search:
        cand = pathlib.Path(sp) / "nvidia" / "cuda_nvcc" / "bin" / "nvcc"
        if cand.exists():
            NVCC = str(cand)
            break
    for sp in search:
        base = pathlib.Path(sp) / "nvidia"
        if not base.exists():
            continue
        for inc in ("cuda_runtime/include", "curand/include", "cuda_nvcc/include"):
            if (base / inc).exists():
                EXTRA_FLAGS += ["-I", str(base / inc)]
        for lib in ("cuda_runtime/lib", "curand/lib"):
            if (base / lib).exists():
                EXTRA_FLAGS += ["-L", str(base / lib)]

assert NVCC is not None, "no nvcc available - use a Kaggle GPU image (or install the CUDA toolkit)"
ver = run([NVCC, "--version"])
ENV_LOG.append("$ nvcc --version\n" + (ver.stdout or "") + (ver.stderr or ""))
assert ARCH is not None, "could not determine the GPU compute capability"

print("\nnvcc       :", NVCC)
print("arch flag  : -arch=%s" % ARCH)
print("extra flags:", EXTRA_FLAGS or "(none)")

ENV_LOG.append("GPU: %s\narch: %s\nCPU: %s (%s logical cores)\n" % (GPU_NAME, ARCH, cpu_model, n_cpu))
(pathlib.Path("results") / "environment.txt").write_text(
    "=== nvidia-smi ===\n" + (smi.stdout or smi.stderr) + "\n" + "\n".join(ENV_LOG))
print("\nwrote results/environment.txt")
""")

# ---------------------------------------------------------------------------
cell3 = (
    r"""
# ================== 2. Source, compile, correctness =========================
import pathlib, subprocess, time

SOURCE = r'''
"""
    + src_text
    + r"""
'''

# The raw literal above carries one extra newline at each end (it is delimited by
# r'''...''' on their own lines). Strip exactly those, so the file written here is
# byte-identical to the src/gbm_montecarlo.cu in the repository - this is the copy
# that gets archived as "the source that produced these numbers".
SOURCE = SOURCE.strip("\n") + "\n"
SRC = pathlib.Path("src/gbm_montecarlo.cu")
SRC.write_text(SOURCE, encoding="utf-8")
print("wrote %s: %d lines, %d bytes" % (SRC, len(SOURCE.splitlines()), len(SOURCE.encode())))

CMD = [NVCC, "-O3", "-std=c++17", "-arch=%s" % ARCH, "-lineinfo", "-Xptxas", "-v",
       "-o", "build/gbm", "src/gbm_montecarlo.cu", "-lcurand"] + EXTRA_FLAGS
t0 = time.time()
p = subprocess.run(CMD, capture_output=True, text=True)
compile_log = "$ " + " ".join(CMD) + "\n\n" + p.stdout + p.stderr
pathlib.Path("results/compile_log.txt").write_text(compile_log)
print("compile rc = %d in %.1f s" % (p.returncode, time.time() - t0))
print(p.stdout[-6000:])
print(p.stderr[-6000:])
assert p.returncode == 0, "nvcc failed - see results/compile_log.txt and the output above"
assert pathlib.Path("build/gbm").exists(), "no binary produced"
print("\n[ptxas resource usage is recorded in results/compile_log.txt]")

banner("device information")
info = run(["./build/gbm", "info", "--out", "results"], quiet=False)

banner("smoke test (must pass before the benchmark matrix)")
smoke = run(["./build/gbm", "smoke", "--out", "results"])
pathlib.Path("results/smoke_stdout.txt").write_text(smoke.stdout + smoke.stderr)
assert smoke.returncode == 0, "smoke test failed"

banner("numerical correctness suite")
verify = run(["./build/gbm", "verify", "--paths", "200000", "--seeds", "3",
              "--steps", "252", "--out", "results"])
pathlib.Path("results/verify_stdout.txt").write_text(verify.stdout + verify.stderr)
print("\nverify exit code:", verify.returncode)
print("checks written to results/correctness.csv")
"""
)
code(cell3)

# ---------------------------------------------------------------------------
code(r"""
# ========================= 3. Benchmarks (CSV) ==============================
import pathlib, time

# ---- steady-state guard ----------------------------------------------------
# A shared Kaggle GPU starts below its steady SM clock: in an earlier session the same
# 1000000-path/block-256 geometry measured 6.167 ms (first heavy run of the session) and
# 3.442 ms later, and one row even reported an end-to-end time below its own kernel time,
# which is only possible while the clock is still rising. Both are measurement artefacts,
# not kernel behaviour, so every timed sweep below is preceded by a sustained untimed soak
# of the real kernel, and the sweeps are re-run once if the result is still self-
# inconsistent. The soak writes to build/, never to results/.
SOAK_SECONDS = 25.0
SOAK_DIR = "build/soak"


def clock_soak(seconds=SOAK_SECONDS):
    banner("clock soak: sustained kernel load before any timed measurement")
    t0 = time.time()
    while time.time() - t0 < seconds:
        run(["./build/gbm", "rng", "--paths", "1000000", "--block", "256",
             "--steps", "252", "--reps", "10", "--out", SOAK_DIR], quiet=True)
    elapsed = time.time() - t0
    print("soak: %.1f s of sustained kernel load (steady-state clocks)" % elapsed)
    return elapsed


# Rows that cannot be real measurements, so the sweep has to be repeated.
def sweep_anomalies():
    import csv

    def rows(name):
        with open("results/%s.csv" % name) as fh:
            return list(csv.DictReader(fh))

    bad = []
    sizes, blocks = rows("benchmark_sizes"), rows("benchmark_blocks")
    for r in sizes + blocks:
        k, e = float(r["gpu_kernel_ms"]), float(r["gpu_e2e_ms"])
        if e < k:
            bad.append("paths=%s block=%s: e2e %.3f ms < kernel %.3f ms (clock still rising)"
                       % (r["paths"], r["block"], e, k))
    # the same geometry is measured in both sweeps: they must agree to within a few percent
    ref = [r for r in sizes if r["block"] == "256"]
    for r in ref:
        same = [b for b in blocks if b["paths"] == r["paths"] and b["block"] == "256"]
        if not same:
            continue
        k1, k2 = float(r["gpu_kernel_ms"]), float(same[0]["gpu_kernel_ms"])
        if k1 > 0 and abs(k2 - k1) / k1 > 0.25:
            bad.append("1M-scale check paths=%s: %.3f ms vs %.3f ms in the other sweep (%.0f%% apart)"
                       % (r["paths"], k1, k2, 100.0 * abs(k2 - k1) / k1))
    return bad


def run_bench_sweep():
    t0 = time.time()
    bench = run(["./build/gbm", "bench",
                 "--sizes", "10000,100000,1000000",
                 "--blocks", "64,128,256,512",
                 "--steps", "252",
                 "--reps", "5",
                 "--cpu-reps", "3",
                 "--ref-block", "256",
                 "--out", "results"])
    pathlib.Path("results/bench_stdout.txt").write_text(bench.stdout + bench.stderr)
    print("\nbench wall time: %.1f s (rc=%d)" % (time.time() - t0, bench.returncode))
    assert bench.returncode == 0, "benchmark run failed"


clock_soak()
banner("benchmark matrix: sizes x block sizes")
run_bench_sweep()
anomalies = sweep_anomalies()
if anomalies:
    print("\n[steady-state guard] the sweep measured something that cannot be physical:")
    for a in anomalies:
        print("   - %s" % a)
    print("   re-soaking the clocks and repeating the sweep once; the repeat is kept.")
    clock_soak()
    run_bench_sweep()
    remaining = sweep_anomalies()
    if remaining:
        print("\n[steady-state guard] still inconsistent after the repeat - reporting as measured:")
        for a in remaining:
            print("   - %s" % a)
        print("   this session's clocks never settled; see the README caveat.")
    else:
        print("\n[steady-state guard] repeat is self-consistent; keeping it.\n")
else:
    print("\n[steady-state guard] all rows self-consistent (e2e >= kernel, both sweeps agree).\n")

banner("RNG setup / launch-overhead study (step sweep + OLS fit)")
rng = run(["./build/gbm", "rng",
           "--paths", "1000000",
           "--block", "256",
           "--steps", "1,4,16,64,252",
           "--reps", "5",
           "--out", "results"])
pathlib.Path("results/rng_stdout.txt").write_text(rng.stdout + rng.stderr)

print("\nfiles produced:")
for f in sorted(pathlib.Path("results").glob("*")):
    print("  %-34s %9d bytes" % (f.name, f.stat().st_size))
""")

# ---------------------------------------------------------------------------
code(r"""
# ============= 4. Plots, profiling attempt, result summary ==================
import os, pathlib, shutil, subprocess, textwrap
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RES = pathlib.Path("results")
sizes = pd.read_csv(RES / "benchmark_sizes.csv")
blocks = pd.read_csv(RES / "benchmark_blocks.csv")
sizes_g = sizes[sizes.gpu_kernel_ms > 0].copy()
blocks_g = blocks[blocks.gpu_kernel_ms > 0].copy()

# ---- plot 1: CPU vs GPU time over simulation size -------------------------
fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
ax.plot(sizes_g.paths, sizes_g.cpu_ms, "o-", lw=2, color="#c0392b",
        label="CPU baseline, 1 thread (mt19937_64 + Box-Muller)")
ax.plot(sizes_g.paths, sizes_g.gpu_kernel_ms, "s-", lw=2, color="#1f6feb",
        label="GPU kernel only (CUDA events)")
ax.plot(sizes_g.paths, sizes_g.gpu_e2e_ms, "^--", lw=2, color="#2e8b57",
        label="GPU end-to-end (malloc + launch + D2H copy + free)")
for _, r in sizes_g.iterrows():
    ax.annotate("%.1fx" % r.speedup_kernel, (r.paths, r.gpu_kernel_ms),
                textcoords="offset points", xytext=(6, -14), fontsize=8, color="#1f6feb")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Monte Carlo paths (252 steps, T = 1 year, S0 = 100, mu = 0.05, sigma = 0.20)")
ax.set_ylabel("wall time [ms]")
ax.set_title("Monte Carlo GBM: CPU vs custom CUDA kernel")
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(RES / "plot_cpu_vs_gpu_time.png")
plt.close(fig)

# ---- plot 2: block-size sweep --------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=140)
for n, grp in blocks_g.groupby("paths"):
    g = grp.sort_values("block")
    axes[0].plot(g.block, g.gpu_kernel_ms, "o-", label="%s paths" % format(int(n), ","))
    axes[1].plot(g.block, g.gpu_paths_per_sec / 1e6, "o-", label="%s paths" % format(int(n), ","))
axes[0].set_xlabel("CUDA block size [threads]")
axes[0].set_ylabel("kernel time [ms]  (median of 5, CUDA events)")
axes[0].set_title("Kernel time vs block size, 252 steps")
axes[0].set_xscale("log", base=2)
axes[0].set_xticks(sorted(blocks_g.block.unique()))
axes[0].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
axes[1].set_xlabel("CUDA block size [threads]")
axes[1].set_ylabel("throughput [Million paths/s]")
axes[1].set_title("Terminal-price throughput vs block size")
axes[1].set_xscale("log", base=2)
axes[1].set_xticks(sorted(blocks_g.block.unique()))
axes[1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
for a in axes:
    a.grid(True, alpha=0.3)
    a.legend(fontsize=8)
fig.tight_layout()
fig.savefig(RES / "plot_block_sizes.png")
plt.close(fig)

# ---- plot 3: speedup -----------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4), dpi=140)
x = np.arange(len(sizes_g))
w = 0.36
ax.bar(x - w / 2, sizes_g.speedup_kernel, w, label="kernel-only speedup", color="#1f6feb")
ax.bar(x + w / 2, sizes_g.speedup_e2e, w, label="end-to-end speedup", color="#2e8b57")
ax.set_xticks(x)
ax.set_xticklabels([format(int(p), ",") for p in sizes_g.paths])
ax.set_xlabel("Monte Carlo paths")
ax.set_ylabel("CPU time / GPU time  [x]")
ax.set_title("Speedup over the single-threaded CPU baseline (252 steps)")
ax.axhline(1.0, color="k", lw=1, ls=":")
ax.grid(True, axis="y", alpha=0.3)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(RES / "plot_speedup.png")
plt.close(fig)
print("wrote results/plot_cpu_vs_gpu_time.png, plot_block_sizes.png, plot_speedup.png")

# ---- profiling -----------------------------------------------------------
# A ladder that runs entirely inside this Kaggle runtime: use Nsight if the image
# ships it AND the container permits it, otherwise fall back to the strongest
# evidence that does work here -- a measured CUDA-event run of the real kernel,
# static kernel statistics read out of the built cubin (cuobjdump / ptxas), the
# occupancy that follows from them, and a sampled nvidia-smi telemetry trace.
# This section never touches the simulation: it reads the cubin, its compile log
# and the device properties, and it writes into results/profile/ so the benchmark
# CSVs written by cell 3 are left exactly as they are.
import re, threading, time

KERNEL = "gbm_terminal_price_kernel"
REP = (1000000, 256, 252, 3)  # paths, block, steps, reps - the representative run
PROFILE_DIR = pathlib.Path("results/profile")
PROFILE_DIR.mkdir(parents=True, exist_ok=True)


def find_tool(name):
    p = shutil.which(name)
    if p:
        return p
    for d in ("/usr/local/cuda/bin", "/opt/nvidia/nsight-systems/bin",
              "/opt/nvidia/nsight-compute/bin", "/usr/local/bin", "/usr/bin"):
        cand = pathlib.Path(d) / name
        if cand.exists():
            return str(cand)
    return None


def run_tool(label, cmd, timeout=900, cap=6000):
    '''Run a tool, echo it, return (text, returncode). Never raises.'''
    print("\n$ " + " ".join(cmd), flush=True)
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        out = (p.stdout or "") + (p.stderr or "")
        print(out[-cap:])
        return out, p.returncode
    except subprocess.TimeoutExpired:
        msg = "%s: timed out after %d s\n" % (label, timeout)
        print(msg)
        return msg, -1
    except OSError as exc:
        msg = "%s: could not be executed (%s)\n" % (label, exc)
        print(msg)
        return msg, -1


# ---- pure helpers (no GPU state, so they can be unit-tested off-device) -----
def _num(text):
    '''"1 590 MHz" / "45 %" / "25.30 W" -> float, or None when unsupported.'''
    t = text.strip()
    for unit in ("%", "MHz", "W", "MiB", "C"):
        t = t.replace(unit, "")
    try:
        return float(t.strip())
    except ValueError:
        return None


def _smi_ts(text):
    '''`2026/10/15 09:12:00.150` -> epoch seconds, keeping the milliseconds.

    time.strptime drops the fractional part, so it is parsed separately: the
    sampler runs at 50 ms, and without it every sample would land in the same
    second and the idle/busy split would collapse.
    '''
    t = text.strip()
    frac = 0.0
    if "." in t:
        t, _, us = t.partition(".")
        frac = float("0." + us)
    try:
        return time.mktime(time.strptime(t, "%Y/%m/%d %H:%M:%S")) + frac
    except ValueError:
        return None


def parse_ptxas_v(text):
    '''Register / smem / spill usage out of the nvcc -Xptxas -v compile log.'''
    info = {}
    for key, pat in (("registers_per_thread", r"Used\s+(\d+)\s+registers"),
                     ("static_smem_bytes", r"(\d+)\s+bytes\s+smem"),
                     ("spill_stores_bytes", r"(\d+)\s+bytes\s+spill stores"),
                     ("spill_loads_bytes", r"(\d+)\s+bytes\s+spill loads"),
                     # CUDA 11+ prints "0 bytes stack frame"; older ptxas printed
                     # "0 bytes cumulative stack size" - accept either.
                     ("stack_bytes", r"(\d+)\s+bytes\s+(?:stack frame|cumulative stack size)")):
        m = re.search(pat, text)
        if m:
            info[key] = int(m.group(1))
    return info


def parse_res_usage(text):
    '''REG/STACK/SHARED/LOCAL/... from `cuobjdump -res-usage`.

    Tolerates both `REG:26` and `REG: 26`, keeps only numeric fields, and gives up
    quietly (empty dict) if the report is not there.
    '''
    for line in text.splitlines():
        if "REG" not in line or ":" not in line:
            continue
        out = {}
        for k, v in re.findall(r"([A-Z][A-Z0-9_\[\]]*)\s*:\s*([0-9]+)", line):
            out[k] = int(v)
        if out:
            return out
    return {}


def census_sass(text, func=KERNEL):
    '''Static SASS instruction census for one kernel: (total, {opcode: count}).'''
    counts, total, inside = {}, 0, False
    for line in text.splitlines():
        if "Function :" in line:
            inside = func in line
            continue
        if not inside:
            continue
        m = re.match(r"\s*(?:[A-Za-z_.$][\w.$]*:\s*)?/\*[0-9a-f]{4,}\*/\s+"
                     r"(?:@!?U?PT?\d\s+)?([A-Z][A-Z0-9._]*)", line)
        if m:
            counts[m.group(1)] = counts.get(m.group(1), 0) + 1
            total += 1
    return total, counts


def parse_smi_trace(text):
    '''Rows out of `nvidia-smi --query-gpu=... --format=csv -lms N`.'''
    rows = []
    for line in text.splitlines():
        if not line.strip() or line.lower().startswith("timestamp"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        row = {"ts": parts[0], "ts_epoch": _smi_ts(parts[0]), "gpu_util": _num(parts[1]),
               "mem_util": _num(parts[2]), "sm_mhz": _num(parts[3]), "mem_mhz": _num(parts[4]),
               "power_w": _num(parts[5]),
               "temp_c": _num(parts[6]) if len(parts) > 6 else None}
        if row["gpu_util"] is None:
            continue
        rows.append(row)
    return rows


def occupancy(regs_per_thread, block, props):
    '''Theoretical occupancy for block threads per block, one thread per path.'''
    if not regs_per_thread or not props.get("sms"):
        return {}
    regs_per_block = ((regs_per_thread * 32 + 255) // 256) * 256 * (block // 32)
    limits = {
        "registers": props["regs_per_sm"] // regs_per_block if regs_per_block else 0,
        "threads": props["max_threads_per_sm"] // block,
        "block slots": 32,
    }
    resident = max(0, min(limits.values()))
    warps_per_sm = props["max_threads_per_sm"] // max(1, props.get("warp_size", 32))
    return {"regs_per_block": regs_per_block, "limits": limits,
            "limiter": min(limits, key=lambda k: limits[k]),
            "resident_blocks_per_sm": resident,
            "resident_warps_per_sm": resident * (block // 32),
            "warps_per_sm": warps_per_sm,
            "theoretical_occupancy_pct": 100.0 * resident * (block // 32) / warps_per_sm}


# ---- device properties (already dumped by ./build/gbm info in cell 3) -------
props = {}
env_file = RES / "environment_cuda.txt"
if env_file.exists():
    env_txt = env_file.read_text(encoding="utf-8", errors="replace")
    for key, pat in (("device", r"device\s*:\s*(.+)"),
                     ("cc", r"compute capability\s*:\s*(\d+\.\d+)"),
                     ("sms", r"multiprocessors \(SMs\)\s*:\s*(\d+)"),
                     ("max_threads_per_sm", r"max threads per SM\s*:\s*(\d+)"),
                     ("max_threads_per_block", r"max threads per block\s*:\s*(\d+)"),
                     ("smem_per_block", r"shared memory per block\s*:\s*(\d+)"),
                     ("warp_size", r"warp size\s*:\s*(\d+)"),
                     ("regs_per_sm", r"registers per SM\s*:\s*(\d+)"),
                     ("regs_per_thread", r"registers per thread\s*:\s*(\d+)"),
                     ("l2_bytes", r"L2 cache size\s*:\s*(\d+)"),
                     ("mem_clock_mhz", r"memory clock rate\s*:\s*([\d.]+)")):
        m = re.search(pat, env_txt)
        if not m:
            continue
        v = m.group(1).strip()
        props[key] = v if key == "device" else (float(v) if "." in v else int(v))

# ---- tool discovery --------------------------------------------------------
TOOLS = {}
for name in ("nsys", "ncu", "nvprof", "cuobjdump", "nvdisasm", "ptxas", "nvidia-smi"):
    TOOLS[name] = find_tool(name)

prof_lines = ["# Profiling evidence", "",
              "GPU: %s (%s)%s | nvcc: %s | runtime: %s"
              % (GPU_NAME, ARCH,
                 (", %d SMs, %d threads/SM, %d regs/SM" % (props["sms"],
                                                           props["max_threads_per_sm"],
                                                           props["regs_per_sm"]))
                 if props.get("sms") else "", NVCC, sys.version.split()[0]), "",
              "Representative workload: `./build/gbm rng --paths %d --block %d --steps %d "
              "--reps %d` (the real kernel, 252 Euler steps per path, terminal price only)."
              % (REP[0], REP[1], REP[2], REP[3]), "",
              "Every number below is labelled **measured** (a tool produced it here), **static** "
              "(read out of the built cubin) or **estimated** (derived, with its assumption "
              "stated). Nothing here is a CUPTI hardware-counter result unless Nsight actually "
              "ran, which is reported in section 2.", "",
              "## 1. Tool discovery in this runtime", "",
              "| tool | resolved path |", "|---|---|"]
for name in sorted(TOOLS):
    prof_lines.append("| `%s` | %s |" % (name, TOOLS[name] or "not found"))
prof_lines.append("")
print("\ntool discovery:", {k: bool(v) for k, v in TOOLS.items()})

# ---- 2. Nsight, if this image actually has it ------------------------------
raw_files = []

if TOOLS["nsys"]:
    cmd = [TOOLS["nsys"], "profile", "--force-overwrite=true", "--stats=true", "-o",
           "results/nsys_gbm", "./build/gbm", "rng", "--paths", str(REP[0]), "--block",
           str(REP[1]), "--steps", str(REP[2]), "--reps", "1", "--out", str(PROFILE_DIR)]
    out, rc = run_tool("nsys", cmd, timeout=900, cap=4000)
    pathlib.Path("results/nsys_stdout.txt").write_text(out)
    raw_files.append("results/nsys_stdout.txt")
    prof_lines += ["## 2. Nsight Systems (`nsys`) - present, exit code %d" % rc, "",
                   "```", " ".join(cmd), "```", "", "```", out[-4000:], "```", ""]
    prof_lines += (["- **Nsight Systems produced a usable timeline.**"] if rc == 0 else
                   ["- **Nsight Systems did not produce results here**; see the output above."])
    prof_lines.append("")
else:
    prof_lines += ["## 2. Nsight Systems (`nsys`) - not in this image", "",
                   "`nsys` is absent, so no CUPTI timeline could be captured here. The commands "
                   "that produce one on a machine that has it are in section 6.", ""]
    print("nsys not available in this environment")

if TOOLS["ncu"]:
    cmd = [TOOLS["ncu"], "--set", "basic", "--launch-count", "1", "--kernel-name",
           "regex:" + KERNEL, "./build/gbm", "rng", "--paths", str(REP[0]), "--block",
           str(REP[1]), "--steps", str(REP[2]), "--reps", "1", "--out", str(PROFILE_DIR)]
    out, rc = run_tool("ncu", cmd, timeout=900, cap=6000)
    pathlib.Path("results/ncu_stdout.txt").write_text(out)
    raw_files.append("results/ncu_stdout.txt")
    prof_lines += ["## 3. Nsight Compute (`ncu`) - present, exit code %d" % rc, "",
                   "```", " ".join(cmd), "```", ""]
    if "ERR_NVGPUCTRPERM" in out:
        prof_lines += ["**The container denies GPU performance counters** (verbatim):", "",
                       "```", "\n".join(l for l in out.splitlines()
                                      if "ERROR" in l or "NVGPUCTRPERM" in l), "```", "",
                       "So `ncu` cannot be used inside Kaggle: reading those counters needs "
                       "`CAP_SYS_ADMIN` / `NVreg_RestrictProfilingToAdminUsers=0`. Section 4 "
                       "replaces it with evidence that the kernel *can* provide here.", ""]
    else:
        prof_lines += ["```", out[-6000:], "```", ""]
else:
    prof_lines += ["## 3. Nsight Compute (`ncu`) - not in this image", "",
                   "`ncu` is absent or unusable here, so no hardware counters were read.", ""]

# ---- 4. What does work here -----------------------------------------------
prof_lines += ["## 4. Kaggle-compatible kernel profile (everything below ran here)", ""]
try:
    # 4a. measured: one representative kernel run, CUDA events, kernel-only timing
    rep_cmd = ["./build/gbm", "rng", "--paths", str(REP[0]), "--block", str(REP[1]),
               "--steps", str(REP[2]), "--reps", str(REP[3]), "--out", str(PROFILE_DIR)]
    rep_out, rep_rc = run_tool("representative kernel run", rep_cmd, timeout=900)
    rep = {}
    rep_csv = PROFILE_DIR / "rng_overhead.csv"
    if rep_csv.exists():
        rep = pd.read_csv(rep_csv).iloc[0].to_dict()
    prof_lines += ["### 4a. Representative run - **measured** (CUDA events, kernel only)", "",
                   "```", " ".join(rep_cmd), "```", ""]
    if rep:
        sec = float(rep.get("kernel_ms", 0.0)) / 1000.0
        exps = REP[0] * REP[2] / sec if sec > 0 else 0.0
        prof_lines += ["| kernel ms | us / step | ns / path / step | MPaths/s | expf calls | expf/s |",
                       "|---|---|---|---|---|---|",
                       "| %.4f | %.4f | %.4f | %.2f | %d | %.3e |"
                       % (float(rep.get("kernel_ms", 0.0)), float(rep.get("us_per_step", 0.0)),
                          float(rep.get("ns_per_path_step", 0.0)),
                          float(rep.get("gpu_paths_per_sec", 0.0)) / 1e6,
                          REP[0] * REP[2], exps), "",
                       "`expf` calls are exact (%d paths x %d steps - one `expf` per Euler step), "
                       "so the expf/s column is a directly measured throughput; the time is the "
                       "median of %d CUDA-event runs and excludes the host-side copy."
                       % (REP[0], REP[2], REP[3]), ""]
        print("representative run: kernel %.4f ms, %.3e expf/s" % (float(rep.get("kernel_ms", 0.0)), exps))
    else:
        prof_lines += ["**The representative run wrote no CSV** (exit code %d); its console output "
                       "is above." % rep_rc, ""]

    # 4b. static: kernel resources, from the cubin that cell 3 produced
    res_usage, ptxas = "", {}
    if TOOLS["cuobjdump"]:
        res_usage, _ = run_tool("cuobjdump -res-usage",
                               [TOOLS["cuobjdump"], "-res-usage", "build/gbm"], timeout=300)
        pathlib.Path("results/cuobjdump_res_usage.txt").write_text(res_usage)
        raw_files.append("results/cuobjdump_res_usage.txt")
    compile_log = RES / "compile_log.txt"
    if compile_log.exists():
        ptxas = parse_ptxas_v(compile_log.read_text(encoding="utf-8", errors="replace"))
    usage = parse_res_usage(res_usage) or ptxas
    regs = int(usage.get("REG", usage.get("registers_per_thread", props.get("regs_per_thread", 0))) or 0)
    prof_lines += ["### 4b. Kernel resource usage - **static** (from the built cubin)", ""]
    if usage:
        prof_lines += ["| registers / thread | static smem | stack | spill stores | spill loads | source |",
                       "|---|---|---|---|---|---|",
                       "| %s | %s | %s | %s | %s | %s |"
                       % (usage.get("REG", ptxas.get("registers_per_thread", "?")),
                          usage.get("SHARED", ptxas.get("static_smem_bytes", "?")),
                          usage.get("STACK", ptxas.get("stack_bytes", "?")),
                          ptxas.get("spill_stores_bytes", 0) if ptxas else "?",
                          ptxas.get("spill_loads_bytes", 0) if ptxas else "?",
                          "cuobjdump -res-usage" if parse_res_usage(res_usage) else "nvcc -Xptxas -v"),
                       "",
                       "No spill and no local memory means the inner loop lives entirely in "
                       "registers, which is what a one-thread-per-path stencil like this needs.", ""]
    else:
        prof_lines += ["Neither `cuobjdump -res-usage` nor `-Xptxas -v` produced register counts "
                       "here.", ""]

    # 4c. static: SASS instruction census
    census = {}
    if TOOLS["cuobjdump"]:
        sass, _ = run_tool("cuobjdump -sass", [TOOLS["cuobjdump"], "-sass", "build/gbm"],
                           timeout=600, cap=400)
        total, census = census_sass(sass)
        if census:
            with open("results/sass_census.csv", "w") as fh:
                fh.write("opcode,count\n")
                for op, c in sorted(census.items(), key=lambda kv: -kv[1]):
                    fh.write("%s,%d\n" % (op, c))
            raw_files.append("results/sass_census.csv")
            prof_lines += ["### 4c. SASS instruction census - **static** (`cuobjdump -sass`)", "",
                           "%d SASS instructions in `%s`." % (total, KERNEL), "",
                           "| opcode | count | share |", "|---|---|---|"]
            for op, c in sorted(census.items(), key=lambda kv: -kv[1])[:12]:
                prof_lines.append("| `%s` | %d | %.1f%% |" % (op, c, 100.0 * c / total))
            prof_lines += ["",
                           "These are instruction counts in the *compiled kernel body as written*, "
                           "not executed counts: `#pragma unroll 4` (and the `curand_normal2` "
                           "pairing) means the loop body in the SASS is several Euler steps wide "
                           "and executes many times per path. Nothing on this line divides by the "
                           "logical 252 steps, because that ratio would be an artefact of the "
                           "unroll factor rather than a measurement.", ""]
        else:
            prof_lines += ["`cuobjdump -sass` ran but no SASS was found for `%s`." % KERNEL, ""]

    # 4d. derived: occupancy / launch geometry
    occ = occupancy(regs, REP[1], props)
    if occ:
        grid_blocks = (REP[0] + REP[1] - 1) // REP[1]
        waves = grid_blocks / float(props["sms"] * max(1, occ["resident_blocks_per_sm"]))
        prof_lines += ["### 4d. Occupancy and launch geometry - **derived** from 4b/4c", "",
                       "| thread / path | regs / thread | block | regs / block | resident blocks / SM | "
                       "resident warps / SM | warps / SM | theoretical occupancy | limiter |",
                       "|---|---|---|---|---|---|---|---|---|",
                       "| 1 | %d | %d | %d | %d | %d | %d | %.1f%% | %s |"
                       % (regs, REP[1], occ["regs_per_block"], occ["resident_blocks_per_sm"],
                          occ["resident_warps_per_sm"], occ["warps_per_sm"],
                          occ["theoretical_occupancy_pct"], occ["limiter"]), "",
                       "| launch (representative run) | blocks | waves | blocks / SM |",
                       "|---|---|---|---|",
                       "| %d paths, block %d | %d | %.1f | %.1f |"
                       % (REP[0], REP[1], grid_blocks, waves,
                          grid_blocks / float(props["sms"])), "",
                       "Register allocation is rounded to the 256-register warp granularity. "
                       "Occupancy is the *theoretical* figure the hardware allows for this "
                       "kernel and block size - without `ncu` it cannot be compared against the "
                       "achieved occupancy, and that gap is stated in section 5.", ""]
    else:
        prof_lines += ["### 4d. Occupancy - not computable here", "",
                       "Needs a register count (4b) and device properties from "
                       "`results/environment_cuda.txt`; one of the two was missing.", ""]

    # 4e. measured: sampled nvidia-smi telemetry during a soak run
    if TOOLS["nvidia-smi"]:
        q = ("timestamp,utilization.gpu,utilization.memory,clocks.sm,clocks.mem,"
             "power.draw,temperature.gpu")
        smi = subprocess.Popen([TOOLS["nvidia-smi"], "--query-gpu=" + q, "--format=csv",
                                "-lms", "50"], stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL, text=True)
        lines = []
        # Read the sampler's pipe on a thread: nvidia-smi's own buffering would
        # otherwise lose the tail of the trace when the sampler is stopped.
        reader = threading.Thread(target=lambda: [lines.append(l) for l in smi.stdout],
                                  daemon=True)
        reader.start()
        marker, busy_end = None, None
        try:
            time.sleep(1.0)  # pre-soak window (not guaranteed idle on a shared GPU)
            marker = time.time()
            soak = ["./build/gbm", "rng", "--paths", str(REP[0]), "--block", str(REP[1]),
                    "--steps", str(REP[2]), "--reps", "150", "--out", str(PROFILE_DIR)]
            run_tool("soak run for the telemetry trace", soak, timeout=900)
            busy_end = time.time()
            time.sleep(0.5)
        finally:
            smi.terminate()
            try:
                smi.wait(timeout=30)
            except subprocess.TimeoutExpired:
                smi.kill()
            reader.join(timeout=10)
        pathlib.Path("results/nvidia_smi_trace.csv").write_text("".join(lines))
        raw_files.append("results/nvidia_smi_trace.csv")
        trace = parse_smi_trace("".join(lines))
        if marker is not None and busy_end is not None:
            before = [r for r in trace if r["ts_epoch"] is None or r["ts_epoch"] < marker]
            soak = [r for r in trace if r["ts_epoch"] is not None
                    and marker <= r["ts_epoch"] <= busy_end]
        else:
            before, soak = [], trace
        if not soak:  # unparsable or missing timestamps: split on utilisation instead
            soak = [r for r in trace if (r["gpu_util"] or 0) >= 20]
            before = [r for r in trace if (r["gpu_util"] or 0) < 20]

        def _stat(rows, key, how):
            vals = [r[key] for r in rows if r.get(key) is not None]
            if not vals:
                return "n/a"
            a = np.array(vals, dtype=float)
            return "%.0f" % {"mean": a.mean(), "median": float(np.median(a)),
                             "peak": a.max()}[how]

        def _row(label, rows):
            return ("| %s | %d | %s%% | %s%% | %s%% | %s%% | %s MHz | %s W | %s W | %s C |"
                    % (label, len(rows), _stat(rows, "gpu_util", "mean"),
                       _stat(rows, "gpu_util", "median"), _stat(rows, "gpu_util", "peak"),
                       _stat(rows, "mem_util", "peak"), _stat(rows, "sm_mhz", "peak"),
                       _stat(rows, "power_w", "mean"), _stat(rows, "power_w", "peak"),
                       _stat(rows, "temp_c", "peak")))

        prof_lines += ["### 4e. Sampled GPU telemetry around the kernel - **measured**", "",
                       "`nvidia-smi --query-gpu=... --format=csv -lms 50` sampled across two "
                       "windows: the second before a soak run (`--reps 150`) of the same kernel "
                       "geometry starts, and the soak itself. Two caveats stated up front: this "
                       "is a *sampled* trace of the GPU's own counters, not a CUPTI timeline, so "
                       "it cannot show individual launches; and on a shared Kaggle GPU "
                       "`utilization.gpu` also counts what other tenants are doing, so the "
                       "'before' window is *not* guaranteed idle and its peak is not a baseline "
                       "of our own. The informative signal is the change in the distribution "
                       "(and in power draw), not either window's peak on its own.", "",
                       "| window | samples | SM util mean | SM util median | SM util peak | "
                       "mem util peak | SM clock peak | power mean | power peak | temp peak |",
                       "|---|---|---|---|---|---|---|---|---|---|",
                       _row("before the soak (1 s)", before), _row("soak window", soak),
                       "", "Raw samples: `results/nvidia_smi_trace.csv`.", ""]
        print("telemetry: %d samples before / %d in the soak window" % (len(before), len(soak)))

    # 4f. cross-checks: does the static picture match what the run actually did?
    if census:
        mufu = sum(c for op, c in census.items() if op.startswith("MUFU"))
        local = sum(c for op, c in census.items() if op in ("LDL", "STL", "LDS", "STS"))
        ldg = sum(c for op, c in census.items() if op.startswith("LDG"))
        stg = sum(c for op, c in census.items() if op.startswith("STG"))
        expf_calls = REP[0] * REP[2]
        prof_lines += ["### 4f. Static-vs-measured cross-checks - **checked against this run**", "",
                       "| check | static observation | verdict |", "|---|---|---|",
                       "| `expf` is a hardware instruction | %d `MUFU` instruction(s) in the kernel "
                       "body, while the run performs exactly %d `expf` calls (%d paths x %d steps) "
                       "| %s |" % (mufu, expf_calls, REP[0], REP[2],
                                   "consistent" if mufu > 0 else "no MUFU: expf is emulated"),
                       "| the inner loop is register-resident | %d local/shared-memory "
                       "instruction(s) in the body | %s |"
                       % (local, "no local-memory traffic, so the loop does not spill" if local == 0
                          else "local memory traffic present"),
                       "| global traffic is one terminal store per path | %d `LDG`, %d `STG` "
                       "instruction(s) in the body | %s |"
                       % (ldg, stg, "paths live in registers and only the terminal price is "
                          "written, as designed" if stg > 0 and ldg <= 1
                          else "unexpected memory pattern"),
                       "| measured throughput the census has to support | %s | from 4a, same run |"
                       % (("%.3e paths/s" % float(rep.get("gpu_paths_per_sec", 0.0)))
                          if rep else "run produced no CSV"), "",
                       "These cross-checks are what the static data can honestly support. "
                       "Converting the census into GFLOP/s or a share of peak would require the "
                       "*executed* per-step instruction count of the unrolled loop - precisely "
                       "the data `ncu` provides and this container blocks - so it is not claimed "
                       "here.", ""]
    print("\nstatic/derived analysis done")
except Exception as exc:  # the profiling ladder must never sink the deliverables
    import traceback
    tb = traceback.format_exc()
    prof_lines += ["**The Kaggle-compatible analysis failed** - recorded verbatim so the rest of "
                   "the run is unaffected:", "", "```", tb[-3000:], "```", ""]
    print("profiling analysis failed:\n" + tb)

# ---- 5/6. limitations + how to get the real thing ---------------------------
prof_lines += ["## 5. Limitations of this profiling (stated, not hidden)", "",
               "- **No hardware-counter data.** Reading SM/L2/DRAM counters needs "
               "`CAP_SYS_ADMIN` / `NVreg_RestrictProfilingToAdminUsers=0`; inside Kaggle `ncu` "
               "returns `ERR_NVGPUCTRPERM` and `nsys` is not installed. Nothing in this file is "
               "a CUPTI hardware-counter measurement, and no section above is labelled as one.",
               "- **What replaces it:** CUDA-event kernel timing (measured), static resource and "
               "SASS statistics from the built cubin (static), occupancy derived from those two "
               "(derived), and sampled `nvidia-smi` telemetry at 50 ms (measured but coarse - it "
               "shows the device busy during the kernel phase, not individual launches).",
               "- **Achieved occupancy, cache hit rates, DRAM traffic, stall reasons and IPC are "
               "not available** without Nsight Compute. Theoretical occupancy in 4d is a hardware "
               "limit, not a measurement.",
               "- The instruction mix in 4c/4f is *static*: it counts SASS instructions, not "
               "executed instructions, so a divergent or partially predicated path would not show "
               "up in it. This kernel has no data-dependent branches, so the two coincide here.",
               "- CUDA-event timings (`benchmark_*.csv`, 4a) remain the primary timing evidence and "
               "are never presented as Nsight results.", "", "Captured raw artifacts:", ""] + \
              ["- `%s`" % f for f in raw_files] + ["- `results/profile/rng_overhead.csv` (the "
               "representative run's own output, kept out of `results/rng_overhead.csv` so the "
               "step sweep from cell 3 stays intact)", ""]

prof_lines += ["## 6. Getting the full picture on a permitted machine", "",
               "```bash", "make ARCH=sm_86",
               "# timeline: kernel launches, gaps, host/device overlap",
               "nsys profile --stats=true -o nsys_gbm ./build/gbm bench --sizes 1000000 \\",
               "    --blocks 256 --steps 252 --reps 5 --out results",
               "# counters: occupancy achieved, stalls, DRAM/L2 traffic, IPC",
               "ncu --set full --kernel-name regex:%s -o ncu_gbm \\" % KERNEL,
               "    ./build/gbm rng --paths 1000000 --block 256 --steps 252 --reps 1 --out results",
               "```", ""]

(RES / "profiling.md").write_text("\n".join(prof_lines))
print("\nwrote results/profiling.md (%d bytes)" % (RES / "profiling.md").stat().st_size)

# ---- machine-generated summary -------------------------------------------
import re

fit_txt = (RES / "rng_overhead_fit.txt").read_text().strip() \
    if (RES / "rng_overhead_fit.txt").exists() else "(no fit written)"


def md_table(df, cols, headers, fmts):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, row in df.iterrows():
        out.append("| " + " | ".join(f(row[c]) for c, f in zip(cols, fmts)) + " |")
    return "\n".join(out)


size_tbl = md_table(
    sizes_g,
    ["paths", "steps", "block", "cpu_ms", "gpu_kernel_ms", "gpu_e2e_ms",
     "speedup_kernel", "speedup_e2e", "gpu_paths_per_sec"],
    ["paths", "steps", "block", "CPU ms", "GPU kernel ms", "GPU e2e ms",
     "speedup (kernel)", "speedup (e2e)", "GPU paths/s"],
    [lambda v: format(int(v), ","), lambda v: "%d" % v, lambda v: "%d" % v,
     lambda v: "%.2f" % v, lambda v: "%.3f" % v, lambda v: "%.3f" % v,
     lambda v: "%.1fx" % v, lambda v: "%.1fx" % v, lambda v: "%.3e" % v])

block_tbl = md_table(
    blocks_g.sort_values(["paths", "block"]),
    ["paths", "block", "gpu_kernel_ms", "gpu_e2e_ms", "gpu_paths_per_sec", "reps"],
    ["paths", "block", "GPU kernel ms", "GPU e2e ms", "GPU paths/s", "reps"],
    [lambda v: format(int(v), ","), lambda v: "%d" % v, lambda v: "%.3f" % v,
     lambda v: "%.3f" % v, lambda v: "%.3e" % v, lambda v: "%d" % v])

header = ("GPU: %s (compute capability %s) | CPU: %s (%s logical cores) | nvcc: %s\n\n"
          "Simulation: S0=100, mu=0.05, sigma=0.20, T=1 year, 252 steps, terminal price only;\n"
          "timings are medians (CPU: 3 reps, GPU: 5 reps) from CUDA events on a single Kaggle GPU."
          % (GPU_NAME, ARCH.replace("sm_", ""), cpu_model, n_cpu, NVCC))

md_block = "\n".join([
    header, "",
    "### CPU vs GPU over simulation size (block size 256)", "", size_tbl, "",
    "### GPU kernel time and throughput vs CUDA block size", "", block_tbl, "",
    "### RNG setup / launch-overhead fit", "", "```", fit_txt, "```", "",
    "Raw data: `results/benchmark_sizes.csv`, `results/benchmark_blocks.csv`, "
    "`results/rng_overhead.csv`. Plots: `results/plot_cpu_vs_gpu_time.png`, "
    "`results/plot_block_sizes.png`, `results/plot_speedup.png`.", "",
])

summary = "\n".join([
    "# Measured results (generated by the notebook - do not edit by hand)", "",
    md_block, "## Files", "", "```",
    "\n".join("%-34s %9d bytes" % (f.name, f.stat().st_size) for f in sorted(RES.glob("*"))
               if f.is_file()),
    "```", "",
])
(RES / "summary.md").write_text(summary)

# Patch the README between the RESULTS markers so the repo carries real numbers.
readme = pathlib.Path("README.md")
if readme.exists():
    txt = readme.read_text(encoding="utf-8")
    if "<!-- RESULTS:START -->" in txt and "<!-- RESULTS:END -->" in txt:
        txt = re.sub(r"<!-- RESULTS:START -->.*?<!-- RESULTS:END -->",
                     "<!-- RESULTS:START -->\n" + md_block + "<!-- RESULTS:END -->",
                     txt, flags=re.S)
        readme.write_text(txt, encoding="utf-8")
        print("patched README.md between the RESULTS markers")
    else:
        print("README.md has no RESULTS markers; paste results/summary.md manually")
else:
    print("no README.md in the working directory; paste results/summary.md manually")

print("\n" + "=" * 74)
print("MEASURED RESULTS (also written to results/summary.md)")
print("=" * 74)
print(summary)
""")

# ---------------------------------------------------------------------------
code(r"""
# ============ 5. Package the deliverables and download them ================
# Runs last, and only if the pipeline really produced what this project promises:
# every required artifact exists and is not truncated, the correctness CSV parses
# with the expected schema, and no check FAILed. Otherwise it prints exactly what
# is missing or wrong and refuses to hand over a half-finished result set.
import os, pathlib, zipfile

from IPython.display import FileLink, display

RES = pathlib.Path("results").resolve()
WORK = pathlib.Path.cwd().resolve()

banner("gate: verify every deliverable before downloading anything")

# artifact -> smallest plausible size, which catches empty/truncated files
REQUIRED = {
    "environment.txt": 200,
    "environment_cuda.txt": 200,
    "compile_log.txt": 200,
    "smoke_stdout.txt": 20,
    "verify_stdout.txt": 200,
    "bench_stdout.txt": 200,
    "rng_stdout.txt": 200,
    "correctness.csv": 200,
    "benchmark_sizes.csv": 100,
    "benchmark_blocks.csv": 300,
    "rng_overhead.csv": 40,
    "rng_overhead_fit.txt": 200,
    "profiling.md": 200,
    "plot_cpu_vs_gpu_time.png": 4000,
    "plot_block_sizes.png": 4000,
    "plot_speedup.png": 2000,
}

missing, problems = [], []
for name, min_bytes in REQUIRED.items():
    f = RES / name
    if not f.exists():
        missing.append(name)
        continue
    n = f.stat().st_size
    if n < min_bytes:
        problems.append("%s is %d bytes, expected >= %d - looks truncated" % (name, n, min_bytes))
    else:
        print("  ok  %-26s %9d bytes" % (name, n))
if missing:
    print("  MISSING: %s" % ", ".join(missing))

# Freshness: everything above must have been written by THIS session, not left
# behind by an earlier (possibly failed) run that cell 1..4 then failed to redo.
run_start = globals().get("RUN_START")
if run_start is None:
    print("  note  RUN_START is not set (cell 1 did not run in this session) - skipping the "
          "freshness check")
else:
    stale = [n for n in REQUIRED
             if (RES / n).exists() and (RES / n).stat().st_mtime < run_start - 5.0]
    if stale:
        problems.append("left over from an earlier session, not rewritten by this run: %s"
                        % ", ".join(stale))
    else:
        print("  ok  %-26s written after cell 1 started" % "all artifacts are fresh")

# summary.md is written last in cell 4, so it must also list the measured tables.
summary_md = RES / "summary.md"
if summary_md.exists():
    text = summary_md.read_text(encoding="utf-8", errors="replace")
    if "### CPU vs GPU over simulation size" not in text or "### RNG setup" not in text:
        problems.append("summary.md is missing one of the measured tables")
    else:
        print("  ok  %-26s %9d bytes, tables present" % ("summary.md", summary_md.stat().st_size))

# The correctness CSV must be strict CSV with no FAIL row, and it must contain GPU
# results: this notebook only makes sense on a GPU session.
CORRECTNESS_COLUMNS = ["check", "backend", "paths", "steps", "seed", "value", "expected",
                       "rel_err", "z_score", "status", "note"]
csv_path = RES / "correctness.csv"
if csv_path.exists():
    try:
        import pandas as pd
        checks = pd.read_csv(csv_path)
    except Exception as exc:
        problems.append("correctness.csv does not parse as CSV: %s" % exc)
    else:
        if list(checks.columns) != CORRECTNESS_COLUMNS:
            problems.append("correctness.csv columns are %s, expected %s"
                            % (list(checks.columns), CORRECTNESS_COLUMNS))
        elif checks.empty:
            problems.append("correctness.csv has no rows")
        else:
            status = checks["status"].astype(str).str.upper()
            backend = checks["backend"].astype(str).str.lower()
            failed = checks[status == "FAIL"]
            n_gpu_pass = int(((backend == "gpu") & (status == "PASS")).sum())
            if len(failed):
                problems.append("%d correctness row(s) FAILED: %s"
                                % (len(failed), ", ".join(sorted(set(failed["check"].astype(str))))))
            if n_gpu_pass == 0:
                problems.append("no PASS row with backend=gpu - this notebook is meant to run on "
                                "a GPU session")
            print("  ok  %-26s %d rows: %d PASS, %d SKIP, %d FAIL, %d GPU PASS"
                  % ("correctness.csv", len(checks), int((status == "PASS").sum()),
                     int((status == "SKIP").sum()), len(failed), n_gpu_pass))

if missing or problems:
    detail = ["incomplete: results/%s was never written" % m for m in missing]
    detail += ["problem: %s" % p for p in problems]
    print("\n" + "=" * 74)
    print("NOT DOWNLOADING - the run did not produce a complete, passing result set")
    print("=" * 74)
    for d in detail:
        print("  " + d)
    raise RuntimeError("result set is incomplete or wrong (%d missing, %d problem(s)): %s"
                       % (len(missing), len(problems), "; ".join(detail)))

# ---- package everything into one archive -----------------------------------
zip_path = WORK / "gbm_montecarlo_results.zip"
if zip_path.exists():
    zip_path.unlink()
with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(RES.rglob("*")):
        if f.is_file():
            z.write(str(f), str(f.relative_to(WORK)))
    source = WORK / "src" / "gbm_montecarlo.cu"
    if source.exists():  # keep the exact source that produced these numbers
        z.write(str(source), str(source.relative_to(WORK)))
with zipfile.ZipFile(str(zip_path)) as z:
    entries = sorted(z.infolist(), key=lambda i: i.filename)
    packed = sum(i.file_size for i in entries)
print("\npackaged %d files (%.1f KiB raw) -> %s (%.1f KiB)"
      % (len(entries), packed / 1024.0, zip_path.name, zip_path.stat().st_size / 1024.0))

# ---- hand the files over ---------------------------------------------------
# One best-effort browser download for the archive, then a FileLink for every
# deliverable. Kaggle's client does not define the Colab `download` JS helper: it
# logs "Javascript Error: download is not defined" while (observed on a T4 session)
# still fetching the file, and that outcome is invisible from Python. So this never
# claims a download succeeded, and the links are always there as the reliable path.
IS_COLAB = bool(os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_GPU"))
IS_KAGGLE = bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.environ.get("KAGGLE_URL_BASE")
                 or pathlib.Path("/kaggle/working").exists())
print("runtime   : %s" % ("Colab" if IS_COLAB else "Kaggle" if IS_KAGGLE
                          else "neither Colab nor Kaggle detected"))


def js_download(path):
    '''Ask the notebook client to download a file. True if the helper was called.'''
    try:
        from google.colab import files as colab_files
    except ImportError:
        return False
    try:
        colab_files.download(str(path))
    except Exception as exc:
        print("  note  JS download attempt raised: %s" % exc)
        return False
    return True


def deliver(path, label, try_js=False):
    p = pathlib.Path(path).resolve()
    if try_js and js_download(p):
        print("  asked the notebook client for %s - if no download arrived, use the link below%s"
              % (p.name, " (Kaggle logs a harmless 'download is not defined' JS error)"
                 if IS_KAGGLE else ""))
    try:
        href = str(p.relative_to(WORK))
    except ValueError:
        href = str(p)
    display(FileLink(href, result_html_prefix="download %s -> " % label))


print("\n" + "=" * 74)
print("ALL GREEN - handing over the results")
print("=" * 74)
deliver(zip_path, "everything (zip)", try_js=True)
for name in ("summary.md", "correctness.csv", "benchmark_sizes.csv", "benchmark_blocks.csv",
             "rng_overhead.csv", "profiling.md", "environment.txt", "plot_cpu_vs_gpu_time.png",
             "plot_block_sizes.png", "plot_speedup.png"):
    f = RES / name
    if f.exists():
        deliver(f, name)

print("\nNo-clicking alternative: Save Version -> Save & Run All, then the notebook's")
print("Output tab contains %s and the whole results/ directory." % zip_path.name)
print("\narchive contents:")
for info in entries:
    print("  %-34s %9d bytes" % (info.filename, info.file_size))
""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
# newline="\n" keeps the generated notebook LF-only whatever platform builds it, so the
# committed file does not churn between a Windows and a Linux checkout.
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8",
               newline="\n")
print("wrote %s (%d cells, %d code cells)"
      % (OUT, len(cells), sum(1 for c in cells if c["cell_type"] == "code")))
