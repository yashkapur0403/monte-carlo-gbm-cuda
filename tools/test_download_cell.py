#!/usr/bin/env python3
"""Exercise cell 5 of notebooks/kaggle_gbm_montecarlo.ipynb (the packaging /
download gate) without a GPU.

The cell source is extracted from the notebook itself, so this tests the
delivered artifact rather than a copy. Fixtures mirror a real Kaggle GPU run
(the benchmark numbers are the measured T4 values from that run).

Covered:
  * positive path: gate passes, the archive is built with workspace-relative
    paths, and a FileLink is emitted for the archive and each headline file
    (IPython's real FileLink/display are used; display is intercepted so the
    objects can be inspected outside a notebook server)
  * six negative paths: a missing artifact, a FAIL row, no GPU PASS rows, a
    truncated plot, a summary.md without its measured tables, and artifacts left
    over from an earlier session must each raise, must not produce an archive and
    must not offer a download.

Needs: IPython, pandas, matplotlib.  Run:  python tools/test_download_cell.py
"""
import contextlib
import io
import json
import os
import pathlib
import shutil
import sys
import tempfile
import time
import traceback
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "kaggle_gbm_montecarlo.ipynb"

nb = json.loads(NB.read_text(encoding="utf-8"))
code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
CELL_NAME = "cell 5 (package + download)"
cell_src = "".join(code_cells[-1]["source"])
assert "NOT DOWNLOADING" in cell_src, "the last code cell is not the download gate"

SIZES_CSV = """paths,steps,block,cpu_ms,gpu_kernel_ms,gpu_e2e_ms,speedup_kernel,speedup_e2e,gpu_paths_per_sec,cpu_reps,gpu_reps
10000,252,256,86.29,0.082,0.196,1050.1,441.0,1.217e+08,3,5
100000,252,256,851.96,0.650,0.870,1310.5,979.1,1.538e+08,3,5
1000000,252,256,8572.62,6.168,7.267,1389.9,1179.6,1.621e+08,3,5
"""

# Column order copied from the fprintf() calls in src/gbm_montecarlo.cu, so the
# fixtures exercise the same schema the real harness writes.
BLOCKS_CSV = """paths,steps,block,gpu_kernel_ms,gpu_e2e_ms,gpu_paths_per_sec,paths_per_sec_per_step,reps
10000,252,64,0.081,0.207,1.236e+08,4.905,5
10000,252,128,0.080,0.191,1.245e+08,4.940,5
10000,252,256,0.082,0.193,1.219e+08,4.837,5
10000,252,512,0.129,0.273,7.737e+07,3.070,5
100000,252,64,0.654,0.891,1.528e+08,6.063,5
100000,252,128,0.654,0.867,1.528e+08,6.063,5
100000,252,256,0.650,0.899,1.539e+08,6.107,5
100000,252,512,0.640,0.875,1.562e+08,6.198,5
1000000,252,64,6.178,7.277,1.619e+08,6.425,5
1000000,252,128,6.178,7.278,1.619e+08,6.425,5
1000000,252,256,3.003,4.088,3.330e+08,13.214,5
1000000,252,512,2.995,4.091,3.339e+08,13.250,5
"""

RNG_CSV = """paths,block,steps,kernel_ms,us_per_step,ns_per_path_step,gpu_paths_per_sec
1000000,256,1,0.039,0.0390,0.039,2.56e+10
1000000,256,4,0.062,0.0155,0.015,1.61e+10
1000000,256,16,0.173,0.0108,0.011,9.25e+09
1000000,256,64,0.585,0.0091,0.009,1.71e+09
1000000,256,252,2.294,0.0091,0.009,4.36e+08
"""

RNG_FIT_TXT = """OLS fit over the step sweep (kernel-only CUDA-event timings)
  t_ms(steps) = a + b*steps
  a = 0.029774 ms   fixed per-launch cost (kernel launch + curand_init for all 1000000 paths)
  b = 0.00898685 ms   marginal cost of one extra step for all 1000000 paths
  R^2 = 0.999994  (points = 5)
  at 252 steps: a + 252*b = 2.294461 ms, of which 1.3% is fixed setup overhead
  per-path curand_init+launch cost ~ a / n = 0.0298 ns
"""

SMOKE_TXT = """== Smoke test ====================================================
S0=100 mu=0.05 sigma=0.2 T=1 steps=252
cpu : 1024 paths ok, 1.31 ms
SMOKE OK
"""

CORRECTNESS_HEADER = ("check,backend,paths,steps,seed,value,expected,rel_err,z_score,status,note")


def correctness_rows(gpu_status="PASS", include_fail=False):
    rows = [CORRECTNESS_HEADER]
    for seed in (12345, 12346):
        rows.append('%s,cpu,200000,252,%d,105.127,105.1271,1.2e-06,0.11,PASS,'
                    '"sample mean vs analytic E[S(T)]; |z| < 4"' % ("mean_matches_analytic", seed))
        rows.append('%s,gpu,200000,252,%d,105.13,105.1271,2.1e-05,%s,PASS,'
                    '"GPU sample mean vs analytic E[S(T)]"' % ("mean_matches_analytic", seed,
                                                               gpu_status))
        rows.append('%s,cpu_gpu,200000,252,%d,0.003,0,0.5,0.4,SKIP,'
                    '"SKIPPED BY DESIGN: different RNG streams"' % ("elementwise_cpu_vs_gpu", seed))
    if include_fail:
        rows.append('%s,gpu,200000,252,99999,1.0,105.1271,-0.99,99.0,FAIL,'
                    '"synthetic failure injected by tools/test_download_cell.py"'
                    % "determinism_same_seed")
    return "\n".join(rows) + "\n"


def make_workspace(fail_row=False):
    ws = pathlib.Path(tempfile.mkdtemp(prefix="gbm_dl_test_"))
    res = ws / "results"
    (ws / "src").mkdir(parents=True)
    res.mkdir()
    (ws / "src" / "gbm_montecarlo.cu").write_text("// fixture source\n", encoding="utf-8")

    texts = {
        "environment.txt": "nvidia-smi fixture\nTesla T4, 75, 15360 MiB, 470.82.01\n" + "x" * 200,
        "environment_cuda.txt": "device: Tesla T4\nSMs: 40\nccc: 7.5\n" + "y" * 200,
        "compile_log.txt": "$ nvcc -O3 ...\nptxas info: 0 bytes spill stores\n" + "z" * 200,
        "smoke_stdout.txt": SMOKE_TXT,
        "verify_stdout.txt": "-- summary: 11 PASS, 0 FAIL, 4 SKIPPED (by design) --\n" + "v" * 200,
        "bench_stdout.txt": "== Benchmark ==========\npaths=10000 block=256 ...\n" + "b" * 200,
        "rng_stdout.txt": "== RNG / launch overhead study =====\n" + "r" * 200,
        "rng_overhead_fit.txt": RNG_FIT_TXT,
        "profiling.md": "# Profiling evidence\n\nTool discovery: nsys -> None, ncu -> /usr/local/cuda/bin/ncu\n" + "p" * 200,
    }
    for name, body in texts.items():
        (res / name).write_text(body, encoding="utf-8")

    (res / "benchmark_sizes.csv").write_text(SIZES_CSV, encoding="utf-8")
    (res / "benchmark_blocks.csv").write_text(BLOCKS_CSV, encoding="utf-8")
    (res / "rng_overhead.csv").write_text(RNG_CSV, encoding="utf-8")
    (res / "correctness.csv").write_text(correctness_rows(include_fail=fail_row), encoding="utf-8")

    summary = ("# Measured results (generated by the notebook - do not edit by hand)\n\n"
               "### CPU vs GPU over simulation size (block size 256)\n\n"
               "| paths | CPU ms | GPU kernel ms |\n|---|---|---|\n| 10,000 | 86.29 | 0.082 |\n\n"
               "### RNG setup / launch-overhead fit\n\n```\nR^2 = 0.999994\n```\n")
    (res / "summary.md").write_text(summary, encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for name in ("plot_cpu_vs_gpu_time.png", "plot_block_sizes.png", "plot_speedup.png"):
            fig, ax = plt.subplots(figsize=(6, 4), dpi=140)
            ax.plot(range(40), [i * i for i in range(40)], label="fixture")
            ax.set_title(name)
            ax.legend()
            fig.tight_layout()
            fig.savefig(res / name)
            plt.close(fig)
    except ImportError as exc:  # pragma: no cover - environment dependent
        sys.exit("matplotlib is required for the plot fixtures: %s" % exc)
    return ws


def banner(t):
    print("\n" + "=" * 74 + "\n" + t + "\n" + "=" * 74)


def run_cell(ws, run_start=None, env=None):
    """Exec cell 5 with cwd=ws.

    run_start stands in for cell 1's RUN_START (the session start timestamp used
    for the staleness check). Returns (exception, captured FileLinks, stdout).
    FileLink renders against the process cwd, so the html is rendered while still
    inside ws.
    """
    import IPython.display as ipd

    captured = []
    real_display = ipd.display
    ipd.display = captured.append
    cwd = os.getcwd()
    out = io.StringIO()
    saved_env = {}
    for key, value in (env or {}).items():
        saved_env[key] = os.environ.get(key)
        os.environ[key] = value
    try:
        os.chdir(ws)
        ns = {"banner": banner, "__name__": "__main__"}
        if run_start is not None:
            ns["RUN_START"] = run_start
        with contextlib.redirect_stdout(out):
            try:
                exec(compile(cell_src, "<notebook-cell-5>", "exec"), ns)
            except Exception as exc:  # noqa: BLE001 - the failure modes are the point
                return exc, captured, out.getvalue()
            html = [o._repr_html_() for o in captured]
        return None, captured, out.getvalue(), html
    finally:
        os.chdir(cwd)
        ipd.display = real_display
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


failures = []


def check(label, ok, detail=""):
    print(("  PASS  " if ok else "  FAIL  ") + label + (("  " + detail) if detail else ""))
    if not ok:
        failures.append(label)


# ---------------------------------------------------------------- positive ---
banner("positive: a complete, passing run must be packaged and offered")
ws = make_workspace()
try:
    # cell 1 started 20 s before the fixtures were written -> all artifacts are fresh
    exc, links, out, html = run_cell(ws, run_start=time.time() - 20.0)
    check("cell runs without raising", exc is None,
          "" if exc is None else traceback.format_exception_only(type(exc), exc)[-1].strip())
    check("cell reports success", "ALL GREEN" in out)
    check("every artifact was reported ok",
          "  ok  " in out and "MISSING:" not in out and "  problem" not in out)
    check("freshness check ran and accepted this session's artifacts",
          "all artifacts are fresh" in out)
    zip_path = ws / "gbm_montecarlo_results.zip"
    check("archive created in the working directory", zip_path.exists())
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as z:
            names = sorted(z.namelist())
            bad = z.testzip()
        check("archive passes CRC check", bad is None, "first bad member: %s" % bad)
        on_disk = sorted(str(p.relative_to(ws)).replace(os.sep, "/")
                         for p in (ws / "results").rglob("*") if p.is_file())
        in_zip = sorted(n for n in names if n.startswith("results/"))
        check("archive holds every file in results/ (%d)" % len(on_disk), in_zip == on_disk,
              "missing: %s" % sorted(set(on_disk) - set(in_zip)))
        check("archive carries the exact source that produced the numbers",
              "src/gbm_montecarlo.cu" in names)
        check("archive paths are workspace-relative (no drive/tmp prefix)",
              all(not n.startswith("/") and ":" not in n for n in names))
    check("a FileLink was offered per deliverable (%d)" % len(links), len(links) >= 10)
    import IPython.display as ipd
    check("offered objects are real IPython FileLinks",
          all(isinstance(o, ipd.FileLink) for o in links))
    check("the zip itself is offered for download",
          any("gbm_montecarlo_results.zip" in o.path for o in links))
    check("offers are relative to the notebook working dir, not absolute",
          all(not o.path.startswith("/") and ":" not in o.path for o in links))
    rendered = " ".join(html)
    check("every offer renders as a working <a href> link (file really exists)",
          rendered.count("<a href=") == len(links) and "doesn't exist" not in rendered,
          "%d links for %d offers" % (rendered.count("<a href="), len(links)))
finally:
    shutil.rmtree(ws, ignore_errors=True)

# ---------------------------------------------------------------- negative ---
banner("negative: an incomplete or failing run must refuse to download")


def expect_refusal(label, mutate, expect_in_message, run_start_offset=-20.0):
    ws = make_workspace()
    try:
        mutate(ws)
        exc, links, out = run_cell(ws, run_start=time.time() + run_start_offset)[:3]
        check("%s -> raises" % label, exc is not None,
              "cell returned normally" if exc is None else type(exc).__name__)
        check("%s -> no archive is produced" % label,
              not (ws / "gbm_montecarlo_results.zip").exists())
        check("%s -> no download offered" % label, not links)
        check("%s -> says why on stdout" % label, "NOT DOWNLOADING" in out)
        if exc is not None:
            check("%s -> exception message names the cause" % label,
                  expect_in_message in str(exc), str(exc)[-90:])
    finally:
        shutil.rmtree(ws, ignore_errors=True)


expect_refusal("missing artifact (profiling.md)",
               lambda ws: (ws / "results" / "profiling.md").unlink(),
               "profiling.md")

expect_refusal("FAIL row in correctness.csv",
               lambda ws: (ws / "results" / "correctness.csv").write_text(
                   correctness_rows(include_fail=True), encoding="utf-8"),
               "FAILED")

expect_refusal("no GPU PASS rows (CPU-only correctness.csv)",
               lambda ws: (ws / "results" / "correctness.csv").write_text(
                   "".join(l + "\n" for l in correctness_rows().splitlines()
                           if ",gpu," not in l), encoding="utf-8"),
               "backend=gpu")

expect_refusal("truncated plot",
               lambda ws: (ws / "results" / "plot_speedup.png").write_bytes(b"\x89PNG\r\n\x1a\n"),
               "plot_speedup.png")

expect_refusal("summary.md without the measured tables",
               lambda ws: (ws / "results" / "summary.md").write_text("# nothing\n" + "n" * 400,
                                                                     encoding="utf-8"),
               "summary.md is missing")

# All files present and well formed, but they predate this session (e.g. leftovers
# from an earlier run). run_start_offset puts RUN_START in the future, so every
# fixture looks stale.
expect_refusal("artifacts left over from an earlier session",
               lambda ws: None,
               "left over from an earlier session",
               run_start_offset=+3600.0)

# ------------------------------------------------------------------ JS path ---
# On Colab (and, empirically, on Kaggle) `google.colab.files.download` exists. The
# cell must use it at most once - for the archive - keep the FileLinks as the
# reliable path, and never claim that a download succeeded.
banner("browser download: one best-effort attempt, links always offered")
import types

calls = []
fake_google = types.ModuleType("google")
fake_colab = types.ModuleType("google.colab")
fake_colab.files = types.SimpleNamespace(download=lambda p: calls.append(str(p)))
fake_google.colab = fake_colab
sys.modules["google"] = fake_google
sys.modules["google.colab"] = fake_colab
ws = make_workspace()
try:
    exc, links, out, _ = run_cell(ws, run_start=time.time() - 20.0,
                                  env={"KAGGLE_KERNEL_RUN_TYPE": "Interactive"})
    check("js: the cell still completes", exc is None,
          "" if exc is None else str(exc)[-80:])
    check("js: the runtime is reported as Kaggle", "runtime   : Kaggle" in out)
    check("js: the Kaggle client's JS error is mentioned up front",
          "download is not defined" in out)
    check("js: exactly one browser-download attempt", len(calls) == 1, str(calls))
    check("js: the attempt is the archive", bool(calls)
          and calls[0].endswith("gbm_montecarlo_results.zip"), str(calls))
    check("js: FileLinks are still rendered for every deliverable", len(links) >= 10,
          "%d links" % len(links))
    check("js: it does not claim a download succeeded",
          "browser download started" not in out and "downloaded" not in out)
finally:
    del sys.modules["google.colab"]
    del sys.modules["google"]
    shutil.rmtree(ws, ignore_errors=True)

banner("RESULT")
print("%d checks failed" % len(failures) if failures else "all checks passed")
for f in failures:
    print("  -", f)
sys.exit(1 if failures else 0)
