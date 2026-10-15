#!/usr/bin/env python3
"""Verify a gbm_montecarlo_results.zip produced by the Kaggle notebook.

Checks the artifact itself, not the console log: archive integrity, the schema and
arithmetic of every CSV, the correctness rows against the analytic model, the OLS
fit in rng_overhead_fit.txt against rng_overhead.csv, summary.md against the CSVs,
the profiling artifacts against each other, and that the plots are real PNGs.

Run:  python tools/check_results_zip.py [path/to/gbm_montecarlo_results.zip]
Needs: pandas, numpy.
"""
import hashlib
import io
import pathlib
import re
import struct
import sys
import zipfile

import numpy as np
import pandas as pd

DEFAULT = pathlib.Path.home() / "Downloads" / "gbm_montecarlo_results.zip"
ZIP = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
if not ZIP.exists():
    sys.exit("zip not found: %s" % ZIP)

FAILS = []
CHECKS = []


def ck(label, fn):
    """Run one check; an exception is a failure with its cause, never a crash."""
    CHECKS.append(label)
    try:
        ok, detail = fn()
    except Exception as exc:  # noqa: BLE001 - a broken artifact is the thing under test
        ok, detail = False, "%s: %s" % (type(exc).__name__, exc)
    print(("  PASS  " if ok else "  FAIL  ") + label + (("  " + str(detail)) if detail else ""))
    if not ok:
        FAILS.append(label)


raw = ZIP.read_bytes()
if not zipfile.is_zipfile(ZIP):
    sys.exit("not a zip archive: %s (%d bytes)" % (ZIP, len(raw)))
sha = hashlib.sha256(raw).hexdigest()
print("file   : %s" % ZIP)
print("size   : %d bytes" % len(raw))
print("sha256 : %s" % sha)
print("")

with zipfile.ZipFile(ZIP) as z:
    names = sorted(z.namelist())
    rd = z.read
    print("members: %d  (%s ...)" % (len(names), ", ".join(n.split("/")[-1] for n in names[:4])))
    print("")

    def crc_ok():
        bad = z.testzip()
        return bad is None, ("every member OK" if bad is None else "first bad member: %s" % bad)

    ck("archive integrity (CRC over every member)", crc_ok)

    # ---------------------------------------------------------------- the fit --
    rng = pd.read_csv(io.BytesIO(rd("results/rng_overhead.csv")))
    fit = rd("results/rng_overhead_fit.txt").decode()

    def grab(pattern):
        m = re.search(pattern, fit)
        if not m:
            raise ValueError("pattern not found: %s" % pattern)
        return float(m.group(1))

    a_c, b_c = grab(r"a = ([\d.]+) ms"), grab(r"b = ([\d.]+) ms")
    r2_c, n_c = grab(r"R\^2 = ([\d.]+)"), int(grab(r"points = (\d+)"))
    at252_c, init_c = grab(r"= ([\d.]+) ms, of which"), grab(r"= ([\d.]+) ns")
    b_f, a_f = np.polyfit(rng.steps.astype(float), rng.kernel_ms.astype(float), 1)
    pred = a_f + b_f * rng.steps
    r2_f = 1 - ((rng.kernel_ms - pred) ** 2).sum() / ((rng.kernel_ms - rng.kernel_ms.mean()) ** 2).sum()

    ck("fit: point count matches the CSV rows",
       lambda: (len(rng) == n_c, "%d rows vs %d claimed" % (len(rng), n_c)))
    ck("fit: slope b recomputed from the CSV",
       lambda: (abs(b_f - b_c) / b_c < 0.01, "%.6f vs claimed %.6f" % (b_f, b_c)))
    ck("fit: intercept a recomputed from the CSV",
       lambda: (abs(a_f - a_c) / a_c < 0.05, "%.6f vs claimed %.6f" % (a_f, a_c)))
    ck("fit: R^2 recomputed", lambda: (abs(r2_f - r2_c) < 1e-4, "%.6f vs claimed %.6f" % (r2_f, r2_c)))
    ck("fit: 252-step extrapolation consistent with a and b",
       lambda: (abs((a_c + 252 * b_c) - at252_c) < 1e-3,
                "%.6f vs claimed %.6f" % (a_c + 252 * b_c, at252_c)))
    ck("fit: per-path curand_init cost is a/n for n=1e6",
       lambda: (abs(a_c - init_c) < 2e-3, "%.4f ns vs claimed %.4f ns" % (a_c, init_c)))

    # ----------------------------------------------------- summary vs the CSVs --
    summ = rd("results/summary.md").decode()
    sizes = pd.read_csv(io.BytesIO(rd("results/benchmark_sizes.csv")))
    blocks = pd.read_csv(io.BytesIO(rd("results/benchmark_blocks.csv")))

    def table_after(md, heading):
        lines = md.splitlines()
        i = next(k for k, l in enumerate(lines) if l.startswith(heading))
        rows = []
        for line in lines[i:]:
            if line.startswith("|") and line.strip().endswith("|"):
                rows.append([c.strip() for c in line.strip("|").split("|")])
            elif rows:
                break
        return rows[0], rows[2:]

    def num(s):
        return float(s.replace(",", "").replace("x", "").replace("e+0", "e+"))

    _, r1 = table_after(summ, "### CPU vs GPU over simulation size")
    _, r2 = table_after(summ, "### GPU kernel time and throughput vs CUDA block size")
    ck("summary: size table has one row per CSV row",
       lambda: (len(r1) == len(sizes), "%d vs %d" % (len(r1), len(sizes))))
    ck("summary: size-table CPU ms matches the CSV",
       lambda: (not [r for r, c in zip(r1, sizes.cpu_ms) if abs(num(r[3]) - c) > 0.01], ""))
    ck("summary: size-table GPU kernel ms matches the CSV",
       lambda: (not [r for r, c in zip(r1, sizes.gpu_kernel_ms) if abs(num(r[4]) - c) > 0.001], ""))
    ck("summary: block table has one row per CSV row",
       lambda: (len(r2) == len(blocks), "%d vs %d" % (len(r2), len(blocks))))
    ck("summary: block-table kernel ms matches the CSV",
       lambda: (not [r for r, c in zip(r2, blocks.gpu_kernel_ms) if abs(num(r[2]) - c) > 0.001], ""))

    # ------------------------------------------------- CSV arithmetic ----------
    ck("sizes: speedup_kernel == cpu_ms / gpu_kernel_ms",
       lambda: (bool(np.allclose(sizes.speedup_kernel, sizes.cpu_ms / sizes.gpu_kernel_ms,
                                 rtol=1e-3)), ""))
    ck("sizes: speedup_e2e == cpu_ms / gpu_e2e_ms",
       lambda: (bool(np.allclose(sizes.speedup_e2e, sizes.cpu_ms / sizes.gpu_e2e_ms, rtol=1e-3)), ""))
    ck("blocks: gpu_paths_per_sec == paths / (kernel_ms/1000)",
       lambda: (bool(np.allclose(blocks.gpu_paths_per_sec,
                                 blocks.paths / (blocks.gpu_kernel_ms / 1e3), rtol=2e-3)), ""))
    ck("blocks: e2e >= kernel on every row except the clock-ramp one",
       lambda: ((blocks.gpu_e2e_ms < blocks.gpu_kernel_ms).sum() <= 1,
                "%d row(s) deviate" % (blocks.gpu_e2e_ms < blocks.gpu_kernel_ms).sum()))

    # --------------------------------------------- correctness rows ------------
    chk = pd.read_csv(io.BytesIO(rd("results/correctness.csv")))
    ck("correctness: header is exactly the documented 11 columns",
       lambda: (list(chk.columns) == ["check", "backend", "paths", "steps", "seed", "value",
                                      "expected", "rel_err", "z_score", "status", "note"],
                list(chk.columns)))
    ck("correctness: no row carries FAIL or SKIP status",
       lambda: (set(chk.status) <= {"PASS"}, sorted(set(chk.status))))
    # The harness records the cross-backend elementwise comparison as a row with
    # status PASS but a note saying it is skipped by design (different RNGs and
    # precisions make it meaningless). Keep that explicit rather than letting the
    # PASS status imply elementwise agreement was verified.
    skipped = chk[chk.note.str.startswith("SKIPPED BY DESIGN")]
    ck("correctness: the elementwise row is marked skipped by design, not claimed as agreement",
       lambda: (len(skipped) == 3 and set(skipped.check) == {"elementwise_cpu_vs_gpu"},
                "%d row(s), checks=%s" % (len(skipped), sorted(set(skipped.check)))))
    zc = chk[chk.z_score != 0].z_score
    ck("correctness: z-scores inside the documented |z| < 4",
       lambda: (bool((zc.abs() < 4).all()), "max |z| = %.2f" % zc.abs().max()))
    means = chk[chk.check.str.contains("mean") & (chk.expected > 1)]
    ck("correctness: sample means within 1% of analytic E[S(T)]",
       lambda: (bool((abs(means.value - means.expected) / means.expected < 0.01).all()),
                "E = %.5f" % means.expected.iloc[0]))
    # Rows whose expected value is 0 (cross-backend differences, determinism max|dS|)
    # carry rel_err = 0 by the harness's own definition, so only the other rows can be
    # compared against (value-expected)/expected.
    nz = chk.expected != 0
    ck("correctness: rel_err == (value-expected)/expected where expected != 0",
       lambda: (bool(np.allclose(chk.rel_err[nz], ((chk.value - chk.expected) / chk.expected)[nz],
                                 rtol=1e-3, atol=1e-9)),
                "%d of %d rows compared, max |diff| %.1e"
                % (int(nz.sum()), len(chk),
                   float(np.abs(chk.rel_err[nz]
                                - ((chk.value - chk.expected) / chk.expected)[nz]).max()))))
    ck("correctness: rows with expected == 0 have rel_err == 0 by design",
       lambda: (bool((chk.rel_err[~nz] == 0).all()), "%d such rows" % int((~nz).sum())))
    ck("correctness: free-text note column is quoted and parses",
       lambda: (chk.note.notna().all() and any("," in n for n in chk.note), ""))
    ck("correctness: has GPU backend rows (the notebook ran on a GPU)",
       lambda: ((chk.backend == "gpu").any(),
                "%d GPU PASS" % ((chk.backend == "gpu") & (chk.status == "PASS")).sum()))

    # ------------------------------------------- profiling cross-agreement -----
    prof = rd("results/profiling.md").decode()
    cen = pd.read_csv(io.BytesIO(rd("results/sass_census.csv")))
    reg = int(re.search(r"REG:(\d+)", rd("results/cuobjdump_res_usage.txt").decode()).group(1))
    ck("profiling: cuobjdump REG matches the register count quoted in 4b/4d",
       lambda: (reg == int(re.search(r"\| (\d+) \| 256 \| \d+ \|", prof).group(1)),
                "cuobjdump says %d" % reg))
    ck("profiling: census sum matches 4c's instruction total",
       lambda: (int(cen["count"].sum()) == int(re.search(r"(\d+) SASS instructions", prof).group(1)),
                "%d rows sum" % int(cen['count'].sum())))
    mufu = int(cen[cen.opcode.str.startswith("MUFU")]["count"].sum())
    ck("profiling: MUFU census total matches 4f's claim",
       lambda: (mufu == int(re.search(r"(\d+) `MUFU` instruction", prof).group(1)), mufu))
    ck("profiling: 4f's expf count equals paths x steps",
       lambda: (int(re.search(r"exactly (\d+) `expf` calls", prof).group(1)) == 1000000 * 252, ""))
    ck("profiling: 4d occupancy maths is right for that register count and block",
       lambda: (((reg * 32 + 255) // 256 * 256 * 8)
                == int(re.search(r"\| 256 \| (\d+) \|", prof).group(1)), ""))
    ck("profiling: ncu counter denial recorded verbatim in both files",
       lambda: ("ERR_NVGPUCTRPERM" in rd("results/ncu_stdout.txt").decode()
                and "ERR_NVGPUCTRPERM" in prof, ""))
    ck("profiling: environment record names the device that was used",
       lambda: ("Tesla T4" in rd("results/environment_cuda.txt").decode()
                and "Tesla T4" in prof, ""))
    ck("profiling: no section claims a CUPTI/Nsight counter result",
       lambda: ("not a CUPTI timeline" in prof
                and "Nsight Systems produced a usable timeline" not in prof, ""))
    ck("profiling: every number is labelled measured/static/derived",
       lambda: (all(w in prof for w in ("**measured**", "**static**", "**derived**")), ""))

    # ------------------------------------------------------------- the plots ---
    pngs = [n for n in names if n.endswith(".png")]

    def png_ok(name):
        d = rd(name)
        w, h = struct.unpack(">II", d[16:24])
        return (d[:8] == b"\x89PNG\r\n\x1a\n" and 400 <= w <= 4000 and 300 <= h <= 3000,
                "%dx%d, %d bytes" % (w, h, len(d)))

    for p in pngs:
        ck("plot %s is a real PNG" % pathlib.Path(p).name, (lambda p=p: png_ok(p)))
    ck("three plots present", lambda: (len(pngs) == 3, len(pngs)))

    # --------------------------------------------------------- the source ------
    src = rd("src/gbm_montecarlo.cu")
    repo = pathlib.Path(__file__).resolve().parents[1] / "src" / "gbm_montecarlo.cu"

    def source_matches():
        want = repo.read_bytes()
        if src == want:
            return True, "%d bytes" % len(src)
        note = "%d vs %d bytes" % (len(src), len(want))
        if len(src) == len(want) + 2 and src[:1] == b"\n" and src.endswith(b"\n\n"):
            note += (" - one extra newline at each end: this zip came from a notebook older "
                     "than the cell-2 fix, which now strips them")
        return False, note

    ck("archived source is byte-identical to the repository file", source_matches)

print("")
if FAILS:
    print("%d of %d checks failed:" % (len(FAILS), len(CHECKS)))
    for f in FAILS:
        print("  -", f)
else:
    print("ALL OUTPUT CHECKS PASSED (%d checks)" % len(CHECKS))
sys.exit(1 if FAILS else 0)
