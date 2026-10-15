#!/usr/bin/env python3
"""Tests the notebook's steady-state guard against real and deliberately broken benchmark output.

The guard cannot live in the CUDA binary: a single invocation only knows its own sweep, while the
inconsistency it looks for is either internal to one sweep (an end-to-end time below the kernel
time, which is physically impossible while the kernel is inside the end-to-end block) or a
disagreement between the size sweep and the block sweep on the same geometry. So it lives in
cell 4 of notebooks/kaggle_gbm_montecarlo.ipynb, and this test loads that exact cell, extracts
``sweep_anomalies()`` from it and runs the function against CSV fixtures:

  * the CSVs committed in results/ - the guard must flag exactly the two clock-ramp artefacts
    the README documents, and nothing else;
  * clean fixtures - no anomaly, so a healthy session is not re-run;
  * fixtures with a single injected artefact of each kind - flagged, so a real one cannot slip
    through.

Standard library only, no GPU and no CUDA toolkit required.
"""

import ast
import csv
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent

SIZES_HEADER = ("paths,steps,block,cpu_ms,gpu_kernel_ms,gpu_e2e_ms,speedup_kernel,speedup_e2e,"
                "gpu_paths_per_sec,cpu_reps,gpu_reps")
BLOCKS_HEADER = ("paths,steps,block,gpu_kernel_ms,gpu_e2e_ms,gpu_paths_per_sec,"
                 "paths_per_sec_per_step,reps")

CHECKS = []


def check(label, ok, detail=""):
    CHECKS.append(bool(ok))
    print(("  PASS  " if ok else "  FAIL  ") + label + (("  " + str(detail)) if detail else ""))
    return ok


def load_guard():
    """Pull sweep_anomalies() out of the notebook cell that defines it."""
    import json

    nb = json.loads((ROOT / "notebooks" / "kaggle_gbm_montecarlo.ipynb").read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        if "def sweep_anomalies" in src:
            tree = ast.parse(src)
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name == "sweep_anomalies":
                    mod = ast.Module(body=[node], type_ignores=[])
                    ast.fix_missing_locations(mod)
                    ns = {}
                    exec(compile(mod, "sweep_anomalies", "exec"), ns)
                    return ns["sweep_anomalies"]
    raise SystemExit("sweep_anomalies() not found in the notebook")


def write_csv(path, header, rows):
    with open(path, "w", newline="") as fh:
        fh.write(header + "\n")
        for r in rows:
            fh.write(",".join(str(x) for x in r) + "\n")


def run_guard(guard, sizes_rows, blocks_rows):
    with tempfile.TemporaryDirectory(prefix="gbm_guard_") as tmp:
        res = pathlib.Path(tmp) / "results"
        res.mkdir()
        write_csv(res / "benchmark_sizes.csv", SIZES_HEADER, sizes_rows)
        write_csv(res / "benchmark_blocks.csv", BLOCKS_HEADER, blocks_rows)
        cwd = pathlib.Path.cwd()
        try:
            import os
            os.chdir(tmp)
            return guard()
        finally:
            os.chdir(cwd)


def main():
    guard = load_guard()
    sizes = list(csv.DictReader(open(ROOT / "results" / "benchmark_sizes.csv")))
    blocks = list(csv.DictReader(open(ROOT / "results" / "benchmark_blocks.csv")))

    print("=" * 74)
    print("the guard, run against the CSVs committed in results/")
    print("=" * 74)
    real = run_guard(guard, [
        (r["paths"], r["steps"], r["block"], r["cpu_ms"], r["gpu_kernel_ms"], r["gpu_e2e_ms"],
         r["speedup_kernel"], r["speedup_e2e"], r["gpu_paths_per_sec"], r["cpu_reps"],
         r["gpu_reps"]) for r in sizes], [
        (r["paths"], r["steps"], r["block"], r["gpu_kernel_ms"], r["gpu_e2e_ms"],
         r["gpu_paths_per_sec"], r["paths_per_sec_per_step"], r["reps"]) for r in blocks])
    print("  detected %d anomaly/anomalies:" % len(real))
    for a in real:
        print("     - %s" % a)
    ramp = [a for a in real if "ms vs" in a and "ms in the other sweep" in a]
    below = [a for a in real if "e2e" in a and "kernel" in a and "<" in a]
    check("flags the cross-sweep clock-ramp disagreement (1000000 paths, block 256)",
          len(ramp) == 1 and "paths=1000000" in ramp[0], ramp[0] if ramp else "none")
    check("flags the impossible row (end-to-end below its own kernel time)",
          len(below) == 1 and "block=128" in below[0], below[0] if below else "none")
    check("flags nothing else in the real data", len(real) == 2, "%d anomaly/anomalies" % len(real))

    print()
    print("=" * 74)
    print("the guard, run against fixtures")
    print("=" * 74)
    clean_sizes = [(10000, 252, 256, 86.5, 0.074, 0.181, 1162.1, 477.2, 1.3e8, 3, 5),
                   (1000000, 252, 256, 8684.6, 3.442, 4.307, 2523.0, 2016.0, 2.9e8, 3, 5)]
    clean_blocks = [(10000, 252, 256, 0.074, 0.177, 1.35e8, 5.4e5, 5),
                    (1000000, 252, 256, 3.442, 4.307, 2.9e8, 1.15e6, 5)]
    check("a healthy session raises no anomaly", run_guard(guard, clean_sizes, clean_blocks) == [])

    broken = [(1000000, 252, 128, 4.488, 4.300, 2.2e8, 8.8e5, 5)]
    anomalies = run_guard(guard, clean_sizes, clean_blocks + broken)
    check("an injected end-to-end < kernel row is caught",
          any("block=128" in a for a in anomalies), anomalies)

    slow = [(1000000, 252, 256, 8684.6, 6.167, 6.966, 1408.3, 1246.7, 1.6e8, 3, 5)]
    anomalies = run_guard(guard, slow, clean_blocks)
    check("an injected cross-sweep disagreement is caught",
          any("ms in the other sweep" in a for a in anomalies), anomalies)

    print()
    if all(CHECKS):
        print("all %d checks passed" % len(CHECKS))
        return 0
    print("%d of %d checks failed" % (CHECKS.count(False), len(CHECKS)))
    return 1


if __name__ == "__main__":
    sys.exit(main())
