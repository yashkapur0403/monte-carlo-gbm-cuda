# Validation and audit trail

Supporting detail for the top-level README: how each claim in it is checked, what the archive
auditor recomputes, and the granular correctness and profiling records that would otherwise
clutter the project overview. Nothing here is a new measurement — everything is either produced by
`./build/gbm` at run time or recomputed from the artifacts in `results/`.

## What is checked where

| Layer | Where | What it establishes |
|---|---|---|
| Correctness harness | `./build/gbm verify` (in the binary) | 34 rows in `results/correctness.csv`, one per check × backend × seed, each with value, expected, relative error, z-score, status and the tolerance it used. Exits non-zero on any failure. |
| Benchmark guard | notebook cell 4 (`sweep_anomalies()`) | Rows that cannot be physical — an end-to-end time below the kernel time, or the two sweeps disagreeing on the same geometry — trigger one re-soak and one repeat of the sweep. |
| Packaging gate | notebook cell 5 | All 16 required artifacts present and non-truncated, none predating the session, no `FAIL` row and at least one GPU `PASS` row in `correctness.csv`, `summary.md` carrying its tables. Only then is the archive produced. |
| Notebook integrity | `make validate` (`tools/validate_notebook.py`) | Every cell parses and the CUDA source embedded in the notebook is byte-identical to `src/gbm_montecarlo.cu`. |
| Packaging fixtures | `make nbtest` (`tools/test_download_cell.py`) | One complete run must be packaged; six broken ones (missing artifact, `FAIL` row, no GPU rows, truncated plot, `summary.md` without its tables, stale artifacts) must each refuse and raise. |
| Profiling parsers | `make proftest` (`tools/test_profiling_cell.py`) | The cell's parsers against real `ptxas -v`, `cuobjdump` and `nvidia-smi` output formats, including the register-limited occupancy case and the missing-facts case. |
| Profiling ladder | `make proftest` (`tools/test_profiling_ladder.py`) | The whole ladder end to end with fake `cuobjdump`/`nvidia-smi` shims: every section of `results/profiling.md`, its measured/static/derived labels and its raw artifacts. |
| Benchmark guard | `make benchtest` (`tools/test_benchmark_guard.py`) | Loads `sweep_anomalies()` out of the notebook and runs it against the committed CSVs — it must flag exactly the two clock-ramp artefacts and nothing else — plus clean and deliberately broken fixtures. |
| Archive audit | `make checkzip ZIP=…` (`tools/check_results_zip.py`) | 39 checks over a downloaded archive; see below. |

No test in this repository needs a GPU except the ones that actually run CUDA.

## What the archive auditor recomputes

`tools/check_results_zip.py` treats a downloaded `gbm_montecarlo_results.zip` as untrusted input
and re-derives rather than re-reads:

- **Integrity** — CRC of every member, and the archived `src/gbm_montecarlo.cu` must be byte-identical
  to the repository copy, so an archive cannot claim numbers the shipped source does not produce.
- **Arithmetic** — `speedup_kernel = cpu_ms/gpu_kernel_ms`, `speedup_e2e = cpu_ms/gpu_e2e_ms` and
  `gpu_paths_per_sec = paths/(kernel_ms/1000)` recomputed from the CSVs.
- **The OLS fit** — `a`, `b`, `R²`, the 252-step extrapolation and the per-path `curand_init` cost
  recomputed from `rng_overhead.csv` alone and compared against the figures quoted in the fit file.
- **Documentation agreement** — every cell of the two tables in `summary.md` compared against the
  CSVs it summarises.
- **Correctness rows** — schema, no `FAIL`/`SKIP` status, `rel_err == (value-expected)/expected` where
  `expected != 0`, the by-design zeros, z-scores inside the documented limit, and the presence of GPU
  rows (i.e. the notebook really ran on a GPU).
- **Profiling cross-consistency** — the SASS census sum against the instruction total quoted in
  `profiling.md`, the `MUFU` total against the `expf` cross-check, the register count in
  `cuobjdump -res-usage` against the occupancy derivation, the counter-denial message recorded in
  both `ncu_stdout.txt` and `profiling.md`, and the absence of any CUPTI/Nsight claim.
- **Plots** — that the three PNGs are real PNGs with plausible dimensions, not empty placeholders.

It fails, rather than warns: for the committed artifact set it reports `ALL OUTPUT CHECKS PASSED (39)`.

## Correctness: the full check list

`./build/gbm verify` defaults to 200 000 paths, 252 steps and 3 seeds (12345–12347). Tolerances are
carried in each row's `note` field.

| Check | Backend | Tolerance |
|---|---|---|
| `output_valid` — size, all finite, all > 0 | CPU + GPU | exact |
| `mean_vs_analytic` — sample mean vs `S0*exp(mu*T)` | CPU + GPU | `\|z\| < 4`, analytical s.e. |
| `variance_vs_analytic` — sample variance vs log-normal variance | CPU + GPU | `\|z\| < 4`, s.e. from the exact central 4th moment `sqrt((m4 - Var²)/N)` |
| `mean_cpu_vs_gpu` — two-sample comparison of the means | cross | `\|z\| < 4`, `s.e. = sqrt(2)*s.e.(mean)` |
| `determinism_same_seed` — identical seed ⇒ identical output | GPU | max difference exactly 0 |
| `different_seed_differs` — guards against a constant stream | GPU | max difference > 0 |
| `non_divisible_paths_structure` / `_mean` — 1000 paths at block 256 | CPU + GPU | exact size and finiteness; mean at 4 s.e. |
| `single_path` — 1 path at block 256 | CPU + GPU | exact (255 lanes must be masked off) |
| `zero_paths` — 0 paths | GPU | no launch, empty output, no crash |
| `steps_invariance` — `steps ∈ {1, 21, 252}` | GPU | `\|z\| < 4` each; `E[S(T)]` must not depend on the step count |
| `elementwise_cpu_vs_gpu` | cross | **not a requirement** — recorded as skipped by design |

**Why `\|z\| < 4`.** Under a correct implementation that is roughly a 1e-4 two-sided false-failure
rate, and the variance check uses an analytical standard error built from the exact log-normal central
4th moment rather than a hand-tuned percentage. The recorded run's largest `\|z\|` is 2.14.

**Why element-wise agreement is not required.** The backends use different generators and precisions
(Mersenne Twister in `double` versus Philox4x32 in `float`) and therefore draw different normal
streams, so element-wise equality is not a meaningful requirement; only distributional agreement is
tested, by the two-sample z-test above. The harness writes those rows with a note beginning
`SKIPPED BY DESIGN` and counts them separately in its console summary (31 PASS, 0 FAIL, 3 skipped),
rather than presenting them as a passed comparison.

**What the harness already caught.** During development it flagged a wrong raw-moment formula — a
log-normal moment written with log-drift `mu` instead of `mu − 0.5σ²` — which biased `E[S(T)]` by
about 2% and produced `z ≈ −32` on every mean check. That is precisely the class of bug a "the kernel
did not crash" smoke test would have missed.

## Kernel facts behind the performance discussion

Measured with CUDA events, static from the built cubin, derived by combining the two — the labels used
in `results/profiling.md` mean exactly this:

- **Measured:** 4.7182 ms for 1 000 000 paths × 252 steps (median of 3 kernel-only runs), i.e.
  2.12 × 10⁸ paths/s; `expf` calls are exact (1 000 000 × 252 = 252 000 000), so 5.34 × 10¹⁰ `expf`/s.
- **Static:** 28 registers/thread, 0 bytes stack, 0 spill stores, 0 spill loads, no local memory;
  920 SASS instructions in the kernel body, including 29 `MUFU` ops (11 `MUFU.EX2`, 7 `MUFU.RSQ`,
  6 `MUFU.SIN`, 5 `MUFU.COS`), 0 `LDG` and 1 `STG`.
- **Derived:** 4 resident blocks/SM × 8 warps = 32 warps/SM = 100% *theoretical* occupancy
  (threads-limited); for 1 000 000 paths at block 256, 3907 blocks = 24.4 waves at 97.7 blocks/SM.
  Theoretical occupancy is a hardware limit, not a measurement — achieved occupancy is in the
  unavailable list below.
- **Measured, coarse:** a 50 ms sampled `nvidia-smi` trace across a pre-soak gap and a `--reps 150`
  soak — SM utilisation mean 16%, median 0%, peak 70%; power 28 W mean rising to a 76 W peak against
  18 W mean before. Sampled telemetry cannot show individual launches, and on a shared GPU
  `utilization.gpu` counts other tenants, so the pre-soak window is not guaranteed idle. The 0% median
  during the soak is real: the soak is dominated by launch and copy overhead between kernels, so the
  device idles between launches.

**Not available in this environment.** `nsys` is absent from the Kaggle image, and `ncu` returns
`ERR_NVGPUCTRPERM` because reading performance counters needs `CAP_SYS_ADMIN` /
`NVreg_RestrictProfilingToAdminUsers=0`. There is therefore no CUPTI hardware-counter result anywhere
in this repository: achieved occupancy, cache hit rates, DRAM traffic, stall reasons and IPC are simply
not measured, and `results/profiling.md` says so in the same table that carries the numbers it does
have. On a machine that permits counters, the two commands in the README produce them.

## Reproducing and auditing a run

```bash
# on a machine with nvcc (or in Kaggle, via notebooks/kaggle_gbm_montecarlo.ipynb, Run All)
make verify bench rng info
# then, on the archive the notebook produced
make checkzip ZIP=~/Downloads/gbm_montecarlo_results.zip
# and the checks that need no GPU at all
make check            # validate + benchmark guard + CPU-only correctness
make nbtest proftest
```

`results/README.md` documents each artifact in the results directory and what produces it.
