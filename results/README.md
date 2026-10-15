# results/

Everything in this directory is **generated** by running the project — nothing here is
hand-written, and nothing here is fabricated.

The copy committed to this repository is the output of one Kaggle session on a single **Tesla T4**
(`sm_75`, 40 SMs), driver 580.178.04, CUDA runtime 12.8 (`nvcc` 12.8.93), on a 4-core Intel Xeon
@ 2.00 GHz. It matches the tables in the top-level README. Verify a downloaded archive against this
repository with `make checkzip ZIP=...`, which recomputes the benchmark arithmetic, the OLS fit and
the summary/CSV agreement, and fails if the archived source is not byte-identical to `src/`.

Regenerate it with:

```bash
# GPU build (full artifact set, as produced by the Kaggle notebook)
make bench rng verify info
# or, for the whole notebook workflow:
#   Run All on notebooks/kaggle_gbm_montecarlo.ipynb
```

Without an NVIDIA GPU the CPU-only build still exercises the correctness harness and the CPU
columns (GPU columns are `-1`):

```bash
make check
```

## Expected artifacts

| File | Produced by | Contents |
|---|---|---|
| `benchmark_sizes.csv` | `gbm bench` | paths, steps, block, CPU ms, GPU kernel-only ms, GPU end-to-end ms, both speedups, GPU paths/s, repetition counts |
| `benchmark_blocks.csv` | `gbm bench` | path count × CUDA block size 64/128/256/512: kernel time (median of 5 CUDA-event runs), e2e time, throughput |
| `rng_overhead.csv` | `gbm rng` | kernel time vs step count at fixed path count |
| `rng_overhead_fit.txt` | `gbm rng` | OLS fit `t(steps) = a + b*steps`: fixed per-launch cost `a` (launch + `curand_init` for all threads), marginal per-step cost `b`, `R^2` |
| `correctness.csv` | `gbm verify` | one row per check x backend x seed: `check, backend, paths, steps, seed, value, expected, rel_err, z_score, status, note`. `status` is `PASS` / `FAIL` / `SKIP`; the tolerance each check uses is stated in its `note` and in the README. Valid CSV (quoted free-text column) and self-checked against the header at write time. |
| `smoke_stdout.txt`, `verify_stdout.txt`, `bench_stdout.txt`, `rng_stdout.txt` | notebook | verbatim console logs of each run, so the numbers can be audited against the CSV |
| `compile_log.txt` | notebook | the exact `nvcc` command plus `-Xptxas -v` resource usage for the kernel |
| `environment.txt`, `environment_cuda.txt` | notebook / `gbm info` | GPU model, compute capability, SM count, driver + runtime versions, `nvcc --version`, CPU model and core count |
| `plot_cpu_vs_gpu_time.png` | notebook | CPU vs GPU (kernel-only and end-to-end) time vs simulation size, log-log |
| `plot_block_sizes.png` | notebook | kernel time and throughput vs CUDA block size, one series per path count |
| `plot_speedup.png` | notebook | CPU/GPU speedup per simulation size, with the break-even line at 1× |
| `profiling.md` | notebook | the whole profiling record: tool discovery, the exact `nsys`/`ncu` commands with their raw output (or the explicit statement that they are absent/blocked), and the Kaggle-compatible fallback — measured CUDA-event run, static kernel resources and SASS census, derived occupancy, sampled `nvidia-smi` telemetry — each number labelled measured / static / estimated |
| `profile/rng_overhead.csv` | notebook (`gbm rng --out results/profile`) | the representative profiling run's own output, kept in a subdirectory so the step sweep in `rng_overhead.csv` is not overwritten |
| `cuobjdump_res_usage.txt` | notebook | raw `cuobjdump -res-usage` report for the compiled kernel: registers, stack, shared and local memory |
| `sass_census.csv` | notebook | static SASS instruction census of `gbm_terminal_price_kernel`, opcode by opcode |
| `nvidia_smi_trace.csv` | notebook | `nvidia-smi --query-gpu=... --format=csv -lms 50` samples taken across an idle baseline and a kernel soak run (GPU/memory utilisation, SM and memory clocks, power, temperature) |
| `summary.md` | notebook | the measured tables, also patched into the README between the `RESULTS` markers |

Binary profiler artifacts (`*.nsys-rep`, `*.ncu-rep`, `*.sqlite`) are intentionally gitignored;
their text reports are kept so the evidence is reviewable without the GUI.

On Kaggle the notebook's last cell checks that everything above is present, well formed and
fresh (written after cell 1 started), and only then packs this directory plus the exact
`src/gbm_montecarlo.cu` that produced it into `../gbm_montecarlo_results.zip`. If any check
fails it refuses and raises instead, so this directory can never be handed over half-finished.
