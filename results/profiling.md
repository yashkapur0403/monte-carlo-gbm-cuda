# Profiling evidence

GPU: Tesla T4 (sm_75), 40 SMs, 1024 threads/SM, 65536 regs/SM | nvcc: /usr/local/cuda/bin/nvcc | runtime: 3.13.15

Representative workload: `./build/gbm rng --paths 1000000 --block 256 --steps 252 --reps 3` (the real kernel, 252 Euler steps per path, terminal price only).

Every number below is labelled **measured** (a tool produced it here), **static** (read out of the built cubin) or **estimated** (derived, with its assumption stated). Nothing here is a CUPTI hardware-counter result unless Nsight actually ran, which is reported in section 2.

## 1. Tool discovery in this runtime

| tool | resolved path |
|---|---|
| `cuobjdump` | /usr/local/cuda/bin/cuobjdump |
| `ncu` | /usr/local/cuda/bin/ncu |
| `nsys` | not found |
| `nvdisasm` | /usr/local/cuda/bin/nvdisasm |
| `nvidia-smi` | /opt/bin/nvidia-smi |
| `nvprof` | /usr/local/cuda/bin/nvprof |
| `ptxas` | /usr/local/cuda/bin/ptxas |

## 2. Nsight Systems (`nsys`) - not in this image

`nsys` is absent, so no CUPTI timeline could be captured here. The commands that produce one on a machine that has it are in section 6.

## 3. Nsight Compute (`ncu`) - present, exit code 1

```
/usr/local/cuda/bin/ncu --set basic --launch-count 1 --kernel-name regex:gbm_terminal_price_kernel ./build/gbm rng --paths 1000000 --block 256 --steps 252 --reps 1 --out results/profile
```

**The container denies GPU performance counters** (verbatim):

```
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters on the target device 0. For instructions on enabling permissions and to get more information see https://developer.nvidia.com/ERR_NVGPUCTRPERM
```

So `ncu` cannot be used inside Kaggle: reading those counters needs `CAP_SYS_ADMIN` / `NVreg_RestrictProfilingToAdminUsers=0`. Section 4 replaces it with evidence that the kernel *can* provide here.

## 4. Kaggle-compatible kernel profile (everything below ran here)

### 4a. Representative run - **measured** (CUDA events, kernel only)

```
./build/gbm rng --paths 1000000 --block 256 --steps 252 --reps 3 --out results/profile
```

| kernel ms | us / step | ns / path / step | MPaths/s | expf calls | expf/s |
|---|---|---|---|---|---|
| 4.7182 | 18.7230 | 0.0187 | 211.94 | 252000000 | 5.341e+10 |

`expf` calls are exact (1000000 paths x 252 steps - one `expf` per Euler step), so the expf/s column is a directly measured throughput; the time is the median of 3 CUDA-event runs and excludes the host-side copy.

### 4b. Kernel resource usage - **static** (from the built cubin)

| registers / thread | static smem | stack | spill stores | spill loads | source |
|---|---|---|---|---|---|
| 28 | 0 | 0 | 0 | 0 | cuobjdump -res-usage |

No spill and no local memory means the inner loop lives entirely in registers, which is what a one-thread-per-path stencil like this needs.

### 4c. SASS instruction census - **static** (`cuobjdump -sass`)

920 SASS instructions in `gbm_terminal_price_kernel`.

| opcode | count | share |
|---|---|---|
| `LOP3.LUT` | 134 | 14.6% |
| `FFMA` | 132 | 14.3% |
| `MOV` | 110 | 12.0% |
| `IMAD.WIDE.U32` | 89 | 9.7% |
| `FMUL` | 51 | 5.5% |
| `BRA` | 46 | 5.0% |
| `IADD3` | 45 | 4.9% |
| `IMAD.HI.U32` | 32 | 3.5% |
| `IMAD` | 28 | 3.0% |
| `ISETP.NE.AND` | 28 | 3.0% |
| `UIADD3` | 18 | 2.0% |
| `FADD` | 17 | 1.8% |

These are instruction counts in the *compiled kernel body as written*, not executed counts: `#pragma unroll 4` (and the `curand_normal2` pairing) means the loop body in the SASS is several Euler steps wide and executes many times per path. Nothing on this line divides by the logical 252 steps, because that ratio would be an artefact of the unroll factor rather than a measurement.

### 4d. Occupancy and launch geometry - **derived** from 4b/4c

| thread / path | regs / thread | block | regs / block | resident blocks / SM | resident warps / SM | warps / SM | theoretical occupancy | limiter |
|---|---|---|---|---|---|---|---|---|
| 1 | 28 | 256 | 8192 | 4 | 32 | 32 | 100.0% | threads |

| launch (representative run) | blocks | waves | blocks / SM |
|---|---|---|---|
| 1000000 paths, block 256 | 3907 | 24.4 | 97.7 |

Register allocation is rounded to the 256-register warp granularity. Occupancy is the *theoretical* figure the hardware allows for this kernel and block size - without `ncu` it cannot be compared against the achieved occupancy, and that gap is stated in section 5.

### 4e. Sampled GPU telemetry around the kernel - **measured**

`nvidia-smi --query-gpu=... --format=csv -lms 50` sampled across two windows: the second before a soak run (`--reps 150`) of the same kernel geometry starts, and the soak itself. Two caveats stated up front: this is a *sampled* trace of the GPU's own counters, not a CUPTI timeline, so it cannot show individual launches; and on a shared Kaggle GPU `utilization.gpu` also counts what other tenants are doing, so the 'before' window is *not* guaranteed idle and its peak is not a baseline of our own. The informative signal is the change in the distribution (and in power draw), not either window's peak on its own.

| window | samples | SM util mean | SM util median | SM util peak | mem util peak | SM clock peak | power mean | power peak | temp peak |
|---|---|---|---|---|---|---|---|---|---|
| before the soak (1 s) | 28 | 0% | 0% | 0% | 0% | 675 MHz | 18 W | 27 W | 46 C |
| soak window | 40 | 16% | 0% | 70% | 1% | 1575 MHz | 28 W | 76 W | 48 C |

Raw samples: `results/nvidia_smi_trace.csv`.

### 4f. Static-vs-measured cross-checks - **checked against this run**

| check | static observation | verdict |
|---|---|---|
| `expf` is a hardware instruction | 29 `MUFU` instruction(s) in the kernel body, while the run performs exactly 252000000 `expf` calls (1000000 paths x 252 steps) | consistent |
| the inner loop is register-resident | 0 local/shared-memory instruction(s) in the body | no local-memory traffic, so the loop does not spill |
| global traffic is one terminal store per path | 0 `LDG`, 1 `STG` instruction(s) in the body | paths live in registers and only the terminal price is written, as designed |
| measured throughput the census has to support | 2.119e+08 paths/s | from 4a, same run |

These cross-checks are what the static data can honestly support. Converting the census into GFLOP/s or a share of peak would require the *executed* per-step instruction count of the unrolled loop - precisely the data `ncu` provides and this container blocks - so it is not claimed here.

## 5. Limitations of this profiling (stated, not hidden)

- **No hardware-counter data.** Reading SM/L2/DRAM counters needs `CAP_SYS_ADMIN` / `NVreg_RestrictProfilingToAdminUsers=0`; inside Kaggle `ncu` returns `ERR_NVGPUCTRPERM` and `nsys` is not installed. Nothing in this file is a CUPTI hardware-counter measurement, and no section above is labelled as one.
- **What replaces it:** CUDA-event kernel timing (measured), static resource and SASS statistics from the built cubin (static), occupancy derived from those two (derived), and sampled `nvidia-smi` telemetry at 50 ms (measured but coarse - it shows the device busy during the kernel phase, not individual launches).
- **Achieved occupancy, cache hit rates, DRAM traffic, stall reasons and IPC are not available** without Nsight Compute. Theoretical occupancy in 4d is a hardware limit, not a measurement.
- The instruction mix in 4c/4f is *static*: it counts SASS instructions, not executed instructions, so a divergent or partially predicated path would not show up in it. This kernel has no data-dependent branches, so the two coincide here.
- CUDA-event timings (`benchmark_*.csv`, 4a) remain the primary timing evidence and are never presented as Nsight results.

Captured raw artifacts:

- `results/ncu_stdout.txt`
- `results/cuobjdump_res_usage.txt`
- `results/sass_census.csv`
- `results/nvidia_smi_trace.csv`
- `results/profile/rng_overhead.csv` (the representative run's own output, kept out of `results/rng_overhead.csv` so the step sweep from cell 3 stays intact)

## 6. Getting the full picture on a permitted machine

```bash
make ARCH=sm_86
# timeline: kernel launches, gaps, host/device overlap
nsys profile --stats=true -o nsys_gbm ./build/gbm bench --sizes 1000000 \
    --blocks 256 --steps 252 --reps 5 --out results
# counters: occupancy achieved, stalls, DRAM/L2 traffic, IPC
ncu --set full --kernel-name regex:gbm_terminal_price_kernel -o ncu_gbm \
    ./build/gbm rng --paths 1000000 --block 256 --steps 252 --reps 1 --out results
```
