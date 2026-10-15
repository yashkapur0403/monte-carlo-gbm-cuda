// =============================================================================
//  Monte Carlo Geometric Brownian Motion (GBM) terminal-price simulator
//  ---------------------------------------------------------------------------
//  One self-contained translation unit that provides:
//    1. A single-threaded CPU baseline  (std::mt19937_64 + Box-Muller)
//    2. A custom CUDA kernel           (curandStatePhilox4_32_10_t, __global__)
//    3. A numerical-correctness harness (analytical log-normal moments, z-scores)
//    4. A benchmark harness            (CUDA events, medians, CSV output)
//
//  Model (real-world drift mu; the Euler update is exact in distribution for
//  the terminal value, so any steps >= 1 gives the same S(T) law):
//      S(t+dt) = S(t) * exp( (mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z ),
//      Z ~ N(0,1),  dt = T / steps
//  Only the terminal price S(T) is stored: one thread per independent path.
//
//  ---------------------------------------------------------------------------
//  Build (GPU, requires nvcc + CUDA toolkit):
//      nvcc -O3 -std=c++17 -arch=sm_75 -lineinfo \
//           -o gbm src/gbm_montecarlo.cu -lcurand
//
//  Build (CPU only, no CUDA toolkit needed -- used to sanity-check this file):
//      g++ -x c++ -O3 -std=c++11 -DGBM_CPU_ONLY -o gbm_cpu src/gbm_montecarlo.cu
//
//  ---------------------------------------------------------------------------
//  Usage:
//      ./gbm info                          # device + runtime information
//      ./gbm smoke                         # fast end-to-end sanity run
//      ./gbm verify   [--paths N] [--seeds K] [--steps S] [--out DIR]
//      ./gbm bench    [--sizes a,b,c] [--blocks a,b,c] [--steps S]
//                     [--reps R] [--cpu-reps R] [--out DIR]
//      ./gbm rng      [--paths N] [--block B] [--steps a,b,c] [--reps R] [--out DIR]
//
//  Every command writes machine-readable CSV into --out (default: results/).
//  No measurement in this file is invented: all numbers are timed at runtime.
// =============================================================================

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#if defined(_WIN32)
#  include <direct.h>
#  define GBM_MKDIR(p) _mkdir(p)
#else
#  include <sys/stat.h>
#  include <sys/types.h>
#  define GBM_MKDIR(p) mkdir((p), 0755)
#endif

#ifdef GBM_CPU_ONLY
#  define GBM_HAS_CUDA 0
#else
#  define GBM_HAS_CUDA 1
#  include <cuda_runtime.h>
#  include <curand_kernel.h>
#endif

#define GBM_PI 3.14159265358979323846

// -----------------------------------------------------------------------------
//  Small utilities
// -----------------------------------------------------------------------------

static std::string g_out_dir = "results";

// Portable recursive mkdir (no <filesystem>, so this also builds with older
// toolchains). Existing directories are ignored; failures surface later as an
// explicit "cannot write <file>" message from the writer that needed the dir.
static void ensure_dir(const std::string& path) {
  if (path.empty()) return;
  std::string cur;
  for (size_t i = 0; i < path.size(); ++i) {
    const char c = path[i];
    if (c == '/' || c == '\\') {
      if (!cur.empty()) GBM_MKDIR(cur.c_str());
    }
    cur += c;
  }
  if (!cur.empty()) GBM_MKDIR(cur.c_str());
}

static std::string path_in_out(const std::string& fname) { return g_out_dir + "/" + fname; }

static double now_ms() {
  using clock = std::chrono::steady_clock;
  static const clock::time_point t0 = clock::now();
  return std::chrono::duration<double, std::milli>(clock::now() - t0).count();
}

// Median of a copy of the samples. Even counts -> mean of the two middle values.
static double median(std::vector<double> v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  size_t n = v.size();
  return (n % 2 == 1) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

static double mean_of(const std::vector<double>& v) {
  if (v.empty()) return 0.0;
  double s = 0.0;
  for (size_t i = 0; i < v.size(); ++i) s += v[i];
  return s / (double)v.size();
}

static std::string get_opt(int argc, char** argv, const char* key, const std::string& def) {
  for (int i = 1; i + 1 < argc; ++i)
    if (std::strcmp(argv[i], key) == 0) return std::string(argv[i + 1]);
  return def;
}
static long long get_opt_ll(int argc, char** argv, const char* key, long long def) {
  std::string s = get_opt(argc, argv, key, "");
  return s.empty() ? def : std::atoll(s.c_str());
}
static std::vector<long long> parse_int_list(const std::string& s) {
  std::vector<long long> out;
  size_t i = 0;
  while (i < s.size()) {
    size_t j = s.find(',', i);
    if (j == std::string::npos) j = s.size();
    std::string tok = s.substr(i, j - i);
    if (!tok.empty()) out.push_back(std::atoll(tok.c_str()));
    i = j + 1;
  }
  return out;
}
static std::string join_csv(const std::vector<long long>& v) {
  std::string s;
  for (size_t i = 0; i < v.size(); ++i) {
    if (i) s += ",";
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%lld", v[i]);
    s += buf;
  }
  return s;
}

// -----------------------------------------------------------------------------
//  Simulation parameters and analytical (log-normal) reference moments
// -----------------------------------------------------------------------------

struct SimParams {
  double S0;     // spot
  double mu;     // annualised drift
  double sigma;  // annualised volatility
  double T;      // horizon in years
  int steps;     // time steps per path (dt = T / steps)
};

// Our process has log-drift (mu - 0.5*sigma^2), so
//   E[S(T)^k] = S0^k * exp(k*mu*T + 0.5*k*(k-1)*sigma^2*T).
// k = 1 gives the textbook E[S(T)] = S0*exp(mu*T) used as the primary check.
static double raw_moment(const SimParams& p, int k) {
  return std::pow(p.S0, (double)k) *
         std::exp((double)k * p.mu * p.T +
                  0.5 * (double)k * (double)(k - 1) * p.sigma * p.sigma * p.T);
}
static double analytic_mean(const SimParams& p) { return raw_moment(p, 1); }
static double analytic_var(const SimParams& p) {
  double m1 = raw_moment(p, 1), m2 = raw_moment(p, 2);
  return m2 - m1 * m1;
}
// Central 4th moment, built from raw moments (numerically stable enough here).
static double analytic_central_m4(const SimParams& p) {
  double m1 = raw_moment(p, 1), m2 = raw_moment(p, 2), m3 = raw_moment(p, 3), m4 = raw_moment(p, 4);
  return m4 - 4.0 * m1 * m3 + 6.0 * m1 * m1 * m2 - 3.0 * m1 * m1 * m1 * m1;
}
// Standard error of the sample mean and of the unbiased sample variance.
static double se_mean(const SimParams& p, long long n) { return std::sqrt(analytic_var(p) / (double)n); }
static double se_var(const SimParams& p, long long n) {
  double v = analytic_var(p);
  double c = analytic_central_m4(p) - v * v;
  return (c > 0.0) ? std::sqrt(c / (double)n) : 0.0;
}

// -----------------------------------------------------------------------------
//  Empirical statistics of a set of terminal prices
// -----------------------------------------------------------------------------

struct Stats {
  long long n;
  long long n_nonfinite;
  long long n_nonpositive;
  double mean;
  double var;  // unbiased (n-1)
  double sd;
  double min;
  double max;
};

static Stats compute_stats(const std::vector<float>& v) {
  Stats s;
  s.n = (long long)v.size();
  s.n_nonfinite = 0;
  s.n_nonpositive = 0;
  s.mean = 0.0;
  s.var = 0.0;
  s.sd = 0.0;
  s.min = 0.0;
  s.max = 0.0;
  if (v.empty()) return s;
  // Accumulate in double regardless of storage type (float prices -> exact sums).
  double sum = 0.0, sumsq = 0.0, mn = 0.0, mx = 0.0;
  bool first = true;
  for (size_t i = 0; i < v.size(); ++i) {
    double x = (double)v[i];
    if (!std::isfinite(x)) {
      s.n_nonfinite++;
      continue;
    }
    if (x <= 0.0) s.n_nonpositive++;
    sum += x;
    sumsq += x * x;
    if (first) {
      mn = mx = x;
      first = false;
    } else {
      mn = std::min(mn, x);
      mx = std::max(mx, x);
    }
  }
  long long m = s.n - s.n_nonfinite;
  s.min = first ? 0.0 : mn;
  s.max = first ? 0.0 : mx;
  if (m > 0) {
    s.mean = sum / (double)m;
    if (m > 1) {
      double c = sumsq - (double)m * s.mean * s.mean;
      if (c < 0.0) c = 0.0;
      s.var = c / (double)(m - 1);
    }
    s.sd = std::sqrt(s.var);
  }
  return s;
}

// -----------------------------------------------------------------------------
//  CPU baseline
//  std::mt19937_64 (Mersenne Twister, 2^19937-1 period) + Box-Muller.
//  All arithmetic in double; the GPU path uses float, so the two are compared
//  statistically (distributional agreement), never element-by-element.
// -----------------------------------------------------------------------------

// Box-Muller generator over std::mt19937_64. Both sin/cos outputs are used, so
// each (log, sqrt, sin, cos) block yields two normal deviates (one is cached).
struct BoxMuller {
  std::mt19937_64 gen;
  bool has_spare;
  double spare;
  explicit BoxMuller(uint64_t seed) : gen(seed), has_spare(false), spare(0.0) {}
  inline double uniform() {
    // 53 random bits -> uniform double in [0,1).
    return (double)(gen() >> 11) * (1.0 / 9007199254740992.0);
  }
  inline double normal() {
    if (has_spare) {
      has_spare = false;
      return spare;
    }
    double u1;
    do {
      u1 = uniform();
    } while (u1 <= 1e-300);  // guard log(0)
    const double u2 = uniform();
    const double r = std::sqrt(-2.0 * std::log(u1));
    const double th = 2.0 * GBM_PI * u2;
    spare = r * std::sin(th);
    has_spare = true;
    return r * std::cos(th);
  }
};

static double cpu_simulate(const SimParams& p, long long n_paths, uint64_t seed,
                           std::vector<float>& out) {
  out.assign((size_t)n_paths, 0.0f);
  const double dt = p.T / (double)p.steps;
  const double drift = (p.mu - 0.5 * p.sigma * p.sigma) * dt;
  const double vol = p.sigma * std::sqrt(dt);

  BoxMuller rng(seed);
  const double t0 = now_ms();
  for (long long i = 0; i < n_paths; ++i) {
    double s = p.S0;
    for (int t = 0; t < p.steps; ++t) s *= std::exp(drift + vol * rng.normal());
    out[(size_t)i] = (float)s;
  }
  return now_ms() - t0;
}

// -----------------------------------------------------------------------------
//  Custom CUDA kernel
//  One thread == one independent Monte Carlo path. Threads are stored in a
//  per-thread cuRAND Philox state (counter-based, high quality, cheap to seed
//  with a distinct subsequence per path). The `i >= n_paths` guard makes any
//  path count legal, including counts that are not a multiple of blockDim.
// -----------------------------------------------------------------------------
#if GBM_HAS_CUDA

__global__ void gbm_terminal_price_kernel(float* __restrict__ out,
                                         const long long n_paths,
                                         const int steps,
                                         const float S0,
                                         const float drift,  // (mu - 0.5 sigma^2) * dt
                                         const float vol,    // sigma * sqrt(dt)
                                         const unsigned long long seed) {
  const long long i = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
  if (i >= n_paths) return;  // handles non-multiples of the block size

  curandStatePhilox4_32_10_t st;
  curand_init(seed, (unsigned long long)i, 0ULL, &st);

  float s = S0;
  const int pairs = steps >> 1;
#pragma unroll 4
  for (int k = 0; k < pairs; ++k) {
    // Two N(0,1) deviates per Philox counter step: halves RNG overhead.
    const float2 z = curand_normal2(&st);
    s *= expf(drift + vol * z.x);
    s *= expf(drift + vol * z.y);
  }
  if (steps & 1) {
    const float z = curand_normal(&st);
    s *= expf(drift + vol * z);
  }
  out[i] = s;
}

static bool cuda_check(cudaError_t rc, const char* what, std::string* err) {
  if (rc != cudaSuccess) {
    if (err) *err = std::string(what) + ": " + cudaGetErrorString(rc);
    return false;
  }
  return true;
}

enum GpuMeasure { GPU_KERNEL_ONLY, GPU_END_TO_END };

// Runs the kernel once. GPU_KERNEL_ONLY times only the kernel with CUDA events;
// GPU_END_TO_END times cudaMalloc + launch + D2H copy + cudaFree.
static bool gpu_run_once(const SimParams& p, long long n_paths, int block, uint64_t seed,
                         GpuMeasure mode, std::vector<float>& out, double* ms, std::string* err) {
  out.assign((size_t)n_paths, 0.0f);
  *ms = -1.0;
  if (n_paths <= 0) {
    *ms = 0.0;
    return true;
  }
  const float dt = (float)(p.T / (double)p.steps);
  const float drift = (float)((p.mu - 0.5 * p.sigma * p.sigma) * (double)dt);
  const float vol = (float)(p.sigma * std::sqrt((double)dt));
  const size_t bytes = (size_t)n_paths * sizeof(float);
  const unsigned grid = (unsigned)((n_paths + (long long)block - 1) / (long long)block);

  cudaEvent_t e0 = NULL, e1 = NULL;
  if (!cuda_check(cudaEventCreate(&e0), "cudaEventCreate", err) ||
      !cuda_check(cudaEventCreate(&e1), "cudaEventCreate", err))
    return false;

  float* d_out = NULL;
  const double t_e2e = (mode == GPU_END_TO_END) ? now_ms() : 0.0;
  if (!cuda_check(cudaMalloc(&d_out, bytes), "cudaMalloc", err)) return false;

  if (mode == GPU_KERNEL_ONLY) cuda_check(cudaEventRecord(e0, 0), "cudaEventRecord", err);
  gbm_terminal_price_kernel<<<grid, block>>>(d_out, n_paths, p.steps, (float)p.S0, drift, vol,
                                            (unsigned long long)seed);
  if (mode == GPU_KERNEL_ONLY) {
    cuda_check(cudaEventRecord(e1, 0), "cudaEventRecord", err);
    cuda_check(cudaEventSynchronize(e1), "cudaEventSynchronize", err);
    float el = 0.0f;
    cuda_check(cudaEventElapsedTime(&el, e0, e1), "cudaEventElapsedTime", err);
    *ms = (double)el;
  }

  // Always validate the launch (also catches async kernel faults before the copy).
  if (!cuda_check(cudaPeekAtLastError(), "kernel launch", err)) {
    cudaFree(d_out);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return false;
  }
  if (!cuda_check(cudaMemcpy(&out[0], d_out, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy D2H",
                  err)) {
    cudaFree(d_out);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return false;
  }

  if (mode == GPU_END_TO_END) *ms = now_ms() - t_e2e;

  cudaFree(d_out);
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  return err == NULL || err->empty();
}

static bool gpu_warmup(const SimParams& p, int block) {
  std::vector<float> scratch;
  double ms = 0.0;
  std::string err;
  return gpu_run_once(p, 1024, block, 1ULL, GPU_KERNEL_ONLY, scratch, &ms, &err);
}

#endif  // GBM_HAS_CUDA

// -----------------------------------------------------------------------------
//  Result rows
// -----------------------------------------------------------------------------

struct CorrectnessRow {
  std::string check;
  std::string backend;
  long long paths;
  int steps;
  uint64_t seed;
  double value;
  double expected;
  double rel_err;
  double z;
  std::string status;  // PASS / FAIL / SKIP
  std::string note;
};

// Column order must match the fprintf() in write_correctness_csv() exactly; the
// field count is asserted below so header/row drift cannot slip through.
static const char* K_CORRECTNESS_HEADER =
    "check,backend,paths,steps,seed,value,expected,rel_err,z_score,status,note";

static int count_csv_fields(const char* s, bool respect_quotes) {
  int n = 1;
  bool in_quotes = false;
  for (const char* q = s; *q; ++q) {
    if (respect_quotes && *q == '"') in_quotes = !in_quotes;
    else if (*q == ',' && !in_quotes) n++;
  }
  return n;
}

// Returns false if the file could not be written or if any row's field count
// disagrees with the header, so a malformed CSV fails the run instead of being
// silently produced.
static bool write_correctness_csv(const std::vector<CorrectnessRow>& rows) {
  FILE* f = std::fopen(path_in_out("correctness.csv").c_str(), "w");
  if (!f) {
    std::printf("[error] cannot write %s\n", path_in_out("correctness.csv").c_str());
    return false;
  }
  const int n_head = count_csv_fields(K_CORRECTNESS_HEADER, false);
  bool ok = true;
  std::fprintf(f, "%s\n", K_CORRECTNESS_HEADER);
  for (size_t i = 0; i < rows.size(); ++i) {
    const CorrectnessRow& r = rows[i];
    // The note field is free text and contains commas, so it is quoted and any
    // embedded double quote is doubled -- otherwise the file is not valid CSV.
    std::string note = r.note;
    std::string quoted;
    for (size_t k = 0; k < note.size(); ++k) {
      if (note[k] == '"') quoted += '"';
      quoted += note[k];
    }
    char line[4096];
    std::snprintf(line, sizeof(line), "%s,%s,%lld,%d,%llu,%.10g,%.10g,%.6g,%.6g,%s,\"%s\"",
                  r.check.c_str(), r.backend.c_str(), r.paths, r.steps,
                  (unsigned long long)r.seed, r.value, r.expected, r.rel_err, r.z,
                  r.status.c_str(), quoted.c_str());
    if (count_csv_fields(line, true) != n_head) {
      std::printf("[error] row/header CSV column mismatch: %d fields vs %d header columns\n",
                  count_csv_fields(line, true), n_head);
      ok = false;
    }
    std::fprintf(f, "%s\n", line);
  }
  std::fclose(f);
  return ok;
}

// -----------------------------------------------------------------------------
//  Correctness command
// -----------------------------------------------------------------------------

static const double GBM_Z_LIMIT = 4.0;  // ~1e-4 two-sided false-failure rate

static void add_check(std::vector<CorrectnessRow>& rows, bool ok, const std::string& check,
                      const std::string& backend, long long paths, int steps, uint64_t seed,
                      double value, double expected, double z, const std::string& note) {
  CorrectnessRow r;
  r.check = check;
  r.backend = backend;
  r.paths = paths;
  r.steps = steps;
  r.seed = seed;
  r.value = value;
  r.expected = expected;
  r.rel_err = (expected != 0.0) ? (value - expected) / expected : 0.0;
  r.z = z;
  r.status = ok ? "PASS" : "FAIL";
  r.note = note;
  rows.push_back(r);
}

static int cmd_verify(int argc, char** argv) {
  const long long n_paths = get_opt_ll(argc, argv, "--paths", 200000);
  const int n_seeds = (int)get_opt_ll(argc, argv, "--seeds", 3);
  const int steps = (int)get_opt_ll(argc, argv, "--steps", 252);
  g_out_dir = get_opt(argc, argv, "--out", "results");
  ensure_dir(g_out_dir);

  SimParams p;
  p.S0 = 100.0;
  p.mu = 0.05;
  p.sigma = 0.2;
  p.T = 1.0;
  p.steps = steps;

  const double E = analytic_mean(p);
  const double V = analytic_var(p);
  const double sem = se_mean(p, n_paths);
  const double sev = se_var(p, n_paths);

  std::vector<CorrectnessRow> rows;
  std::printf("== Correctness verification =====================================\n");
  std::printf("model   : GBM  S(t+dt)=S(t)*exp((mu-0.5*sigma^2)dt + sigma*sqrt(dt)*Z)\n");
  std::printf("params  : S0=%.3f mu=%.3f sigma=%.3f T=%.3f steps=%d paths=%lld seeds=%d\n", p.S0,
              p.mu, p.sigma, p.T, p.steps, n_paths, n_seeds);
  std::printf("analytic: E[S(T)]=%.6f  Var[S(T)]=%.6f  sd=%.6f\n", E, V, std::sqrt(V));
  std::printf("s.e.    : mean=%.6f  variance=%.6f  (z-limit |z| < %.1f)\n\n", sem, sev,
              GBM_Z_LIMIT);

  bool all_ok = true;
  int n_pass = 0, n_fail = 0, n_skip = 0;

  // --- per-seed distributional checks -------------------------------------
  for (int k = 0; k < n_seeds; ++k) {
    const uint64_t seed = 12345ULL + (uint64_t)k;
    std::vector<float> cpu_px;

    // ---- CPU
    double cpu_ms = cpu_simulate(p, n_paths, seed, cpu_px);
    Stats cs = compute_stats(cpu_px);
    {
      bool ok = (cs.n == n_paths) && (cs.n_nonfinite == 0) && (cs.n_nonpositive == 0);
      add_check(rows, ok, "output_valid", "cpu", n_paths, p.steps, seed, (double)cs.n,
                (double)n_paths, 0.0,
                "size + all finite + all > 0 (nonfinite=" + std::to_string(cs.n_nonfinite) +
                    ", nonpositive=" + std::to_string(cs.n_nonpositive) + ")");
      if (!ok) all_ok = false;
      ok ? n_pass++ : n_fail++;
    }
    {
      // Restrict the moment checks to the valid subset.
      double z = (cs.mean - E) / sem;
      bool ok = std::fabs(z) < GBM_Z_LIMIT;
      add_check(rows, ok, "mean_vs_analytic", "cpu", n_paths, p.steps, seed, cs.mean, E, z,
                "H0: sample mean == S0*exp(mu*T); tolerance = 4 analytical s.e.");
      if (!ok) all_ok = false;
      ok ? n_pass++ : n_fail++;
    }
    {
      double z = (sev > 0.0) ? (cs.var - V) / sev : 0.0;
      bool ok = (sev > 0.0) && (std::fabs(z) < GBM_Z_LIMIT);
      add_check(rows, ok, "variance_vs_analytic", "cpu", n_paths, p.steps, seed, cs.var, V, z,
                "log-normal central-4th-moment based s.e.");
      if (!ok) all_ok = false;
      ok ? n_pass++ : n_fail++;
    }
    std::printf("[cpu] seed=%llu  mean=%8.4f (E=%.4f, z=%+6.2f)  sd=%7.3f  time=%7.1f ms\n",
                (unsigned long long)seed, cs.mean, E, (cs.mean - E) / sem, cs.sd, cpu_ms);

#if GBM_HAS_CUDA
    // ---- GPU
    std::vector<float> gpu_px;
    double gpu_ms = -1.0;
    std::string err;
    bool launched = false;
    for (int rep = 0; rep < 3 && !launched; ++rep) {  // 3 attempts: warm-up then measure
      err.clear();
      launched = gpu_run_once(p, n_paths, 256, seed, GPU_KERNEL_ONLY, gpu_px, &gpu_ms, &err);
      if (!launched) std::printf("[warn] gpu attempt %d failed: %s\n", rep + 1, err.c_str());
    }
    if (!launched) {
      add_check(rows, false, "kernel_launch", "gpu", n_paths, p.steps, seed, 0, 0, 0, err);
      std::printf("[gpu] seed=%llu  LAUNCH FAILED: %s\n", (unsigned long long)seed, err.c_str());
      all_ok = false;
      n_fail++;
    } else {
      Stats gs = compute_stats(gpu_px);
      {
        bool ok = (gs.n == n_paths) && (gs.n_nonfinite == 0) && (gs.n_nonpositive == 0);
        add_check(rows, ok, "output_valid", "gpu", n_paths, p.steps, seed, (double)gs.n,
                  (double)n_paths, 0.0,
                  "size + all finite + all > 0 (nonfinite=" + std::to_string(gs.n_nonfinite) +
                      ", nonpositive=" + std::to_string(gs.n_nonpositive) + ")");
        if (!ok) all_ok = false;
        ok ? n_pass++ : n_fail++;
      }
      {
        double z = (gs.mean - E) / sem;
        bool ok = std::fabs(z) < GBM_Z_LIMIT;
        add_check(rows, ok, "mean_vs_analytic", "gpu", n_paths, p.steps, seed, gs.mean, E, z,
                  "H0: sample mean == S0*exp(mu*T); tolerance = 4 analytical s.e.");
        if (!ok) all_ok = false;
        ok ? n_pass++ : n_fail++;
      }
      {
        double z = (sev > 0.0) ? (gs.var - V) / sev : 0.0;
        bool ok = (sev > 0.0) && (std::fabs(z) < GBM_Z_LIMIT);
        add_check(rows, ok, "variance_vs_analytic", "gpu", n_paths, p.steps, seed, gs.var, V, z,
                  "log-normal central-4th-moment based s.e.");
        if (!ok) all_ok = false;
        ok ? n_pass++ : n_fail++;
      }
      {
        // Cross-implementation agreement on the MEAN only (different RNGs, so
        // element-by-element equality is not a meaningful requirement).
        double z = (cs.mean - gs.mean) / std::sqrt(2.0) / sem;
        bool ok = std::fabs(z) < GBM_Z_LIMIT;
        add_check(rows, ok, "mean_cpu_vs_gpu", "cpu_gpu", n_paths, p.steps, seed,
                  cs.mean - gs.mean, 0.0, z,
                  "two independent samples, s.e. = sqrt(2)*s.e.(mean); elementwise equality NOT required");
        if (!ok) all_ok = false;
        ok ? n_pass++ : n_fail++;
        add_check(rows, true, "elementwise_cpu_vs_gpu", "cpu_gpu", n_paths, p.steps, seed, 0, 0, 0,
                  "SKIPPED BY DESIGN: CPU uses mt19937_64+Box-Muller (double), GPU uses "
                  "Philox4x32 (float); the two draw different normal streams");
        n_skip++;
      }
      std::printf("[gpu] seed=%llu  mean=%8.4f (E=%.4f, z=%+6.2f)  sd=%7.3f  kernel=%7.3f ms\n",
                  (unsigned long long)seed, gs.mean, E, (gs.mean - E) / sem, gs.sd, gpu_ms);

      // ---- determinism: identical seed must reproduce identical output -----
      if (k == 0) {
        std::vector<float> again;
        double ms2 = 0.0;
        std::string e2;
        bool ok2 = gpu_run_once(p, n_paths, 256, seed, GPU_KERNEL_ONLY, again, &ms2, &e2);
        if (ok2) {
          double maxdiff = 0.0;
          for (size_t i = 0; i < again.size(); ++i)
            maxdiff = std::max(maxdiff, (double)std::fabs((double)again[i] - (double)gpu_px[i]));
          bool ok = (maxdiff == 0.0);
          add_check(rows, ok, "determinism_same_seed", "gpu", n_paths, p.steps, seed, maxdiff, 0.0,
                    0.0, "bitwise reproducibility for a fixed (seed, path count, block size)");
          if (!ok) all_ok = false;
          ok ? n_pass++ : n_fail++;
        }
        // Different seed -> different sample (guards against a constant stream).
        std::vector<float> other;
        double ms3 = 0.0;
        std::string e3;
        bool ok3 = gpu_run_once(p, n_paths, 256, seed + 7777ULL, GPU_KERNEL_ONLY, other, &ms3, &e3);
        if (ok3) {
          double d = 0.0;
          for (size_t i = 0; i < other.size(); ++i)
            d = std::max(d, (double)std::fabs((double)other[i] - (double)gpu_px[i]));
          bool ok = (d > 0.0);
          add_check(rows, ok, "different_seed_differs", "gpu", n_paths, p.steps, seed + 7777ULL, d,
                    0.0, 0.0, "max |dS| between two seeds must be > 0");
          if (!ok) all_ok = false;
          ok ? n_pass++ : n_fail++;
        }
      }
    }
#else
    (void)0;  // CPU-only build: no GPU checks in this loop
#endif
  }

  // --- edge cases (small, structural) --------------------------------------
  {
    const long long small = 1000;  // not a multiple of 256 -> 24 masked lanes
    SimParams ps = p;
    ps.steps = 252;
    std::vector<float> px;
    double ms = 0.0;
    std::string err;

#if GBM_HAS_CUDA
    bool ok = gpu_run_once(ps, small, 256, 999ULL, GPU_KERNEL_ONLY, px, &ms, &err);
    Stats s = compute_stats(px);
    double z = (s.mean - analytic_mean(ps)) / se_mean(ps, small);
    bool structural = ok && (s.n == small) && (s.n_nonfinite == 0) && (s.n_nonpositive == 0);
    bool statistical = structural && (std::fabs(z) < GBM_Z_LIMIT);
    add_check(rows, structural, "non_divisible_paths_structure", "gpu", small, 252, 999ULL,
              (double)s.n, (double)small, 0.0,
              "paths=1000, block=256 -> 24 threads must exit via the bounds guard");
    structural ? n_pass++ : n_fail++;
    add_check(rows, statistical, "non_divisible_paths_mean", "gpu", small, 252, 999ULL, s.mean,
              analytic_mean(ps), z, "loose mean check at N=1000");
    statistical ? n_pass++ : n_fail++;
    if (!structural || !statistical) all_ok = false;
    std::printf("[gpu] paths=1000 block=256 -> size=%lld finite_ok=%d mean_z=%+.2f\n", s.n,
                (int)structural, z);
#endif
    // CPU structural check on the same tiny case (independent of CUDA presence)
    std::vector<float> cpx;
    cpu_simulate(ps, small, 999ULL, cpx);
    Stats cs2 = compute_stats(cpx);
    bool okc = (cs2.n == small) && (cs2.n_nonfinite == 0) && (cs2.n_nonpositive == 0);
    add_check(rows, okc, "non_divisible_paths_structure", "cpu", small, 252, 999ULL, (double)cs2.n,
              (double)small, 0.0, "same case on the CPU baseline");
    okc ? n_pass++ : n_fail++;
    if (!okc) all_ok = false;
  }
  // single path
  {
    SimParams ps = p;
    std::vector<float> px;
    double ms = 0.0;
    std::string err;
#if GBM_HAS_CUDA
    bool ok = gpu_run_once(ps, 1, 256, 7ULL, GPU_KERNEL_ONLY, px, &ms, &err);
    bool good = ok && px.size() == 1 && std::isfinite((double)px[0]) && px[0] > 0.0f;
    add_check(rows, good, "single_path", "gpu", 1, p.steps, 7ULL, good ? (double)px[0] : 0.0, 0.0,
              0.0, "paths=1, block=256: 255 inactive threads");
    good ? n_pass++ : n_fail++;
    if (!good) all_ok = false;
#else
    SimParams ps2 = ps;
    std::vector<float> c1;
    cpu_simulate(ps2, 1, 7ULL, c1);
    bool good = c1.size() == 1 && std::isfinite((double)c1[0]) && c1[0] > 0.0f;
    add_check(rows, good, "single_path", "cpu", 1, p.steps, 7ULL, (double)c1[0], 0.0, 0.0,
              "paths=1 on the CPU baseline");
    good ? n_pass++ : n_fail++;
    if (!good) all_ok = false;
#endif
  }
#if GBM_HAS_CUDA
  // zero paths must not crash
  {
    std::vector<float> px;
    double ms = -1.0;
    std::string err;
    bool ok = gpu_run_once(p, 0, 256, 1ULL, GPU_KERNEL_ONLY, px, &ms, &err);
    bool good = ok && px.empty();
    add_check(rows, good, "zero_paths", "gpu", 0, p.steps, 1ULL, 0, 0, 0,
              "paths=0 -> no launch, empty output, no crash");
    good ? n_pass++ : n_fail++;
    if (!good) all_ok = false;
  }
#endif
  // step-count invariance: the terminal distribution does not depend on `steps`
  {
    const long long n = std::max<long long>(50000, n_paths / 4);
    const int step_list[3] = {1, 21, 252};
    for (int si = 0; si < 3; ++si) {
      SimParams ps = p;
      ps.steps = step_list[si];
      std::vector<float> px;
      double ms = -1.0;
      std::string err;
      bool ok = true;
#if GBM_HAS_CUDA
      ok = gpu_run_once(ps, n, 256, 4242ULL, GPU_KERNEL_ONLY, px, &ms, &err);
#else
      cpu_simulate(ps, n, 4242ULL, px);
#endif
      Stats s = compute_stats(px);
      double z = (s.mean - analytic_mean(ps)) / se_mean(ps, n);
      bool good = ok && (std::fabs(z) < GBM_Z_LIMIT) && (s.n_nonfinite == 0) &&
                  (s.n_nonpositive == 0);
      add_check(rows, good, "steps_invariance", GBM_HAS_CUDA ? "gpu" : "cpu", n, ps.steps, 4242ULL,
                s.mean, analytic_mean(ps), z, "E[S(T)] must not depend on the number of Euler steps");
      good ? n_pass++ : n_fail++;
      if (!good) all_ok = false;
    }
  }

  const bool csv_ok = write_correctness_csv(rows);
  if (!csv_ok) {
    all_ok = false;
    n_fail++;
  }

  std::printf("\n-- summary: %d PASS, %d FAIL, %d SKIPPED (by design) --\n", n_pass, n_fail, n_skip);
  std::printf("-- wrote %s --\n", path_in_out("correctness.csv").c_str());
  if (!GBM_HAS_CUDA)
    std::printf("NOTE: built with -DGBM_CPU_ONLY, so every GPU check was not executed.\n");
  std::printf(all_ok ? "RESULT: ALL CHECKS PASSED\n" : "RESULT: FAILURES PRESENT\n");
  return all_ok ? 0 : 1;
}

// -----------------------------------------------------------------------------
//  Benchmark command: CPU vs GPU over sizes, and GPU over block sizes
// -----------------------------------------------------------------------------

struct BenchSizeRow {
  long long paths;
  int steps;
  int block;
  double cpu_ms;
  double gpu_kernel_ms;
  double gpu_e2e_ms;
  double speedup_kernel;
  double speedup_e2e;
  double gpu_paths_per_sec;
  int cpu_reps;
  int gpu_reps;
};

struct BenchBlockRow {
  long long paths;
  int steps;
  int block;
  double gpu_kernel_ms;
  double gpu_e2e_ms;
  double gpu_paths_per_sec;
  double paths_per_sec_per_step;  // terminal-step throughput normalised by steps
  int reps;
};

static int cmd_bench(int argc, char** argv) {
  g_out_dir = get_opt(argc, argv, "--out", "results");
  ensure_dir(g_out_dir);

  const std::string sizes_s = get_opt(argc, argv, "--sizes", "10000,100000,1000000");
  const std::string blocks_s = get_opt(argc, argv, "--blocks", "64,128,256,512");
  const int steps = (int)get_opt_ll(argc, argv, "--steps", 252);
  const int reps = (int)get_opt_ll(argc, argv, "--reps", 5);
  const int cpu_reps_cfg = (int)get_opt_ll(argc, argv, "--cpu-reps", 3);
  const int ref_block = (int)get_opt_ll(argc, argv, "--ref-block", 256);

  std::vector<long long> sizes = parse_int_list(sizes_s);
  std::vector<long long> blocks = parse_int_list(blocks_s);
  if (sizes.empty() || blocks.empty()) {
    std::printf("empty --sizes/--blocks\n");
    return 2;
  }

  SimParams p;
  p.S0 = 100.0;
  p.mu = 0.05;
  p.sigma = 0.2;
  p.T = 1.0;
  p.steps = steps;

  std::printf("== Benchmarks ===================================================\n");
  std::printf("steps=%d  gpu reps=%d  cpu reps=%d  block sweep=%s  sizes=%s\n", steps, reps,
              cpu_reps_cfg, blocks_s.c_str(), sizes_s.c_str());
  std::printf("method: warm-up run, then median of %d CUDA-event kernel-only timings; GPU e2e = "
              "median of %d runs incl. cudaMalloc/launch/D2H copy/cudaFree\n\n",
              reps, reps);

#if GBM_HAS_CUDA
  gpu_warmup(p, ref_block);  // size/arch warm-up before any timed run
#endif

  std::vector<BenchSizeRow> size_rows;
  std::vector<BenchBlockRow> block_rows;

  for (size_t si = 0; si < sizes.size(); ++si) {
    const long long n = sizes[si];
    const int cpu_reps = std::max(1, cpu_reps_cfg);

    // ---------------- CPU baseline -------------------------------------
    std::vector<double> cpu_times;
    std::vector<float> cpu_px;
    Stats cpu_stats;
    for (int r = 0; r < cpu_reps; ++r) {
      const uint64_t seed = 2024ULL + (uint64_t)r;
      cpu_times.push_back(cpu_simulate(p, n, seed, cpu_px));
      cpu_stats = compute_stats(cpu_px);
    }
    const double cpu_ms = median(cpu_times);

    std::printf("paths=%-9lld  CPU  median=%9.3f ms  (reps=%d, mean=%.3f, sd=%.3f)\n", n, cpu_ms,
                cpu_reps, cpu_stats.mean, cpu_stats.sd);

    // The CPU row is recorded in every build; GPU columns stay at -1/0 when no
    // CUDA device is compiled in, so the CSV schema is always the same.
    BenchSizeRow row;
    row.paths = n;
    row.steps = steps;
    row.block = ref_block;
    row.cpu_ms = cpu_ms;
    row.gpu_kernel_ms = -1.0;
    row.gpu_e2e_ms = -1.0;
    row.speedup_kernel = 0.0;
    row.speedup_e2e = 0.0;
    row.gpu_paths_per_sec = 0.0;
    row.cpu_reps = cpu_reps;
    row.gpu_reps = 0;

#if GBM_HAS_CUDA
    // ---------------- GPU: kernel-only + end-to-end --------------------
    std::vector<double> kern_times, e2e_times;
    std::vector<float> gpu_px;
    std::string err;
    for (int r = 0; r < reps; ++r) {
      const uint64_t seed = 2024ULL;  // fixed seed -> identical work every rep
      double ms = 0.0;
      err.clear();
      if (!gpu_run_once(p, n, ref_block, seed, GPU_KERNEL_ONLY, gpu_px, &ms, &err)) {
        std::printf("  [error] kernel run failed: %s\n", err.c_str());
        break;
      }
      kern_times.push_back(ms);
    }
    for (int r = 0; r < reps; ++r) {
      double ms = 0.0;
      std::vector<float> tmp;
      err.clear();
      if (!gpu_run_once(p, n, ref_block, 2024ULL, GPU_END_TO_END, tmp, &ms, &err)) {
        std::printf("  [error] e2e run failed: %s\n", err.c_str());
        break;
      }
      e2e_times.push_back(ms);
    }
    const double gk = median(kern_times);
    const double ge = median(e2e_times);

    row.gpu_kernel_ms = gk;
    row.gpu_e2e_ms = ge;
    row.speedup_kernel = (gk > 0.0) ? cpu_ms / gk : 0.0;
    row.speedup_e2e = (ge > 0.0) ? cpu_ms / ge : 0.0;
    row.gpu_paths_per_sec = (gk > 0.0) ? (double)n / (gk / 1000.0) : 0.0;
    row.gpu_reps = reps;
    std::printf("paths=%-9lld  GPU  kernel=%9.3f ms  e2e=%9.3f ms  kernel speedup=%7.2fx  "
                "e2e speedup=%7.2fx\n\n",
                n, gk, ge, row.speedup_kernel, row.speedup_e2e);

    // ---------------- GPU: block-size sweep at this size ---------------
    for (size_t bi = 0; bi < blocks.size(); ++bi) {
      const int b = (int)blocks[bi];
      std::vector<double> kt, et;
      for (int r = 0; r < reps; ++r) {
        double ms = 0.0;
        err.clear();
        if (!gpu_run_once(p, n, b, 2024ULL, GPU_KERNEL_ONLY, gpu_px, &ms, &err)) break;
        kt.push_back(ms);
      }
      for (int r = 0; r < reps; ++r) {
        double ms = 0.0;
        std::vector<float> tmp;
        err.clear();
        if (!gpu_run_once(p, n, b, 2024ULL, GPU_END_TO_END, tmp, &ms, &err)) break;
        et.push_back(ms);
      }
      if (kt.empty()) continue;
      BenchBlockRow br;
      br.paths = n;
      br.steps = steps;
      br.block = b;
      br.gpu_kernel_ms = median(kt);
      br.gpu_e2e_ms = et.empty() ? -1.0 : median(et);
      br.gpu_paths_per_sec = br.gpu_kernel_ms > 0.0 ? (double)n / (br.gpu_kernel_ms / 1000.0) : 0.0;
      br.paths_per_sec_per_step = br.gpu_paths_per_sec / (double)steps;
      br.reps = (int)kt.size();
      block_rows.push_back(br);
      std::printf("    block=%-4d kernel=%9.3f ms  e2e=%9.3f ms  %10.3f Mpath/s\n", b,
                  br.gpu_kernel_ms, br.gpu_e2e_ms, br.gpu_paths_per_sec / 1e6);
    }
    std::printf("\n");
#else
    std::printf("  (CPU-only build: no GPU benchmarks for this size)\n\n");
#endif
    size_rows.push_back(row);
  }

  // ---------------- write CSVs ------------------------------------------
  {
    FILE* f = std::fopen(path_in_out("benchmark_sizes.csv").c_str(), "w");
    if (f) {
      std::fprintf(f,
                   "paths,steps,block,cpu_ms,gpu_kernel_ms,gpu_e2e_ms,speedup_kernel,speedup_e2e,"
                   "gpu_paths_per_sec,cpu_reps,gpu_reps\n");
      for (size_t i = 0; i < size_rows.size(); ++i) {
        const BenchSizeRow& r = size_rows[i];
        std::fprintf(f, "%lld,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.1f,%d,%d\n", r.paths, r.steps,
                     r.block, r.cpu_ms, r.gpu_kernel_ms, r.gpu_e2e_ms, r.speedup_kernel,
                     r.speedup_e2e, r.gpu_paths_per_sec, r.cpu_reps, r.gpu_reps);
      }
      std::fclose(f);
      std::printf("wrote %s\n", path_in_out("benchmark_sizes.csv").c_str());
    }
    FILE* g = std::fopen(path_in_out("benchmark_blocks.csv").c_str(), "w");
    if (g) {
      std::fprintf(g,
                   "paths,steps,block,gpu_kernel_ms,gpu_e2e_ms,gpu_paths_per_sec,"
                   "paths_per_sec_per_step,reps\n");
      for (size_t i = 0; i < block_rows.size(); ++i) {
        const BenchBlockRow& r = block_rows[i];
        std::fprintf(g, "%lld,%d,%d,%.6f,%.6f,%.1f,%.3f,%d\n", r.paths, r.steps, r.block,
                     r.gpu_kernel_ms, r.gpu_e2e_ms, r.gpu_paths_per_sec, r.paths_per_sec_per_step,
                     r.reps);
      }
      std::fclose(g);
      std::printf("wrote %s\n", path_in_out("benchmark_blocks.csv").c_str());
    }
  }
  return 0;
}

// -----------------------------------------------------------------------------
//  RNG / launch-overhead experiment
//  Sweeping the step count at a fixed path count gives  t(steps) = a + b*steps.
//  `b` is the marginal per-step cost (normal draws + exp + multiply) and `a`
//  captures the fixed per-launch cost: kernel launch + `curand_init` for every
//  thread. The fit is produced by ordinary least squares below.
// -----------------------------------------------------------------------------
static int cmd_rng(int argc, char** argv) {
  g_out_dir = get_opt(argc, argv, "--out", "results");
  ensure_dir(g_out_dir);

  const long long n = get_opt_ll(argc, argv, "--paths", 1000000);
  const int block = (int)get_opt_ll(argc, argv, "--block", 256);
  const int reps = (int)get_opt_ll(argc, argv, "--reps", 5);
  std::vector<long long> steps_list = parse_int_list(get_opt(argc, argv, "--steps", "1,4,16,64,252"));

  SimParams p;
  p.S0 = 100.0;
  p.mu = 0.05;
  p.sigma = 0.2;
  p.T = 1.0;

  std::printf("== RNG / launch overhead study ==================================\n");
  std::printf("paths=%lld block=%d reps=%d steps=%s\n\n", n, block, reps,
              join_csv(steps_list).c_str());

#if GBM_HAS_CUDA
  gpu_warmup(p, block);
#endif

  std::vector<double> xs, ys;
  FILE* f = std::fopen(path_in_out("rng_overhead.csv").c_str(), "w");
  if (f)
    std::fprintf(f, "paths,block,steps,kernel_ms,us_per_step,ns_per_path_step,gpu_paths_per_sec\n");

  for (size_t i = 0; i < steps_list.size(); ++i) {
    const int st = (int)steps_list[i];
    if (st < 1) continue;
    SimParams ps = p;
    ps.steps = st;

    double ms = -1.0;
#if GBM_HAS_CUDA
    std::vector<double> times;
    std::vector<float> px;
    std::string err;
    for (int r = 0; r < reps; ++r) {
      double t = 0.0;
      err.clear();
      if (!gpu_run_once(ps, n, block, 2024ULL, GPU_KERNEL_ONLY, px, &t, &err)) {
        std::printf("  [error] %s\n", err.c_str());
        break;
      }
      times.push_back(t);
    }
    if (!times.empty()) ms = median(times);
#else
    std::vector<float> px;
    ms = cpu_simulate(ps, n, 2024ULL, px);
#endif
    if (ms < 0.0) continue;
    xs.push_back((double)st);
    ys.push_back(ms);
    std::printf("  steps=%-4d kernel=%9.4f ms  %8.4f us/step  %7.3f ns/path/step  %10.1f Mpath/s\n",
                st, ms, ms * 1000.0 / (double)st,
                ms * 1e6 / ((double)st * (double)n), (double)n / (ms / 1000.0) / 1e6);
    if (f)
      std::fprintf(f, "%lld,%d,%d,%.6f,%.4f,%.4f,%.1f\n", n, block, st, ms,
                   ms * 1000.0 / (double)st, ms * 1e6 / ((double)st * (double)n),
                   (double)n / (ms / 1000.0));
  }
  if (f) std::fclose(f);

  // Ordinary least squares fit  t = a + b*steps
  if (xs.size() >= 2) {
    const double m = (double)xs.size();
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (size_t i = 0; i < xs.size(); ++i) {
      sx += xs[i];
      sy += ys[i];
      sxx += xs[i] * xs[i];
      sxy += xs[i] * ys[i];
    }
    const double denom = m * sxx - sx * sx;
    const double b = (denom != 0.0) ? (m * sxy - sx * sy) / denom : 0.0;
    const double a = (sy - b * sx) / m;
    double ss_tot = 0, ss_res = 0;
    const double ybar = sy / m;
    for (size_t i = 0; i < xs.size(); ++i) {
      const double fit = a + b * xs[i];
      ss_tot += (ys[i] - ybar) * (ys[i] - ybar);
      ss_res += (ys[i] - fit) * (ys[i] - fit);
    }
    const double r2 = (ss_tot > 0.0) ? 1.0 - ss_res / ss_tot : 1.0;
    const double fixed_frac = (a + b * 252.0 != 0.0) ? a / (a + b * 252.0) : 0.0;

    char buf[1024];
#if GBM_HAS_CUDA
    const char* src_label = "kernel-only CUDA-event timings";
    const char* fixed_label = "fixed per-launch cost (kernel launch + curand_init for all";
    const char* per_path_label = "per-path curand_init+launch cost";
#else
    const char* src_label = "CPU wall-clock timings (CPU-only build)";
    const char* fixed_label = "fixed per-run cost (RNG seeding for all";
    const char* per_path_label = "per-path RNG-seeding cost";
#endif
    std::snprintf(buf, sizeof(buf),
                  "OLS fit over the step sweep (%s)\n"
                  "  t_ms(steps) = a + b*steps\n"
                  "  a = %.6f ms   %s %lld paths)\n"
                  "  b = %.8f ms   marginal cost of one extra step for all %lld paths\n"
                  "  R^2 = %.6f  (points = %d)\n"
                  "  at 252 steps: a + 252*b = %.6f ms, of which %.1f%% is fixed setup overhead\n"
                  "  %s ~ a / n = %.4f ns\n",
                  src_label, a, fixed_label, n, b, n, r2, (int)xs.size(), a + 252.0 * b,
                  100.0 * fixed_frac, per_path_label, a * 1e6 / (double)n);
    std::printf("\n%s", buf);
    FILE* ff = std::fopen(path_in_out("rng_overhead_fit.txt").c_str(), "w");
    if (ff) {
      std::fprintf(ff, "%s", buf);
      std::fclose(ff);
      std::printf("wrote %s (and rng_overhead.csv)\n", path_in_out("rng_overhead_fit.txt").c_str());
    }
  }
  return 0;
}

// -----------------------------------------------------------------------------
//  Device / environment information
// -----------------------------------------------------------------------------
static int cmd_info(int argc, char** argv) {
  g_out_dir = get_opt(argc, argv, "--out", "results");
  ensure_dir(g_out_dir);
  FILE* f = std::fopen(path_in_out("environment_cuda.txt").c_str(), "w");
  std::string log;

#if GBM_HAS_CUDA
  int dev_count = 0;
  cudaError_t rc = cudaGetDeviceCount(&dev_count);
  char buf[4096];
  std::snprintf(buf, sizeof(buf), "cudaGetDeviceCount -> %d (%s)\n", dev_count,
                cudaGetErrorString(rc));
  log += buf;
  if (rc == cudaSuccess && dev_count > 0) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int rt_ver = 0, drv_ver = 0;
    cudaRuntimeGetVersion(&rt_ver);
    cudaDriverGetVersion(&drv_ver);
    std::snprintf(buf, sizeof(buf),
                  "device                        : %s\n"
                  "compute capability            : %d.%d\n"
                  "multiprocessors (SMs)         : %d\n"
                  "max threads per SM            : %d\n"
                  "max threads per block         : %d\n"
                  "shared memory per block       : %zu bytes\n"
                  "warp size                     : %d\n"
                  "registers per SM              : %d\n"
                  "global memory                 : %.2f GiB\n"
                  "memory bus width              : %d bit\n"
                  "memory clock rate             : %.0f MHz\n"
                  "L2 cache size                 : %d bytes\n"
                  "concurrent kernels            : %d\n"
                  "cudaRuntimeGetVersion         : %d\n"
                  "cudaDriverGetVersion          : %d\n"
                  "compiled with nvcc            : %d.%d\n",
                  prop.name, prop.major, prop.minor, prop.multiProcessorCount,
                  prop.maxThreadsPerMultiProcessor, prop.maxThreadsPerBlock,
                  (size_t)prop.sharedMemPerBlock, prop.warpSize, prop.regsPerMultiprocessor,
                  (double)prop.totalGlobalMem / 1073741824.0, prop.memoryBusWidth,
                  (double)prop.memoryClockRate / 1000.0, prop.l2CacheSize, prop.concurrentKernels,
                  rt_ver, drv_ver,
#if defined(__CUDACC__)
                  __CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__
#else
                  0, 0
#endif
    );
    log += buf;
  }
#else
  log += "built with -DGBM_CPU_ONLY: CUDA support not compiled in\n";
#endif
  log += "host: see nvidia-smi / lscpu output captured by the notebook\n";

  std::printf("%s", log.c_str());
  if (f) {
    std::fprintf(f, "%s", log.c_str());
    std::fclose(f);
    std::printf("wrote %s\n", path_in_out("environment_cuda.txt").c_str());
  }
  return 0;
}

// -----------------------------------------------------------------------------
//  Smoke test: fast correctness + tiny benchmark. Run this before the full
//  benchmark matrix so a broken build fails in seconds instead of minutes.
// -----------------------------------------------------------------------------
static int cmd_smoke(int argc, char** argv) {
  g_out_dir = get_opt(argc, argv, "--out", "results");
  SimParams p;
  p.S0 = 100.0;
  p.mu = 0.05;
  p.sigma = 0.2;
  p.T = 1.0;
  p.steps = 252;
  const long long n = 100000;
  const double E = analytic_mean(p);
  const double sem = se_mean(p, n);
  bool ok = true;

  std::printf("== Smoke test ===================================================\n");
  std::vector<float> px;
  double t_cpu = cpu_simulate(p, n, 1ULL, px);
  Stats cs = compute_stats(px);
  double zc = (cs.mean - E) / sem;
  std::printf("cpu: %lld paths, %6.1f ms, mean=%.4f (E=%.4f, z=%+.2f)\n", n, t_cpu, cs.mean, E, zc);
  ok = ok && std::fabs(zc) < GBM_Z_LIMIT && cs.n_nonfinite == 0 && cs.n_nonpositive == 0;

#if GBM_HAS_CUDA
  gpu_warmup(p, 128);
  double ms = -1.0, e2e = -1.0;
  std::string err;
  std::vector<float> gx, gy;
  bool launched = gpu_run_once(p, n, 128, 1ULL, GPU_KERNEL_ONLY, gx, &ms, &err);
  Stats gs = compute_stats(gx);
  double zg = (gs.mean - E) / sem;
  gpu_run_once(p, n, 128, 1ULL, GPU_END_TO_END, gy, &e2e, &err);
  std::printf("gpu: %lld paths, block=128, kernel=%7.3f ms, e2e=%7.3f ms, mean=%.4f (z=%+.2f)\n", n,
              ms, e2e, gs.mean, zg);
  std::printf("speedup (kernel): %.2fx\n", (ms > 0.0) ? t_cpu / ms : 0.0);
  ok = ok && launched && std::fabs(zg) < GBM_Z_LIMIT && gs.n_nonfinite == 0 &&
       gs.n_nonpositive == 0;
#else
  std::printf("gpu: SKIPPED (built with -DGBM_CPU_ONLY)\n");
#endif
  std::printf(ok ? "SMOKE OK\n" : "SMOKE FAILED\n");
  return ok ? 0 : 1;
}

static void usage() {
  std::printf(
      "Monte Carlo GBM (CUDA) -- commands:\n"
      "  info    [--out DIR]\n"
      "  smoke   [--out DIR]\n"
      "  verify  [--paths N] [--seeds K] [--steps S] [--out DIR]\n"
      "  bench   [--sizes a,b,c] [--blocks a,b,c] [--steps S] [--reps R] [--cpu-reps R]\n"
      "          [--ref-block B] [--out DIR]\n"
      "  rng     [--paths N] [--block B] [--steps a,b,c] [--reps R] [--out DIR]\n");
}

int main(int argc, char** argv) {
  if (argc < 2) {
    usage();
    return 2;
  }
  const std::string cmd = argv[1];
  if (cmd == "verify") return cmd_verify(argc, argv);
  if (cmd == "bench") return cmd_bench(argc, argv);
  if (cmd == "rng") return cmd_rng(argc, argv);
  if (cmd == "info") return cmd_info(argc, argv);
  if (cmd == "smoke") return cmd_smoke(argc, argv);
  usage();
  return 2;
}
