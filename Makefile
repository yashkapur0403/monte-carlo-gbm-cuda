# Monte Carlo GBM simulator -- CUDA + CPU-only builds.
#
#   make                build the CUDA binary for ARCH (default sm_75, Kaggle T4)
#   make ARCH=sm_60     Tesla P100
#   make ARCH=sm_80     A100
#   make cpu            build the CPU-only baseline (no CUDA toolkit required)
#   make verify         CPU-only correctness run
#   make bench          full benchmark matrix (GPU build)
#   make notebook       regenerate notebooks/kaggle_gbm_montecarlo.ipynb from the source
#   make nbtest         run the notebook's download-gate cell against fixtures
#                       (needs IPython, pandas, matplotlib; no GPU needed)
#   make proftest       test the profiling cell: parsers against real tool-output formats,
#                       then the whole ladder with fake cuobjdump/nvidia-smi shims
#   make benchtest      test the notebook's steady-state benchmark guard against the committed
#                       CSVs and against injected artefacts (no GPU needed)
#   make checkzip ZIP=~/Downloads/gbm_montecarlo_results.zip
#                       verify a results archive produced by the Kaggle run
#   make check          notebook validation + benchmark guard + CPU-only correctness
#                       (no GPU needed)
#   make clean

NVCC      ?= nvcc
CXX       ?= g++
PYTHON    ?= python3
ARCH      ?= sm_75
NVCCFLAGS ?= -O3 -std=c++17 -Xptxas -v -lineinfo
CXXFLAGS  ?= -O3 -std=c++11 -DGBM_CPU_ONLY
LDLIBS    ?= -lcurand

BUILD   := build
SRC     := src/gbm_montecarlo.cu
GPU_BIN := $(BUILD)/gbm
CPU_BIN := $(BUILD)/gbm_cpu

.PHONY: all gpu cpu verify bench rng smoke info notebook validate nbtest proftest benchtest \
	checkzip check clean

all: gpu

gpu: $(GPU_BIN)

$(GPU_BIN): $(SRC)
	@mkdir -p $(BUILD)
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) -o $@ $< $(LDLIBS)

cpu: $(CPU_BIN)

$(CPU_BIN): $(SRC)
	@mkdir -p $(BUILD)
	$(CXX) -x c++ $(CXXFLAGS) -o $@ $<

info:   gpu ; @$(GPU_BIN) info   --out results
smoke:  gpu ; @$(GPU_BIN) smoke  --out results
verify: gpu ; @$(GPU_BIN) verify --paths 200000 --seeds 3 --steps 252 --out results
rng:    gpu ; @$(GPU_BIN) rng    --paths 1000000 --block 256 --steps 1,4,16,64,252 --reps 5 --out results

bench: gpu
	@$(GPU_BIN) bench --sizes 10000,100000,1000000 --blocks 64,128,256,512 \
	    --steps 252 --reps 5 --cpu-reps 3 --ref-block 256 --out results

notebook:
	$(PYTHON) tools/build_notebook.py

validate:
	$(PYTHON) tools/validate_notebook.py

nbtest: validate
	$(PYTHON) tools/test_download_cell.py

proftest: validate
	$(PYTHON) tools/test_profiling_cell.py
	$(PYTHON) tools/test_profiling_ladder.py

benchtest:
	$(PYTHON) tools/test_benchmark_guard.py

checkzip:
	$(PYTHON) tools/check_results_zip.py $(ZIP)

# Everything that can be checked without an NVIDIA GPU:
# notebook integrity, the benchmark guard, and the CPU half of the model and its harness.
check: validate benchtest cpu
	@$(CPU_BIN) verify --paths 200000 --seeds 2 --steps 252 --out results

# Only build outputs. results/ holds the measured artifacts that this repo commits as evidence,
# so it is never removed here -- a re-run overwrites it, and nothing else does.
clean:
	rm -rf $(BUILD)
	rm -f gbm_montecarlo_results.zip
