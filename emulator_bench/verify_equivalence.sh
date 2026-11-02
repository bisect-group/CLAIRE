#!/usr/bin/env bash
# Checks that the CLAIRE wrapper matches the original code. Usage: verify_equivalence.sh [--gpu | --negative-control]
# Expected env: torch, numba, pandas, pyarrow, rxnfp, drfp, rdkit, transformers<5, setuptools<81, pytest.
# The script does not create envs. It needs no downloaded weights and no private data; rxnfp ships its model.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
if [[ ${1:-} == --negative-control ]]; then
    python -c "import torch,sys; sys.exit(not torch.cuda.is_available())" || { echo "FAIL: the negative control needs a CUDA device."; exit 1; }
    echo "Negative control: the bf16 gap run starts from weights times 1.05. The GPU test must FAIL."
    if out=$(EQUIV_NEGATIVE_CONTROL=1 python -m pytest tests/test_gpu_precision.py -q -p no:cacheprovider --color=no -s -m gpu 2>&1); then
        echo "$out"; echo "FAIL: the GPU test passed with the negative control (the bound is vacuous)."; exit 1
    fi
    grep -q "ratio=" <<<"$out" || { echo "$out"; echo "FAIL: the GPU test crashed before it measured a ratio."; exit 1; }
    grep "ratio=" <<<"$out"
    echo "Negative control failed as expected."
    exit 0
fi
files=(tests/test_equivalence.py)
marker=(-m "not gpu")
if [[ ${1:-} == --gpu ]]; then
    python -c "import torch,sys; sys.exit(not torch.cuda.is_available())" || { echo "FAIL: --gpu needs a CUDA device."; exit 1; }
    files=(tests/test_gpu_precision.py)  # the CPU tests need the CPU: they run without --gpu
    marker=(-s)
else
    export CUDA_VISIBLE_DEVICES=""
fi
out=$(python -m pytest "${files[@]}" -q -rs -p no:cacheprovider --color=no "${marker[@]}" 2>&1) || { echo "$out"; exit 1; }
echo "$out"
if grep -Eq "[0-9]+ skipped" <<<"$out"; then
    echo "FAIL: tests were skipped"
    exit 1
fi
