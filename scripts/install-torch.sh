#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

function has() {
  command -v "${@}" &> /dev/null
}

if has nvidia-smi; then
  conda install --channel="nvidia/label/cuda-11.7.1" --yes cuda
  pip install torch
else
  pip install --index-url="https://download.pytorch.org/whl/cpu" torch
fi
