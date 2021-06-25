#!/bin/bash

TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
TORCH_VERSION=${TORCH_VERSION%+*}
TORCH_VERSION=${TORCH_VERSION%.*}
CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)")

export CONDA_PYTORCH_CONSTRAINT="pytorch==$TORCH_VERSION.*"
export TORCH_VERSION="$TORCH_VERSION.0"

if [[ "$(uname)" == Darwin ]]; then
  export CONDA_CUDATOOLKIT_CONSTRAINT=""
  export CUDA_VERSION="cpu"
  export FORCE_ONLY_CPU=1
elif [ "${CUDA_VERSION}" = "None" ]; then
  export CONDA_CUDATOOLKIT_CONSTRAINT="cpuonly"
  export CUDA_VERSION="cpu"
  export FORCE_ONLY_CPU=1
else
  export CONDA_CUDATOOLKIT_CONSTRAINT="cudatoolkit==$CUDA_VERSION.*"
  export CUDA_VERSION="cu${CUDA_VERSION/./}"
  export FORCE_CUDA=1
fi

echo "PyTorch $TORCH_VERSION+$CUDA_VERSION:"
echo "$CONDA_PYTORCH_CONSTRAINT"
echo "$CONDA_CUDATOOLKIT_CONSTRAINT"

conda build . -c defaults -c nvidia -c pytorch
