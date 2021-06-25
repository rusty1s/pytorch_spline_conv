#!/bin/bash

export PYTHON_VERSION=$2
export TORCH_VERSION=$2
export CUDA_VERSION=$3

export CONDA_PYTHON_CONSTRAINT="python==$PYTHON_VERSION"

export CONDA_PYTORCH_CONSTRAINT="pytorch==${TORCH_VERSION%.*}.*"

if [ "${CUDA_VERSION}" = "cpu" ]; then
  export CONDA_CUDATOOLKIT_CONSTRAINT="cpuonly"
else
  case $CUDA_VERSION in
    cu111)
      export CONDA_CUDATOOLKIT_CONSTRAINT="cudatoolkit==11.1.*"
      ;;
    cu102)
      export CONDA_CUDATOOLKIT_CONSTRAINT="cudatoolkit==10.2.*"
      ;;
    cu101)
      export CONDA_CUDATOOLKIT_CONSTRAINT="cudatoolkit==10.1.*"
      ;;
    *)
      echo "Unrecognized CUDA_VERSION=$CUDA_VERSION"
      exit 1
      ;;
  esac
fi

echo "PyTorch $TORCH_VERSION+$CUDA_VERSION"
echo "- $CONDA_PYTORCH_CONSTRAINT"
echo "- $CONDA_CUDATOOLKIT_CONSTRAINT"

"$CONDA/bin/conda" build . -c defaults -c nvidia -c pytorch --output-folder "$HOME/conda-bld/"
