#!/bin/bash

# Install NVIDIA drivers, see:
# https://github.com/pytorch/vision/blob/master/packaging/windows/internal/cuda_install.bat#L99-L102
curl -k -L "https://drive.google.com/u/0/uc?id=1injUyo3lnarMgWyRcXqKg4UGnN0ysmuq&export=download" --output "/tmp/gpu_driver_dlls.zip"
7z x "/tmp/gpu_driver_dlls.zip" -o"/c/Windows/System32"

export CUDA_SHORT=13.0
export CUDA_URL=https://developer.download.nvidia.com/compute/cuda/${CUDA_SHORT}.2/local_installers
export CUDA_FILE=cuda_${CUDA_SHORT}.2_windows.exe

# Install CUDA:
curl -k -L "${CUDA_URL}/${CUDA_FILE}" --output "${CUDA_FILE}"
echo ""
echo "Installing from ${CUDA_FILE}..."
PowerShell -Command "Start-Process -FilePath \"${CUDA_FILE}\" -ArgumentList \"-s\" -Wait -NoNewWindow"
echo "Done!"
rm -f "${CUDA_FILE}"
