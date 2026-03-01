#!/bin/bash
# Compile TensorRT plugins for UniV2X
# Usage: ./compile_plugins.sh [TENSORRT_PATH] [GPU_SM]
#
# TENSORRT_PATH: path to TensorRT installation (default: /usr/local/TensorRT)
# GPU_SM:        GPU compute capability without the dot, e.g. 87 for Orin, 86 for A100
#                If not provided, the CMakeLists will auto-detect via nvcc.
#
# Example (DRIVE Orin):
#   ./compile_plugins.sh /path/to/TensorRT-10.x 87
# Example (x86 A100):
#   ./compile_plugins.sh /path/to/TensorRT-10.x 80

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TENSORRT_PATH="${1:-/usr/local/TensorRT}"
GPU_SM="${2:-}"

echo "Building TensorRT plugins for UniV2X..."
echo "  TensorRT path : ${TENSORRT_PATH}"
echo "  GPU SM        : ${GPU_SM:-auto-detect}"

cd "${SCRIPT_DIR}/build"

if [ -n "${GPU_SM}" ]; then
    cmake .. -DCMAKE_TENSORRT_PATH="${TENSORRT_PATH}" -DARCH="${GPU_SM}"
else
    cmake .. -DCMAKE_TENSORRT_PATH="${TENSORRT_PATH}"
fi

make -j$(nproc)
make install

echo "Done. Plugin library installed to: ${SCRIPT_DIR}/lib/"
