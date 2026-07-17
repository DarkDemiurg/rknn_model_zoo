#!/bin/bash
# Установка NPU runtime библиотек (libVIPhal.so, libNBGlinker.so) на целевую систему
#
# Использование (от root):
#   sudo ./install_npu_libs.sh

set -e

SCRIPT_DIR=$(cd $(dirname $0) && pwd)
NPU_LIB_DIR=${SCRIPT_DIR}/3rdparty/npu_lib

if [ ! -d "${NPU_LIB_DIR}" ]; then
    echo "ERROR: NPU libs not found: ${NPU_LIB_DIR}"
    exit 1
fi

echo "=== Installing NPU runtime libraries ==="

cp ${NPU_LIB_DIR}/*.so* /usr/local/lib/
echo "Installed to /usr/local/lib/"

/sbin/ldconfig 2>/dev/null || ldconfig 2>/dev/null || echo "Warning: run 'sudo /sbin/ldconfig' manually"

echo "=== Done ==="
