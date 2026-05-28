#!/bin/bash
# Скрипт нативной сборки zmq8_cubie на плате Radxa Cubie A7Z
#
# Использование:
#   ./build_native.sh

set -e

SCRIPT_DIR=$(cd $(dirname $0) && pwd)
BUILD_DIR=${SCRIPT_DIR}/build_native

echo "=== Native build zmq8_cubie ==="

mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}

cmake .. \
    -DNATIVE_BUILD=ON \
    -DTARGET_NAME=A733 \
    -DEXTERN_DEFINE_TARGET=ON

make -j$(nproc)
make install

cd ..
rm -rf build_native

echo "=== Build complete ==="
echo "Output: ${SCRIPT_DIR}/install/"
