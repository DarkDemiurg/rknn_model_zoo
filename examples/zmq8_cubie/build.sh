#!/bin/bash
# Скрипт сборки zmq8_cubie для Radxa Cubie A7Z (Allwinner A733)
# Использует toolchain и библиотеки из awnpu_model_zoo
#
# Использование:
#   ./build.sh [TARGET]
#   TARGET по умолчанию: A733

set -e

TARGET_NAME=${1:-A733}
SCRIPT_DIR=$(cd $(dirname $0) && pwd)
MODEL_ZOO_HOME=${SCRIPT_DIR}/../../../../cubie/awnpu_model_zoo-v1.0.0-20260423-f562dd16
TOOLCHAIN_FILE=${MODEL_ZOO_HOME}/cmake_toolchain/compiler.toolchain-linux-aarch64.cmake

BUILD_DIR=${SCRIPT_DIR}/build_linux_aarch64

echo "=== Building zmq8_cubie for ${TARGET_NAME} ==="
echo "Toolchain: ${TOOLCHAIN_FILE}"

mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}

cmake -DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN_FILE} \
      -DUSE_EXTERN_TOOLCHAIN=ON \
      -DTARGET_NAME=${TARGET_NAME} \
      -DEXTERN_DEFINE_TARGET=ON \
      ..

make -j$(nproc)
make install

cd ..
rm -rf build_linux_aarch64

echo "=== Build complete ==="
echo "Output: ${SCRIPT_DIR}/install/"
