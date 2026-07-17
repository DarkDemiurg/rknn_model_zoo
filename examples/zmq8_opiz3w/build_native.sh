#!/bin/bash
# Нативная сборка zmq8_opiz3w на Orange Pi Zero 3W (Allwinner A733)
#
# Использование:
#   ./build_native.sh

set -e

SCRIPT_DIR=$(cd $(dirname $0) && pwd)
BUILD_DIR=${SCRIPT_DIR}/build_native

echo "=== Native build zmq8_opiz3w ==="

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
echo "Output: ${SCRIPT_DIR}/install/zmq8_opiz3w_linux_a733/"
echo "Run YOLOv8:  ./install/zmq8_opiz3w_linux_a733/zmq8_opiz3w ./model/yolov8n_6_uint8_a733.nb 0"
echo "Run YOLOv11: ./install/zmq8_opiz3w_linux_a733/zmq8_opiz3w -v 11 ./model/yolo11n_6_uint8_a733.nb 0"
