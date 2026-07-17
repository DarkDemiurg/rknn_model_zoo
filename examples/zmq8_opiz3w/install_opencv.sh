#!/bin/bash
# Установка OpenCV 4.9.0 (aarch64) на целевую систему Radxa Cubie A7Z
# Распаковывает заголовки и библиотеки в /usr/local/
#
# Использование (от root):
#   sudo ./install_opencv.sh

set -e

SCRIPT_DIR=$(cd $(dirname $0) && pwd)
ARCHIVE=${SCRIPT_DIR}/3rdparty/opencv-4.9.0-aarch64-linux-sunxi-glibc.zip
TMPDIR=/tmp/opencv_install

if [ ! -f "${ARCHIVE}" ]; then
    echo "ERROR: Archive not found: ${ARCHIVE}"
    exit 1
fi

echo "=== Installing OpenCV 4.9.0 aarch64 ==="

rm -rf ${TMPDIR}
mkdir -p ${TMPDIR}
unzip -q ${ARCHIVE} -d ${TMPDIR}

OPENCV_DIR=${TMPDIR}/opencv-4.9.0-aarch64-linux-sunxi-glibc
if [ ! -d "${OPENCV_DIR}" ]; then
    echo "ERROR: Expected dir not found: ${OPENCV_DIR}"
    exit 1
fi

echo "Source dir: ${OPENCV_DIR}"

# Копируем заголовки
mkdir -p /usr/local/include
cp -r ${OPENCV_DIR}/include/opencv4 /usr/local/include/
echo "Headers installed to /usr/local/include/opencv4/"

# Копируем библиотеки
mkdir -p /usr/local/lib
cp -d ${OPENCV_DIR}/lib/libopencv* /usr/local/lib/
echo "Libraries installed to /usr/local/lib/"

# Копируем 3rdparty зависимости OpenCV
mkdir -p /usr/local/lib/opencv4/3rdparty
cp ${OPENCV_DIR}/lib/opencv4/3rdparty/* /usr/local/lib/opencv4/3rdparty/
echo "3rdparty libs installed to /usr/local/lib/opencv4/3rdparty/"

# Копируем cmake конфиги
mkdir -p /usr/local/lib/cmake
cp -r ${OPENCV_DIR}/lib/cmake/opencv4 /usr/local/lib/cmake/
echo "CMake configs installed to /usr/local/lib/cmake/opencv4/"

# Обновляем кэш линковщика
/sbin/ldconfig 2>/dev/null || ldconfig 2>/dev/null || echo "Warning: ldconfig not found, run manually: sudo /sbin/ldconfig"

rm -rf ${TMPDIR}

echo "=== OpenCV 4.9.0 installed successfully ==="
