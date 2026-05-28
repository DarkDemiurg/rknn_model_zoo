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

# Определяем корневую папку внутри архива
OPENCV_DIR=$(find ${TMPDIR} -maxdepth 1 -type d -name "opencv*" | head -1)
if [ -z "${OPENCV_DIR}" ]; then
    OPENCV_DIR=${TMPDIR}
fi

echo "Source dir: ${OPENCV_DIR}"

# Копируем заголовки
if [ -d "${OPENCV_DIR}/include" ]; then
    cp -r ${OPENCV_DIR}/include/opencv4 /usr/local/include/ 2>/dev/null || \
    cp -r ${OPENCV_DIR}/include/* /usr/local/include/
    echo "Headers installed to /usr/local/include/"
fi

# Копируем библиотеки
if [ -d "${OPENCV_DIR}/lib" ]; then
    cp -d ${OPENCV_DIR}/lib/libopencv* /usr/local/lib/
    echo "Libraries installed to /usr/local/lib/"
fi

# Копируем cmake конфиги
if [ -d "${OPENCV_DIR}/lib/cmake" ]; then
    cp -r ${OPENCV_DIR}/lib/cmake/opencv4 /usr/local/lib/cmake/ 2>/dev/null || \
    mkdir -p /usr/local/lib/cmake && cp -r ${OPENCV_DIR}/lib/cmake/opencv4 /usr/local/lib/cmake/
    echo "CMake configs installed to /usr/local/lib/cmake/opencv4/"
fi

# Обновляем кэш линковщика
ldconfig

rm -rf ${TMPDIR}

echo "=== OpenCV 4.9.0 installed successfully ==="
echo "Verify: pkg-config --modversion opencv4"
