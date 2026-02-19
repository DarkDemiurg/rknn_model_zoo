#!/bin/bash
# Скрипт для обновления проекта с YOLOv5 на YOLOv8

echo "=== Обновление проекта на YOLOv8 ==="

# Удаляем старые файлы YOLOv5
echo "Удаление файлов YOLOv5..."
rm -f cpp/rkYolov5s.cc cpp/postprocess.cc cpp/include/rkYolov5s.hpp cpp/include/postprocess.h

# Копируем новые файлы (предполагается что архив распакован)
echo "Файлы YOLOv8 должны быть уже распакованы"

# Пересборка
echo "Пересборка проекта..."
cd cpp/build
rm -rf *
cmake -DTARGET_SOC=rk3588 ..
make -j4

echo "=== Готово! ==="
echo "Запустите: ./zmq_opi_yolo8 ./model/rk3588/yolov8n.rknn 0"
