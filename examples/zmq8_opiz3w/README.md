# zmq8_opiz3w — YOLOv8 / YOLOv11 + ZMQ для Orange Pi Zero 3W (Allwinner A733)

Детекция объектов с USB-камеры. Один бинарник поддерживает YOLOv8 и YOLOv11 (флаг `-v`).

## Требования

- Orange Pi Zero 3W с Ubuntu 22.04
- `libopencv-dev`, `cmake`, `libzmq3-dev`, `cppzmq`
- NPU runtime: `libVIPhal.so`, `libNBGlinker.so`
- GStreamer

## Сборка на плате

```bash
cd examples/zmq8_opiz3w
chmod +x build_native.sh
./build_native.sh
```

Бинарник: `install/zmq8_opiz3w_linux_a733/zmq8_opiz3w`

## Запуск

```bash
# YOLOv8 (по умолчанию)
./install/zmq8_opiz3w_linux_a733/zmq8_opiz3w ./model/yolov8n_6_uint8_a733.nb 0

# YOLOv11
./install/zmq8_opiz3w_linux_a733/zmq8_opiz3w -v 11 ./model/yolo11n_6_uint8_a733.nb 0

# С параметрами камеры и фильтром классов
./install/zmq8_opiz3w_linux_a733/zmq8_opiz3w ./model/yolov8n_6_uint8_a733.nb 0 -w 960 -h 720 -c person
```

## ZMQ

- Адрес: `tcp://127.0.0.1:5757`
- Формат: multipart `[детекции;][RGB 640x640x3]`

## Модели

| Файл | Описание |
|------|----------|
| `model/yolov8n_6_uint8_a733.nb` | YOLOv8n, включена |
| `model/yolo11n_6_uint8_a733.nb` | YOLOv11n, нужно сконвертировать |

### Fisheye-камера

По опыту `zmq_dragon` для fisheye лучше всего **crop центра**:

```bash
# Рекомендуется: обрезать центральные 50% (меньше дисторсии по краям)
./zmq8_opiz3w ./model/yolov8n_6_uint8_a733.nb 0 -r 0.5 -d 100

# Дополнительно lens correction (подбор k1:k2)
./zmq8_opiz3w ./model/yolov8n_6_uint8_a733.nb 0 -r 0.5 -f -0.2:0.0
```

| Опция | Описание |
|-------|----------|
| `-r 0.5` | Crop центральных 50% кадра перед детекцией |
| `-f -0.2:0.0` | OpenCV fisheye undistort (k1:k2) |

Класс чашки в COCO: `cup` (id 41).

### Конвертация YOLOv11

Используйте `awnpu_model_zoo` или ai-sdk Pegasus (Docker):

```bash
# В awnpu_model_zoo/examples/yolo11/convert_model/
# после конвертации скопируйте .nb в model/yolo11n_6_uint8_a733.nb
```

Скрипты конвертации: `cubie/awnpu_model_zoo-v1.0.0-.../examples/yolo11/convert_model/`

## Архитектура

3-поточный конвейер: Capture → G2D letterbox → NPU (v8 или v11) → ZMQ.

Постпроцессинг:
- YOLOv8: `postprocess_v8.cpp` (6 выходов, HWC)
- YOLOv11: `postprocess_yolo11.cpp` (6 выходов, CHW→transpose)
