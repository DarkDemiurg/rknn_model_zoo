# zmq_dragon — Radxa Dragon Q6A + YOLO (v5/v8/v11/v26)

ZMQ-детектор для платы Radxa Dragon Q6A (Qualcomm QCS6490).
Инференс на Hexagon NPU через QAI AppBuilder.

## Структура

```
zmq_dragon/
├── Makefile
├── python/
│   ├── main.py              # основной скрипт
│   ├── benchmark_camera.py  # бенчмарк захвата + inference
│   └── clnt.py              # тестовый клиент
└── model/
    ├── yolov5/              # модель YOLOv5
    └── yolov8/
        ├── w8a8/            # быстрая квантизация (~90 FPS)
        │   └── yolov8_det.bin
        └── w8a16/           # точная квантизация (~50 FPS)
            └── yolov8_det.bin
```

## Запуск

```bash
# YOLOv8 w8a8, камера /dev/video4, 960x720
make run SOURCE=4

# YOLOv8 w8a16 (точнее, но медленнее)
make run SOURCE=4 VARIANT=w8a16

# Fisheye камера с кропом центра (рекомендуется)
make run-crop SOURCE=0

# Fisheye камера с lenscorrection
make run-fisheye SOURCE=0
```

## Работа с fisheye камерами

Fisheye объективы создают сильную бочкообразную дисторсию, которая мешает
детектированию (модель обучена на обычных кадрах). Два подхода:

### Crop центра (рекомендуется)

Обрезает края кадра, где дисторсия максимальна. Центр fisheye-изображения
практически не искажён.

```bash
make run-crop SOURCE=0 CROP=0.5          # 50% центра (по умолчанию)
make run-crop SOURCE=0 CROP=0.6          # 60% — больше угол обзора
```

### Lenscorrection (требует калибровки)

Коррекция через ffmpeg фильтр `lenscorrection`. Требует подбора коэффициентов
k1, k2 под конкретный объектив.

```bash
make run-fisheye SOURCE=0 FISHEYE="-0.2:0.0"
```

### Результаты тестирования

| Камера | Режим | FPS | Качество детекций |
|--------|-------|-----|-------------------|
| USB (обычная) | без коррекции | 94 (w8a8) | Отличное, confidence 0.7+ |
| Fisheye | без коррекции | 87 | Плохое, max score ~0.1 |
| Fisheye | crop 0.5 | 87 | Хорошее, confidence 0.3-0.5 |
| Fisheye | lenscorrection | 85 | Зависит от коэффициентов |

**Важно**: модель ожидает на входе нормализованное изображение float32 [0,1] в формате RGB.
Нормализация выполняется в capture process параллельно с inference.

## Бенчмарк

```bash
make bench                    # только захват камеры
make bench-inference          # захват + inference
```

## Производительность

| Вариант | Inference (NPU) | Итого с камерой |
|---------|-----------------|-----------------|
| w8a8    | ~4.4 ms         | ~90 FPS         |
| w8a16   | ~9 ms           | ~50 FPS         |

## ZMQ протокол

Multipart сообщение (PUB/SUB, `tcp://127.0.0.1:5757`):
- Part 1: текст детекций `"name@x1,y1,x2,y2@confidence;..."`
- Part 2: raw BGR image bytes (640x640)

## Подготовка моделей

Модели компилируются через Qualcomm AI Hub:

```bash
# YOLOv8 w8a8 (рекомендуется)
python -m qai_hub_models.models.yolov8_det.export \
  --device "Dragonwing RB3 Gen 2 Vision Kit" \
  --target-runtime qnn_context_binary \
  --quantize w8a8

# YOLOv8 w8a16 (точнее)
python -m qai_hub_models.models.yolov8_det.export \
  --device "Dragonwing RB3 Gen 2 Vision Kit" \
  --target-runtime qnn_context_binary \
  --quantize w8a16
```

Скопируйте `.bin` в `model/yolov8/w8a8/` или `model/yolov8/w8a16/`.

## Параметры main.py

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `--model` | yolov8 | Архитектура модели |
| `--variant` | w8a8 | Квантизация: w8a8 или w8a16 |
| `--source` | 0 | Индекс камеры (/dev/videoN) |
| `-w` | 960 | Ширина захвата |
| `-ht` | 720 | Высота захвата |
| `--score-thresh` | 0.5 | Порог confidence |
| `--iou-thresh` | 0.7 | Порог NMS IoU |
| `--crop` | — | Crop центра (0.0-1.0) |
| `--fisheye` | — | Коррекция fisheye (k1:k2) |
| `--zmq-addr` | tcp://127.0.0.1:5757 | Адрес ZMQ PUB |
