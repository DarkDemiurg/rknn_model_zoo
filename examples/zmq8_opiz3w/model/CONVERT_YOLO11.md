# Конвертация YOLOv11 для Orange Pi Zero 3W (A733)

Модель `yolo11n_6_uint8_a733.nb` нужно получить через Pegasus (Allwinner NPU toolchain).

## Вариант 1: awnpu_model_zoo

```bash
cd cubie/awnpu_model_zoo-v1.0.0-20260423-f562dd16/examples/yolo11/convert_model/
# Следуйте README в awnpu_model_zoo/examples/yolo11/
```

## Вариант 2: ai-sdk Docker (User Manual §3.35)

```bash
# В контейнере ubuntu-npu:v2.0.10
cd /workspace/ai-sdk/models
source env.sh v3
# Импорт ONNX → квантизация → export NBG
```

Сконвертированный `.nb` положите в:

```
examples/zmq8_opiz3w/model/yolo11n_6_uint8_a733.nb
```

Запуск:

```bash
./zmq8_opiz3w -v 11 ./model/yolo11n_6_uint8_a733.nb 0
```
