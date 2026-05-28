/**
 * @file npu_yolov8.h
 * @brief Обёртка над Allwinner NPU runtime для YOLOv8 inference.
 *        Инкапсулирует инициализацию, загрузку модели и запуск inference.
 */
#ifndef _NPU_YOLOV8_H_
#define _NPU_YOLOV8_H_

#include "npulib.h"
#include "postprocess.h"
#include <vector>
#include <string>

class NpuYolov8 {
public:
    NpuYolov8();
    ~NpuYolov8();

    /**
     * Инициализация NPU и загрузка модели.
     * @param model_path Путь к .nb файлу модели
     * @return 0 при успехе
     */
    int init(const std::string &model_path);

    /**
     * Запуск inference на подготовленных данных.
     * @param input_rgb  Указатель на RGB данные (640x640x3, uint8)
     * @param img_w      Ширина оригинального изображения (для обратного маппинга координат)
     * @param img_h      Высота оригинального изображения
     * @param results    Вектор результатов детекции
     * @return 0 при успехе
     */
    int infer(unsigned char *input_rgb, int img_w, int img_h,
              std::vector<DetectResult> &results);

private:
    NpuUint npu_unit_;
    NetworkItem network_;
    int output_cnt_;
    bool initialized_;
};

#endif
