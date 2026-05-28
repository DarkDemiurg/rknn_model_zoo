/**
 * @file postprocess.h
 * @brief Постпроцессинг YOLOv8 для Allwinner NPU (6 выходов: 3 пары grid+score).
 */
#ifndef _POSTPROCESS_H_
#define _POSTPROCESS_H_

#include <vector>
#include <string>

// Максимальное количество детекций
#define OBJ_NUMB_MAX_SIZE 128

struct DetectResult {
    float x1, y1, x2, y2;
    float confidence;
    int class_id;
};

/**
 * Постпроцессинг 6 выходов YOLOv8 (Allwinner NPU формат).
 * Выходы: [grid_8, score_8, grid_16, score_16, grid_32, score_32]
 *
 * @param output        Массив указателей на float-выходы NPU (6 штук)
 * @param img_w         Ширина исходного изображения
 * @param img_h         Высота исходного изображения
 * @param results       Вектор результатов детекции
 * @return              Количество детекций
 */
int postprocess_yolov8_6(float **output, int img_w, int img_h,
                         std::vector<DetectResult> &results);

#endif
