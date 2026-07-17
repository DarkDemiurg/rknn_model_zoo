/**
 * @file postprocess.h
 * @brief Постпроцессинг YOLOv8 / YOLOv11 для Allwinner A733 NPU (6 выходов).
 */
#ifndef _POSTPROCESS_H_
#define _POSTPROCESS_H_

#include <vector>

#define OBJ_NUMB_MAX_SIZE 128

enum YoloModelVersion {
    YOLO_V8 = 8,
    YOLO_V11 = 11
};

struct DetectResult {
    float x1, y1, x2, y2;
    float confidence;
    int class_id;
};

int postprocess_yolov8_6(float **output, int img_w, int img_h,
                         std::vector<DetectResult> &results,
                         const std::vector<int> &class_filter = {});

int postprocess_yolo11_6(float **output, int img_w, int img_h,
                         std::vector<DetectResult> &results,
                         const std::vector<int> &class_filter = {});

#endif
