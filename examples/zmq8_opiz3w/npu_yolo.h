/**
 * @file npu_yolo.h
 * @brief Обёртка Allwinner NPU для YOLOv8 / YOLOv11 (Orange Pi Zero 3W / A733).
 */
#ifndef _NPU_YOLO_H_
#define _NPU_YOLO_H_

#include "npulib.h"
#include "postprocess.h"
#include <vector>
#include <string>

class NpuYolo {
public:
    NpuYolo();
    ~NpuYolo();

    int init(const std::string &model_path, YoloModelVersion version);

    int infer(unsigned char *input_rgb, int img_w, int img_h,
              std::vector<DetectResult> &results,
              const std::vector<int> &class_filter = {});

private:
    NpuUint npu_unit_;
    NetworkItem network_;
    int output_cnt_;
    bool initialized_;
    YoloModelVersion version_;
};

#endif
