/**
 * @file npu_yolo.cpp
 */
#include "npu_yolo.h"
#include "model_config.h"

#include <stdio.h>
#include <string.h>

NpuYolo::NpuYolo() : output_cnt_(0), initialized_(false), version_(YOLO_V8) {}

NpuYolo::~NpuYolo() {}

int NpuYolo::init(const std::string &model_path, YoloModelVersion version)
{
    version_ = version;

    int ret = npu_unit_.npu_init();
    if (ret != 0) {
        fprintf(stderr, "[NPU] npu_init failed\n");
        return -1;
    }

    unsigned int network_id = 0;
    ret = network_.network_create((char *)model_path.c_str(), network_id);
    if (ret != 0) {
        fprintf(stderr, "[NPU] network_create failed\n");
        return -1;
    }

    ret = network_.network_prepare();
    if (ret != 0) {
        fprintf(stderr, "[NPU] network_prepare failed\n");
        return -1;
    }

    output_cnt_ = network_.get_output_cnt();
    fprintf(stderr, "[NPU] YOLOv%d model loaded: %s, outputs: %d\n",
            (int)version_, model_path.c_str(), output_cnt_);

    if (output_cnt_ != 6) {
        fprintf(stderr, "[NPU] Warning: expected 6 outputs for YOLOv8/v11, got %d\n", output_cnt_);
    }

    initialized_ = true;
    return 0;
}

int NpuYolo::infer(unsigned char *input_rgb, int img_w, int img_h,
                   std::vector<DetectResult> &results,
                   const std::vector<int> &class_filter)
{
    if (!initialized_) return -1;

    void *input_ptr = nullptr;
    unsigned int input_size = 0;
    network_.get_network_input_buff_info(0, &input_ptr, &input_size);

    unsigned int data_size = LETTERBOX_ROWS * LETTERBOX_COLS * 3 * sizeof(unsigned char);
    if (data_size > input_size) {
        fprintf(stderr, "[NPU] Input data size %u > buffer size %u\n", data_size, input_size);
        return -1;
    }
    memcpy(input_ptr, input_rgb, data_size);

    int ret = network_.network_input_output_set();
    if (ret != 0) return -1;

    ret = network_.network_run();
    if (ret != 0) return -1;

    output_info_s outputs_info[output_cnt_];
    network_.get_output_nocopy(outputs_info);

    float *output_data[output_cnt_];
    for (int i = 0; i < output_cnt_; i++) {
        output_data[i] = (float *)outputs_info[i].ptr;
    }

    if (version_ == YOLO_V11)
        postprocess_yolo11_6(output_data, img_w, img_h, results, class_filter);
    else
        postprocess_yolov8_6(output_data, img_w, img_h, results, class_filter);

    return 0;
}
