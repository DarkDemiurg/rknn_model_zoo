/**
 * @file npu_yolov8.cpp
 * @brief Реализация обёртки NPU для YOLOv8.
 *        Использует Allwinner VIPLite/NBGLinker runtime через npulib.
 */
#include "npu_yolov8.h"
#include "model_config.h"

#include <stdio.h>
#include <string.h>

NpuYolov8::NpuYolov8() : output_cnt_(0), initialized_(false) {}

NpuYolov8::~NpuYolov8() {}

int NpuYolov8::init(const std::string &model_path)
{
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
    fprintf(stderr, "[NPU] Model loaded: %s, outputs: %d\n", model_path.c_str(), output_cnt_);

    initialized_ = true;
    return 0;
}

int NpuYolov8::infer(unsigned char *input_rgb, int img_w, int img_h,
                     std::vector<DetectResult> &results)
{
    if (!initialized_) return -1;

    // Получаем указатель на входной буфер NPU и копируем данные
    void *input_ptr = nullptr;
    unsigned int input_size = 0;
    network_.get_network_input_buff_info(0, &input_ptr, &input_size);

    unsigned int data_size = LETTERBOX_ROWS * LETTERBOX_COLS * 3 * sizeof(unsigned char);
    if (data_size > input_size) {
        fprintf(stderr, "[NPU] Input data size %u > buffer size %u\n", data_size, input_size);
        return -1;
    }
    memcpy(input_ptr, input_rgb, data_size);

    // Настраиваем I/O и запускаем inference
    int ret = network_.network_input_output_set();
    if (ret != 0) return -1;

    ret = network_.network_run();
    if (ret != 0) return -1;

    // Получаем выходные данные (zero-copy)
    output_info_s outputs_info[output_cnt_];
    network_.get_output_nocopy(outputs_info);

    float *output_data[output_cnt_];
    for (int i = 0; i < output_cnt_; i++) {
        output_data[i] = (float *)outputs_info[i].ptr;
    }

    // Постпроцессинг
    postprocess_yolov8_6(output_data, img_w, img_h, results);

    return 0;
}
