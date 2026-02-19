#ifndef _RKYOLOV8_H_
#define _RKYOLOV8_H_

#include "rknn_api.h"
#include "opencv2/core/core.hpp"
#include <string>
#include <mutex>

struct Res {
    cv::Mat img;
    std::string msg;
};

class rkYolov8 {
public:
    rkYolov8(const std::string &model_path);
    ~rkYolov8();
    
    int init(rknn_context *ctx_in = nullptr, bool share_weight = false);
    Res infer(Res &orig_img);
    rknn_context *get_pctx();

private:
    std::string model_path;
    rknn_context ctx;
    rknn_input_output_num io_num;
    rknn_tensor_attr *input_attrs = nullptr;
    rknn_tensor_attr *output_attrs = nullptr;
    rknn_input inputs[1];
    unsigned char *model_data = nullptr;
    
    int ret;
    int channel;
    int width;
    int height;
    int img_width;
    int img_height;
    
    float nms_threshold;
    float box_conf_threshold;
    
    std::mutex mtx;
};

#endif
