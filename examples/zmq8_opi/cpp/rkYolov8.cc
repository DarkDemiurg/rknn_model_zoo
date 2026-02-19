#include <stdio.h>
#include <string>
#include <mutex>
#include "rknn_api.h"
#include "postprocess_yolov8.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "coreNum.hpp"
#include "rkYolov8.hpp"

static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz)
{
    unsigned char *data = NULL;
    if (NULL == fp) return NULL;
    
    if (fseek(fp, ofst, SEEK_SET) != 0) {
        printf("blob seek failure.\n");
        return NULL;
    }
    
    data = (unsigned char *)malloc(sz);
    if (data == NULL) {
        printf("buffer malloc failure.\n");
        return NULL;
    }
    fread(data, 1, sz, fp);
    return data;
}

static unsigned char *load_model(const char *filename, int *model_size)
{
    FILE *fp = fopen(filename, "rb");
    if (NULL == fp) {
        printf("Open file %s failed.\n", filename);
        return NULL;
    }
    
    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);
    unsigned char *data = load_data(fp, 0, size);
    fclose(fp);
    
    *model_size = size;
    return data;
}

rkYolov8::rkYolov8(const std::string &model_path)
{
    this->model_path = model_path;
    nms_threshold = NMS_THRESH;
    box_conf_threshold = BOX_THRESH;
}

int rkYolov8::init(rknn_context *ctx_in, bool share_weight)
{
    printf("Loading model...\n");
    int model_data_size = 0;
    model_data = load_model(model_path.c_str(), &model_data_size);
    
    if (share_weight == true)
        ret = rknn_dup_context(ctx_in, &ctx);
    else
        ret = rknn_init(&ctx, model_data, model_data_size, 0, NULL);

    if (ret < 0) {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }

    rknn_core_mask core_mask;
    switch (get_core_num()) {
        case 0: core_mask = RKNN_NPU_CORE_0; break;
        case 1: core_mask = RKNN_NPU_CORE_1; break;
        case 2: core_mask = RKNN_NPU_CORE_2; break;
    }

    ret = rknn_set_core_mask(ctx, core_mask);
    if (ret < 0) {
        printf("rknn_init core error ret=%d\n", ret);
        return -1;
    }

    rknn_sdk_version version;
    ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
    if (ret < 0) {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }
    printf("sdk version: %s driver version: %s\n", version.api_version, version.drv_version);

    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0) {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    input_attrs = (rknn_tensor_attr *)calloc(io_num.n_input, sizeof(rknn_tensor_attr));
    for (int i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0) {
            printf("rknn_init error ret=%d\n", ret);
            return -1;
        }
    }

    output_attrs = (rknn_tensor_attr *)calloc(io_num.n_output, sizeof(rknn_tensor_attr));
    for (int i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
    }

    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
        printf("model is NCHW input fmt\n");
        channel = input_attrs[0].dims[1];
        height = input_attrs[0].dims[2];
        width = input_attrs[0].dims[3];
    } else {
        printf("model is NHWC input fmt\n");
        height = input_attrs[0].dims[1];
        width = input_attrs[0].dims[2];
        channel = input_attrs[0].dims[3];
    }
    printf("model input height=%d, width=%d, channel=%d\n", height, width, channel);

    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = width * height * channel;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].pass_through = 0;

    return 0;
}

rknn_context *rkYolov8::get_pctx()
{
    return &ctx;
}

Res rkYolov8::infer(Res &orig_img)
{
    std::lock_guard<std::mutex> lock(mtx);
    cv::Mat img;
    cv::cvtColor(orig_img.img, img, cv::COLOR_BGR2RGB);
    img_width = img.cols;
    img_height = img.rows;

    cv::Mat resized_img(height, width, CV_8UC3);
    float scale_w = (float)width / img.cols;
    float scale_h = (float)height / img.rows;

    if (img_width != width || img_height != height) {
        cv::resize(img, resized_img, cv::Size(width, height));
        inputs[0].buf = (void*)resized_img.data;
    } else {
        inputs[0].buf = (void*)img.data;
    }

    rknn_inputs_set(ctx, io_num.n_input, inputs);

    rknn_output outputs[io_num.n_output];
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < io_num.n_output; i++) {
        outputs[i].want_float = 0;
    }

    ret = rknn_run(ctx, NULL);
    ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);

    detect_result_group_t detect_result_group;
    post_process_yolov8((int8_t *)outputs[0].buf, height, width,
                        box_conf_threshold, nms_threshold,
                        scale_w, scale_h,
                        output_attrs[0].zp, output_attrs[0].scale,
                        &detect_result_group);

    std::string msg;
    char text[256];
    
    for (int i = 0; i < detect_result_group.count; i++) {
        detect_result_t *det_result = &(detect_result_group.results[i]);

        int x1 = det_result->box.left;
        int y1 = det_result->box.top;
        int x2 = det_result->box.right;
        int y2 = det_result->box.bottom;
        
        cv::rectangle(orig_img.img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0));

        sprintf(text, "%s@%d,%d,%d,%d@%.2f;", det_result->name, x1, y1, x2, y2, det_result->prop);
        msg += text;

        sprintf(text, "%s %.1f%%", det_result->name, det_result->prop * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = x1;
        int y = y1 - label_size.height - baseLine;
        if (y < 0) y = 0;
        if (x + label_size.width > orig_img.img.cols) x = orig_img.img.cols - label_size.width;

        cv::rectangle(orig_img.img, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)), 
                     cv::Scalar(255, 255, 255), -1);
        cv::putText(orig_img.img, text, cv::Point(x, y + label_size.height), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }

    ret = rknn_outputs_release(ctx, io_num.n_output, outputs);

    Res r;
    r.img = orig_img.img;
    r.msg = msg;

    return r;
}

rkYolov8::~rkYolov8()
{
    deinitPostProcess();
    ret = rknn_destroy(ctx);
    if (model_data) free(model_data);
    if (input_attrs) free(input_attrs);
    if (output_attrs) free(output_attrs);
}
