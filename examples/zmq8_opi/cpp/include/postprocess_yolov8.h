#ifndef _RKNN_YOLOV8_DEMO_POSTPROCESS_H_
#define _RKNN_YOLOV8_DEMO_POSTPROCESS_H_

#include <stdint.h>
#include <vector>

#define OBJ_NAME_MAX_SIZE 16
#define OBJ_NUMB_MAX_SIZE 64
#define OBJ_CLASS_NUM 80
#define NMS_THRESH 0.45
#define BOX_THRESH 0.25

typedef struct _BOX_RECT
{
    int left;
    int right;
    int top;
    int bottom;
} BOX_RECT;

typedef struct __detect_result_t
{
    char name[OBJ_NAME_MAX_SIZE];
    BOX_RECT box;
    float prop;
} detect_result_t;

typedef struct _detect_result_group_t
{
    int id;
    int count;
    detect_result_t results[OBJ_NUMB_MAX_SIZE];
} detect_result_group_t;

int post_process_yolov8(int8_t *input, int model_in_h, int model_in_w,
                        float conf_threshold, float nms_threshold,
                        float scale_w, float scale_h,
                        int32_t qnt_zp, float qnt_scale,
                        detect_result_group_t *group);

int post_process_yolov8_multi(int8_t **inputs, int num_outputs, 
                               rknn_tensor_attr *output_attrs,
                               int model_in_h, int model_in_w,
                               float conf_threshold, float nms_threshold,
                               float scale_w, float scale_h,
                               detect_result_group_t *group);

void deinitPostProcess();
#endif
