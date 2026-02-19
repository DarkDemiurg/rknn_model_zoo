#include "postprocess_yolov8.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <vector>

#define LABEL_NALE_TXT_PATH "./model/coco_80_labels_list.txt"

static char *labels[OBJ_CLASS_NUM];

inline static int clamp(float val, int min, int max) { 
    return val > min ? (val < max ? val : max) : min; 
}

static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) { 
    return ((float)qnt - (float)zp) * scale; 
}

char *readLine(FILE *fp, char *buffer, int *len)
{
    int ch;
    int i = 0;
    size_t buff_len = 0;
    buffer = (char *)malloc(buff_len + 1);
    if (!buffer) return NULL;
    
    while ((ch = fgetc(fp)) != '\n' && ch != EOF)
    {
        buff_len++;
        void *tmp = realloc(buffer, buff_len + 1);
        if (tmp == NULL) {
            free(buffer);
            return NULL;
        }
        buffer = (char *)tmp;
        buffer[i] = (char)ch;
        i++;
    }
    buffer[i] = '\0';
    *len = buff_len;
    
    if (ch == EOF && (i == 0 || ferror(fp))) {
        free(buffer);
        return NULL;
    }
    return buffer;
}

int readLines(const char *fileName, char *lines[], int max_line)
{
    FILE *file = fopen(fileName, "r");
    char *s;
    int i = 0;
    int n = 0;
    
    if (file == NULL) {
        printf("Open %s fail!\n", fileName);
        return -1;
    }
    
    while ((s = readLine(file, s, &n)) != NULL) {
        lines[i++] = s;
        if (i >= max_line) break;
    }
    fclose(file);
    return i;
}

int loadLabelName(const char *locationFilename, char *label[])
{
    printf("loadLabelName %s\n", locationFilename);
    readLines(locationFilename, label, OBJ_CLASS_NUM);
    return 0;
}

static float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, 
                              float xmin1, float ymin1, float xmax1, float ymax1)
{
    float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1));
    float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1));
    float i = w * h;
    float u = (xmax0 - xmin0) * (ymax0 - ymin0) + (xmax1 - xmin1) * (ymax1 - ymin1) - i;
    return u <= 0.f ? 0.f : (i / u);
}

int post_process_yolov8(int8_t *input, int model_in_h, int model_in_w,
                        float conf_threshold, float nms_threshold,
                        float scale_w, float scale_h,
                        int32_t qnt_zp, float qnt_scale,
                        detect_result_group_t *group)
{
    static int init = -1;
    if (init == -1) {
        if (loadLabelName(LABEL_NALE_TXT_PATH, labels) < 0) {
            return -1;
        }
        init = 0;
    }
    
    memset(group, 0, sizeof(detect_result_group_t));
    
    // YOLOv8 output: [1, 84, 8400] -> 84 = 4 bbox + 80 classes
    const int num_classes = OBJ_CLASS_NUM;
    const int num_boxes = 8400;
    const int box_dim = 4;
    
    std::vector<float> boxes;
    std::vector<float> scores;
    std::vector<int> class_ids;
    
    // Parse detections
    for (int i = 0; i < num_boxes; i++) {
        // Find max class score
        float max_score = -1.0f;
        int max_class_id = -1;
        
        for (int c = 0; c < num_classes; c++) {
            int idx = (box_dim + c) * num_boxes + i;
            float score = deqnt_affine_to_f32(input[idx], qnt_zp, qnt_scale);
            if (score > max_score) {
                max_score = score;
                max_class_id = c;
            }
        }
        
        if (max_score < conf_threshold) continue;
        
        // Get bbox coordinates (x_center, y_center, width, height)
        float x = deqnt_affine_to_f32(input[i], qnt_zp, qnt_scale);
        float y = deqnt_affine_to_f32(input[num_boxes + i], qnt_zp, qnt_scale);
        float w = deqnt_affine_to_f32(input[2 * num_boxes + i], qnt_zp, qnt_scale);
        float h = deqnt_affine_to_f32(input[3 * num_boxes + i], qnt_zp, qnt_scale);
        
        boxes.push_back(x - w / 2);
        boxes.push_back(y - h / 2);
        boxes.push_back(x + w / 2);
        boxes.push_back(y + h / 2);
        scores.push_back(max_score);
        class_ids.push_back(max_class_id);
    }
    
    // NMS
    std::vector<int> indices(scores.size());
    for (size_t i = 0; i < indices.size(); i++) indices[i] = i;
    
    std::sort(indices.begin(), indices.end(), [&scores](int a, int b) {
        return scores[a] > scores[b];
    });
    
    std::vector<bool> keep(scores.size(), true);
    for (size_t i = 0; i < indices.size(); i++) {
        if (!keep[indices[i]]) continue;
        
        for (size_t j = i + 1; j < indices.size(); j++) {
            if (!keep[indices[j]]) continue;
            if (class_ids[indices[i]] != class_ids[indices[j]]) continue;
            
            float iou = CalculateOverlap(
                boxes[indices[i] * 4], boxes[indices[i] * 4 + 1],
                boxes[indices[i] * 4 + 2], boxes[indices[i] * 4 + 3],
                boxes[indices[j] * 4], boxes[indices[j] * 4 + 1],
                boxes[indices[j] * 4 + 2], boxes[indices[j] * 4 + 3]
            );
            
            if (iou > nms_threshold) keep[indices[j]] = false;
        }
    }
    
    // Fill results
    int count = 0;
    for (size_t i = 0; i < keep.size() && count < OBJ_NUMB_MAX_SIZE; i++) {
        if (!keep[i]) continue;
        
        group->results[count].box.left = (int)(clamp(boxes[i * 4] / scale_w, 0, model_in_w));
        group->results[count].box.top = (int)(clamp(boxes[i * 4 + 1] / scale_h, 0, model_in_h));
        group->results[count].box.right = (int)(clamp(boxes[i * 4 + 2] / scale_w, 0, model_in_w));
        group->results[count].box.bottom = (int)(clamp(boxes[i * 4 + 3] / scale_h, 0, model_in_h));
        group->results[count].prop = scores[i];
        strncpy(group->results[count].name, labels[class_ids[i]], OBJ_NAME_MAX_SIZE);
        count++;
    }
    
    group->count = count;
    return 0;
}

void deinitPostProcess()
{
    for (int i = 0; i < OBJ_CLASS_NUM; i++) {
        if (labels[i] != nullptr) {
            free(labels[i]);
            labels[i] = nullptr;
        }
    }
}

// For YOLOv8 with multiple outputs (split heads)
int post_process_yolov8_multi(int8_t **inputs, int num_outputs,
                               rknn_tensor_attr *output_attrs,
                               int model_in_h, int model_in_w,
                               float conf_threshold, float nms_threshold,
                               float scale_w, float scale_h,
                               detect_result_group_t *group)
{
    static int init = -1;
    if (init == -1) {
        if (loadLabelName(LABEL_NALE_TXT_PATH, labels) < 0) {
            return -1;
        }
        init = 0;
    }
    
    memset(group, 0, sizeof(detect_result_group_t));
    
    const int num_classes = OBJ_CLASS_NUM;
    std::vector<float> boxes;
    std::vector<float> scores;
    std::vector<int> class_ids;
    
    // YOLOv8 split outputs: bbox (64ch), classes (80ch), objectness (1ch) for each scale
    // Process 3 scales: 80x80, 40x40, 20x20
    for (int scale_idx = 0; scale_idx < 3; scale_idx++) {
        int bbox_idx = scale_idx * 3;      // 0, 3, 6
        int class_idx = scale_idx * 3 + 1; // 1, 4, 7
        int obj_idx = scale_idx * 3 + 2;   // 2, 5, 8
        
        if (bbox_idx >= num_outputs || class_idx >= num_outputs || obj_idx >= num_outputs) break;
        
        int8_t *bbox_output = inputs[bbox_idx];
        int8_t *class_output = inputs[class_idx];
        int8_t *obj_output = inputs[obj_idx];
        
        int32_t bbox_zp = output_attrs[bbox_idx].zp;
        float bbox_scale = output_attrs[bbox_idx].scale;
        int32_t class_zp = output_attrs[class_idx].zp;
        float class_scale = output_attrs[class_idx].scale;
        int32_t obj_zp = output_attrs[obj_idx].zp;
        float obj_scale = output_attrs[obj_idx].scale;
        
        int grid_h = output_attrs[class_idx].dims[2];
        int grid_w = output_attrs[class_idx].dims[3];
        int bbox_ch = output_attrs[bbox_idx].dims[1];
        
        printf("DEBUG: Scale %d - grid %dx%d, bbox_ch=%d\n", scale_idx, grid_h, grid_w, bbox_ch);
        
        int stride = model_in_h / grid_h;
        int detections_found = 0;
        
        for (int h = 0; h < grid_h; h++) {
            for (int w = 0; w < grid_w; w++) {
                int grid_idx = h * grid_w + w;
                
                // Get objectness
                float objectness = deqnt_affine_to_f32(obj_output[grid_idx], obj_zp, obj_scale);
                
                if (objectness < conf_threshold) continue;
                
                // Find max class score
                float max_class_score = -1.0f;
                int max_class_id = -1;
                
                for (int c = 0; c < num_classes; c++) {
                    int class_offset = c * grid_h * grid_w + grid_idx;
                    float score = deqnt_affine_to_f32(class_output[class_offset], class_zp, class_scale);
                    if (score > max_class_score) {
                        max_class_score = score;
                        max_class_id = c;
                    }
                }
                
                float final_score = objectness * max_class_score;
                if (final_score < conf_threshold) continue;
                
                detections_found++;
                
                // Decode bbox (DFL format: 64 channels = 4 directions × 16 bins)
                float bbox[4] = {0};
                int reg_max = bbox_ch / 4; // 16
                
                for (int k = 0; k < 4; k++) {
                    float sum = 0;
                    float max_val = -1e9;
                    
                    // Softmax over reg_max bins
                    std::vector<float> vals(reg_max);
                    for (int i = 0; i < reg_max; i++) {
                        int offset = (k * reg_max + i) * grid_h * grid_w + grid_idx;
                        vals[i] = deqnt_affine_to_f32(bbox_output[offset], bbox_zp, bbox_scale);
                        if (vals[i] > max_val) max_val = vals[i];
                    }
                    
                    float exp_sum = 0;
                    for (int i = 0; i < reg_max; i++) {
                        vals[i] = expf(vals[i] - max_val);
                        exp_sum += vals[i];
                    }
                    
                    for (int i = 0; i < reg_max; i++) {
                        sum += (vals[i] / exp_sum) * i;
                    }
                    
                    bbox[k] = sum;
                }
                
                // Convert to x1,y1,x2,y2
                float cx = (w + 0.5f) * stride;
                float cy = (h + 0.5f) * stride;
                
                float x1 = cx - bbox[0] * stride;
                float y1 = cy - bbox[1] * stride;
                float x2 = cx + bbox[2] * stride;
                float y2 = cy + bbox[3] * stride;
                
                if (detections_found <= 3) {
                    printf("DEBUG: Det %d: grid(%d,%d) obj=%.3f cls=%.3f final=%.3f bbox=[%.1f,%.1f,%.1f,%.1f]\n",
                           detections_found, w, h, objectness, max_class_score, final_score, x1, y1, x2, y2);
                }
                
                boxes.push_back(x1);
                boxes.push_back(y1);
                boxes.push_back(x2);
                boxes.push_back(y2);
                scores.push_back(final_score);
                class_ids.push_back(max_class_id);
            }
        }
        
        printf("DEBUG: Scale %d found %d detections\n", scale_idx, detections_found);
    }
    
    printf("DEBUG: Total detections before NMS: %zu\n", boxes.size() / 4);
    
    if (boxes.empty()) {
        group->count = 0;
        return 0;
    }
    
    // NMS
    std::vector<int> indices(scores.size());
    for (size_t i = 0; i < indices.size(); i++) indices[i] = i;
    
    std::sort(indices.begin(), indices.end(), [&scores](int a, int b) {
        return scores[a] > scores[b];
    });
    
    std::vector<bool> keep(scores.size(), true);
    for (size_t i = 0; i < indices.size(); i++) {
        if (!keep[indices[i]]) continue;
        
        for (size_t j = i + 1; j < indices.size(); j++) {
            if (!keep[indices[j]]) continue;
            if (class_ids[indices[i]] != class_ids[indices[j]]) continue;
            
            float iou = CalculateOverlap(
                boxes[indices[i] * 4], boxes[indices[i] * 4 + 1],
                boxes[indices[i] * 4 + 2], boxes[indices[i] * 4 + 3],
                boxes[indices[j] * 4], boxes[indices[j] * 4 + 1],
                boxes[indices[j] * 4 + 2], boxes[indices[j] * 4 + 3]
            );
            
            if (iou > nms_threshold) keep[indices[j]] = false;
        }
    }
    
    // Fill results
    int count = 0;
    for (size_t i = 0; i < keep.size() && count < OBJ_NUMB_MAX_SIZE; i++) {
        if (!keep[i]) continue;
        
        group->results[count].box.left = (int)(clamp(boxes[i * 4] / scale_w, 0, model_in_w));
        group->results[count].box.top = (int)(clamp(boxes[i * 4 + 1] / scale_h, 0, model_in_h));
        group->results[count].box.right = (int)(clamp(boxes[i * 4 + 2] / scale_w, 0, model_in_w));
        group->results[count].box.bottom = (int)(clamp(boxes[i * 4 + 3] / scale_h, 0, model_in_h));
        group->results[count].prop = scores[i];
        strncpy(group->results[count].name, labels[class_ids[i]], OBJ_NAME_MAX_SIZE);
        count++;
    }
    
    group->count = count;
    printf("DEBUG: Final detections after NMS: %d\n", count);
    return 0;
}
