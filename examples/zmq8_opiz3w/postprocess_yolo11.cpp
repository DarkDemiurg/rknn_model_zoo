/**
 * @file postprocess_yolo11.cpp
 * @brief Постпроцессинг YOLOv11 (6 выходов NPU) для Allwinner A733.
 */
#include "postprocess.h"
#include "model_config.h"

#include <opencv2/core/core.hpp>
#include <cmath>
#include <algorithm>
#include <cfloat>
#include <vector>

struct Object {
    cv::Rect_<float> rect;
    int label;
    float prob;
};

static inline float intersection_area(const Object &a, const Object &b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object> &objects, int left, int right)
{
    int i = left;
    int j = right;
    float p = objects[(left + right) / 2].prob;

    while (i <= j) {
        while (objects[i].prob > p) i++;
        while (objects[j].prob < p) j--;
        if (i <= j) {
            std::swap(objects[i], objects[j]);
            i++;
            j--;
        }
    }

    if (left < j) qsort_descent_inplace(objects, left, j);
    if (i < right) qsort_descent_inplace(objects, i, right);
}

static void qsort_descent_inplace(std::vector<Object> &objects)
{
    if (!objects.empty())
        qsort_descent_inplace(objects, 0, (int)objects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object> &objects, std::vector<int> &picked,
                              float nms_threshold)
{
    picked.clear();
    const int n = (int)objects.size();
    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
        areas[i] = objects[i].rect.area();

    for (int i = 0; i < n; i++) {
        const Object &a = objects[i];
        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++) {
            const Object &b = objects[picked[j]];
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }
        if (keep)
            picked.push_back(i);
    }
}

static inline float sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

static inline float desigmoid(float x)
{
    return -logf(1.0f / x - 1.0f);
}

static bool class_allowed(int class_id, const std::vector<int> &class_filter)
{
    if (class_filter.empty()) return true;
    for (int c : class_filter) {
        if (c == class_id) return true;
    }
    return false;
}

static float softmax16(const float *src, float *dst)
{
    float alpha = -FLT_MAX;
    for (int c = 0; c < 16; c++) {
        if (src[c] > alpha) alpha = src[c];
    }

    float denominator = 0;
    float dis_sum = 0;
    for (int i = 0; i < 16; ++i) {
        dst[i] = expf(src[i] - alpha);
        denominator += dst[i];
    }
    for (int i = 0; i < 16; ++i) {
        dst[i] /= denominator;
        dis_sum += i * dst[i];
    }
    return dis_sum;
}

static void generate_proposals_6(int stride, const float *feat_grid, const float *feat_score,
                                 float prob_threshold, std::vector<Object> &objects,
                                 int letterbox_cols, int letterbox_rows,
                                 const std::vector<int> &class_filter)
{
    const int num_grid_x = letterbox_cols / stride;
    const int num_grid_y = letterbox_rows / stride;
    const int num_grid_size = num_grid_x * num_grid_y;
    const int reg_max_1 = 16;
    const int num_class = CLASS_NUM;
    float dst[16];

    float deprob_threshold = desigmoid(prob_threshold);

    cv::Mat out_score = cv::Mat(num_class, num_grid_size, CV_32FC1, (float *)feat_score);
    cv::transpose(out_score, out_score);

    for (int y = 0; y < num_grid_y; y++) {
        for (int x = 0; x < num_grid_x; x++) {
            int num_grid_idx = y * num_grid_x + x;

            int label = -1;
            float score = -FLT_MAX;
            const float *pred_score = (float *)out_score.data + num_grid_idx * num_class;

            for (int k = 0; k < num_class; k++) {
                float s = *(pred_score + k);
                if (s > score) {
                    label = k;
                    score = s;
                }
            }

            if (score < deprob_threshold)
                continue;
            if (!class_allowed(label, class_filter))
                continue;

            score = sigmoid(score);
            if (score < prob_threshold)
                continue;

            const float *cur_pred_grid = feat_grid + num_grid_idx;
            float pred_grid[reg_max_1 * 4] = {0.0f};
            for (int i = 0; i < reg_max_1 * 4; i++) {
                pred_grid[i] = *(cur_pred_grid + i * num_grid_size);
            }

            float x0 = x + 0.5f - softmax16(pred_grid, dst);
            float y0 = y + 0.5f - softmax16(pred_grid + 16, dst);
            float x1 = x + 0.5f + softmax16(pred_grid + 2 * 16, dst);
            float y1 = y + 0.5f + softmax16(pred_grid + 3 * 16, dst);

            x0 *= stride;
            y0 *= stride;
            x1 *= stride;
            y1 *= stride;

            Object obj;
            obj.rect.x = x0;
            obj.rect.y = y0;
            obj.rect.width = x1 - x0;
            obj.rect.height = y1 - y0;
            obj.label = label;
            obj.prob = score;
            objects.push_back(obj);
        }
    }
}

int postprocess_yolo11_6(float **output, int img_w, int img_h,
                         std::vector<DetectResult> &results,
                         const std::vector<int> &class_filter)
{
    results.clear();

    std::vector<Object> proposals;
    std::vector<Object> objects8, objects16, objects32;

    generate_proposals_6(8, output[0], output[1], SCORE_THRESHOLD, objects8,
                         LETTERBOX_COLS, LETTERBOX_ROWS, class_filter);
    proposals.insert(proposals.end(), objects8.begin(), objects8.end());

    generate_proposals_6(16, output[2], output[3], SCORE_THRESHOLD, objects16,
                         LETTERBOX_COLS, LETTERBOX_ROWS, class_filter);
    proposals.insert(proposals.end(), objects16.begin(), objects16.end());

    generate_proposals_6(32, output[4], output[5], SCORE_THRESHOLD, objects32,
                         LETTERBOX_COLS, LETTERBOX_ROWS, class_filter);
    proposals.insert(proposals.end(), objects32.begin(), objects32.end());

    const LetterboxLayout lb = letterbox_layout(img_w, img_h);
    proposals.erase(
        std::remove_if(proposals.begin(), proposals.end(),
                       [&lb](const Object &o) {
                           return !letterbox_center_inside(
                               o.rect.x, o.rect.y, o.rect.width, o.rect.height, lb);
                       }),
        proposals.end());

    qsort_descent_inplace(proposals);

    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, NMS_THRESHOLD);

    float scale_letterbox = 1.0f;
    if ((LETTERBOX_ROWS * 1.0f / img_h) < (LETTERBOX_COLS * 1.0f / img_w))
        scale_letterbox = LETTERBOX_ROWS * 1.0f / img_h;
    else
        scale_letterbox = LETTERBOX_COLS * 1.0f / img_w;

    int resize_cols = (int)round(scale_letterbox * img_w);
    int resize_rows = (int)round(scale_letterbox * img_h);
    int hpad = (LETTERBOX_ROWS - resize_rows);
    int wpad = (LETTERBOX_COLS - resize_cols);
    float ratio_y = (float)img_h / resize_rows;
    float ratio_x = (float)img_w / resize_cols;

    int count = 0;
    for (int idx : picked) {
        if (count >= OBJ_NUMB_MAX_SIZE) break;

        Object obj = proposals[idx];
        float x0 = (obj.rect.x - (wpad / 2)) * ratio_x;
        float y0 = (obj.rect.y - (hpad / 2)) * ratio_y;
        float x1 = (obj.rect.x + obj.rect.width - (wpad / 2)) * ratio_x;
        float y1 = (obj.rect.y + obj.rect.height - (hpad / 2)) * ratio_y;

        x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

        DetectResult r;
        r.x1 = x0;
        r.y1 = y0;
        r.x2 = x1;
        r.y2 = y1;
        r.confidence = obj.prob;
        r.class_id = obj.label;
        results.push_back(r);
        count++;
    }

    return count;
}
