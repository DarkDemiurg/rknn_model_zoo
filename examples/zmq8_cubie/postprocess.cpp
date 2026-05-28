/**
 * @file postprocess.cpp
 * @brief Постпроцессинг YOLOv8 для Allwinner NPU.
 *        Модель обрезана до 6 выходов (3 масштаба × 2: grid + score).
 *        Формат совпадает с примером из awnpu_model_zoo.
 */
#include "postprocess.h"
#include "model_config.h"

#include <cmath>
#include <algorithm>
#include <cstring>
#include <cfloat>

struct Object {
    float x, y, w, h;
    float prob;
    int label;
};

static inline float sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

static inline float desigmoid(float x)
{
    return -logf(1.0f / x - 1.0f);
}

static float softmax_dfl(const float *src, int length)
{
    float alpha = -FLT_MAX;
    for (int i = 0; i < length; i++)
        if (src[i] > alpha) alpha = src[i];

    float denominator = 0;
    float dis_sum = 0;
    for (int i = 0; i < length; i++) {
        float val = expf(src[i] - alpha);
        denominator += val;
        dis_sum += i * val;
    }
    return dis_sum / denominator;
}

static float iou(const Object &a, const Object &b)
{
    float x1 = std::max(a.x, b.x);
    float y1 = std::max(a.y, b.y);
    float x2 = std::min(a.x + a.w, b.x + b.w);
    float y2 = std::min(a.y + a.h, b.y + b.h);
    float inter = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float area_a = a.w * a.h;
    float area_b = b.w * b.h;
    return inter / (area_a + area_b - inter);
}

/**
 * Генерация proposals для одного масштаба (stride 8/16/32).
 * Формат выходов NPU: grid [64, grid_h*grid_w], score [num_class, grid_h*grid_w]
 */
static void generate_proposals(int stride, const float *feat_grid, const float *feat_score,
                               float prob_threshold, std::vector<Object> &objects)
{
    const int grid_w = LETTERBOX_COLS / stride;
    const int grid_h = LETTERBOX_ROWS / stride;
    const int grid_size = grid_w * grid_h;
    const int reg_max = 16;
    const int num_class = CLASS_NUM;

    float deprob = desigmoid(prob_threshold);

    for (int y = 0; y < grid_h; y++) {
        for (int x = 0; x < grid_w; x++) {
            int idx = y * grid_w + x;

            // Находим класс с максимальным score
            int best_class = -1;
            float best_score = -FLT_MAX;
            for (int c = 0; c < num_class; c++) {
                float s = feat_score[c * grid_size + idx];
                if (s > best_score) {
                    best_score = s;
                    best_class = c;
                }
            }

            if (best_score < deprob) continue;
            best_score = sigmoid(best_score);
            if (best_score < prob_threshold) continue;

            // Декодируем bbox через DFL (Distribution Focal Loss)
            float pred_grid[64];
            for (int i = 0; i < 64; i++)
                pred_grid[i] = feat_grid[i * grid_size + idx];

            float x0 = x + 0.5f - softmax_dfl(pred_grid, reg_max);
            float y0 = y + 0.5f - softmax_dfl(pred_grid + 16, reg_max);
            float x1 = x + 0.5f + softmax_dfl(pred_grid + 32, reg_max);
            float y1 = y + 0.5f + softmax_dfl(pred_grid + 48, reg_max);

            Object obj;
            obj.x = x0 * stride;
            obj.y = y0 * stride;
            obj.w = (x1 - x0) * stride;
            obj.h = (y1 - y0) * stride;
            obj.label = best_class;
            obj.prob = best_score;
            objects.push_back(obj);
        }
    }
}

int postprocess_yolov8_6(float **output, int img_w, int img_h,
                         std::vector<DetectResult> &results)
{
    results.clear();

    std::vector<Object> proposals;

    // 3 масштаба: stride 8, 16, 32
    // output[0]=grid_8, output[1]=score_8
    // output[2]=grid_16, output[3]=score_16
    // output[4]=grid_32, output[5]=score_32
    generate_proposals(8,  output[0], output[1], SCORE_THRESHOLD, proposals);
    generate_proposals(16, output[2], output[3], SCORE_THRESHOLD, proposals);
    generate_proposals(32, output[4], output[5], SCORE_THRESHOLD, proposals);

    // Сортировка по confidence (убывание)
    std::sort(proposals.begin(), proposals.end(),
              [](const Object &a, const Object &b) { return a.prob > b.prob; });

    // NMS
    std::vector<bool> suppressed(proposals.size(), false);
    std::vector<Object> picked;

    for (size_t i = 0; i < proposals.size(); i++) {
        if (suppressed[i]) continue;
        picked.push_back(proposals[i]);
        for (size_t j = i + 1; j < proposals.size(); j++) {
            if (suppressed[j]) continue;
            if (iou(proposals[i], proposals[j]) > NMS_THRESHOLD)
                suppressed[j] = true;
        }
    }

    // Преобразование координат из letterbox в оригинальное изображение
    float scale = std::min((float)LETTERBOX_COLS / img_w, (float)LETTERBOX_ROWS / img_h);
    int resize_w = (int)(scale * img_w);
    int resize_h = (int)(scale * img_h);
    float pad_w = (LETTERBOX_COLS - resize_w) / 2.0f;
    float pad_h = (LETTERBOX_ROWS - resize_h) / 2.0f;

    for (auto &obj : picked) {
        DetectResult r;
        r.x1 = std::max(0.0f, (obj.x - pad_w) / scale);
        r.y1 = std::max(0.0f, (obj.y - pad_h) / scale);
        r.x2 = std::min((float)(img_w - 1), (obj.x + obj.w - pad_w) / scale);
        r.y2 = std::min((float)(img_h - 1), (obj.y + obj.h - pad_h) / scale);
        r.confidence = obj.prob;
        r.class_id = obj.label;
        results.push_back(r);

        if ((int)results.size() >= OBJ_NUMB_MAX_SIZE) break;
    }

    return (int)results.size();
}
