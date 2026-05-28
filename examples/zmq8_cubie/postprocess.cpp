/**
 * @file postprocess.cpp
 * @brief Оптимизированный постпроцессинг YOLOv8 для Allwinner NPU.
 *
 * Формат выходов NPU (HWC):
 *   output[0]: grid  stride 8,  shape [80, 80, 64] → data[(y*80+x)*64 + c]
 *   output[1]: score stride 8,  shape [80, 80, 80] → data[(y*80+x)*80 + c]
 *   output[2]: grid  stride 16, shape [40, 40, 64]
 *   output[3]: score stride 16, shape [40, 40, 80]
 *   output[4]: grid  stride 32, shape [20, 20, 64]
 *   output[5]: score stride 32, shape [20, 20, 80]
 *
 * HWC формат — данные для одной ячейки лежат последовательно в памяти.
 * Это cache-friendly, транспозиция не нужна.
 */
#include "postprocess.h"
#include "model_config.h"

#include <cmath>
#include <algorithm>
#include <cstring>
#include <cfloat>
#include <vector>

struct Object {
    float x, y, w, h;
    float prob;
    int label;
};

static inline float fast_sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

static inline float desigmoid(float x)
{
    return -logf(1.0f / x - 1.0f);
}

static inline float softmax_dfl(const float *src)
{
    float alpha = src[0];
    for (int i = 1; i < 16; i++)
        if (src[i] > alpha) alpha = src[i];

    float sum = 0, weighted = 0;
    for (int i = 0; i < 16; i++) {
        float v = expf(src[i] - alpha);
        sum += v;
        weighted += i * v;
    }
    return weighted / sum;
}

static float iou(const Object &a, const Object &b)
{
    float x1 = std::max(a.x, b.x);
    float y1 = std::max(a.y, b.y);
    float x2 = std::min(a.x + a.w, b.x + b.w);
    float y2 = std::min(a.y + a.h, b.y + b.h);
    float inter = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    return inter / (a.w * a.h + b.w * b.h - inter);
}

/**
 * Генерация proposals для одного масштаба.
 * Данные в формате HWC: feat[idx * channels + c]
 * где idx = y * grid_w + x
 */
static void generate_proposals(int stride, const float *feat_grid, const float *feat_score,
                               float prob_threshold, std::vector<Object> &objects,
                               const std::vector<int> &class_filter)
{
    const int grid_w = LETTERBOX_COLS / stride;
    const int grid_h = LETTERBOX_ROWS / stride;
    const int grid_size = grid_w * grid_h;
    const int num_class = CLASS_NUM;
    const float deprob = desigmoid(prob_threshold);
    const int max_proposals = 100; // Ограничиваем количество proposals на масштаб
    const bool filter_active = !class_filter.empty();

    for (int idx = 0; idx < grid_size; idx++) {
        // Score для этой ячейки лежит последовательно (HWC)
        const float *scores = feat_score + idx * num_class;

        // Быстрый поиск максимума — только по нужным классам
        float best_score = -FLT_MAX;
        int best_class = 0;

        if (filter_active) {
            // Проверяем только указанные классы (очень быстро)
            for (int c : class_filter) {
                if (scores[c] > best_score) {
                    best_score = scores[c];
                    best_class = c;
                }
            }
        } else {
            // Все классы
            int c = 0;
            for (; c + 3 < num_class; c += 4) {
                float s0 = scores[c], s1 = scores[c+1], s2 = scores[c+2], s3 = scores[c+3];
                if (s0 > best_score) { best_score = s0; best_class = c; }
                if (s1 > best_score) { best_score = s1; best_class = c+1; }
                if (s2 > best_score) { best_score = s2; best_class = c+2; }
                if (s3 > best_score) { best_score = s3; best_class = c+3; }
            }
            for (; c < num_class; c++) {
                if (scores[c] > best_score) { best_score = scores[c]; best_class = c; }
            }
        }

        if (best_score < deprob) continue;

        float score = fast_sigmoid(best_score);
        if (score < prob_threshold) continue;

        // Ограничиваем количество proposals для стабильного FPS
        if ((int)objects.size() >= max_proposals) return;

        int x = idx % grid_w;
        int y = idx / grid_w;

        // Grid для этой ячейки тоже лежит последовательно (HWC): 64 значений подряд
        const float *grid = feat_grid + idx * 64;

        float x0 = x + 0.5f - softmax_dfl(grid);
        float y0 = y + 0.5f - softmax_dfl(grid + 16);
        float x1 = x + 0.5f + softmax_dfl(grid + 32);
        float y1 = y + 0.5f + softmax_dfl(grid + 48);

        Object obj;
        obj.x = x0 * stride;
        obj.y = y0 * stride;
        obj.w = (x1 - x0) * stride;
        obj.h = (y1 - y0) * stride;
        obj.label = best_class;
        obj.prob = score;
        objects.push_back(obj);
    }
}

int postprocess_yolov8_6(float **output, int img_w, int img_h,
                         std::vector<DetectResult> &results,
                         const std::vector<int> &class_filter)
{
    results.clear();

    std::vector<Object> proposals;
    proposals.reserve(256);

    generate_proposals(32, output[4], output[5], SCORE_THRESHOLD, proposals, class_filter);
    generate_proposals(16, output[2], output[3], SCORE_THRESHOLD, proposals, class_filter);
    generate_proposals(8,  output[0], output[1], SCORE_THRESHOLD, proposals, class_filter);

    // Сортировка
    std::sort(proposals.begin(), proposals.end(),
              [](const Object &a, const Object &b) { return a.prob > b.prob; });

    // NMS
    std::vector<int> picked;
    picked.reserve(64);
    std::vector<bool> suppressed(proposals.size(), false);

    for (size_t i = 0; i < proposals.size(); i++) {
        if (suppressed[i]) continue;
        picked.push_back(i);
        for (size_t j = i + 1; j < proposals.size(); j++) {
            if (suppressed[j]) continue;
            if (iou(proposals[i], proposals[j]) > NMS_THRESHOLD)
                suppressed[j] = true;
        }
    }

    // Координаты letterbox → оригинал
    float scale = std::min((float)LETTERBOX_COLS / img_w, (float)LETTERBOX_ROWS / img_h);
    float pad_w = (LETTERBOX_COLS - scale * img_w) * 0.5f;
    float pad_h = (LETTERBOX_ROWS - scale * img_h) * 0.5f;
    float inv_scale = 1.0f / scale;

    int count = 0;
    for (int i : picked) {
        if (count >= OBJ_NUMB_MAX_SIZE) break;
        Object &obj = proposals[i];
        DetectResult r;
        r.x1 = std::max(0.0f, (obj.x - pad_w) * inv_scale);
        r.y1 = std::max(0.0f, (obj.y - pad_h) * inv_scale);
        r.x2 = std::min((float)(img_w - 1), (obj.x + obj.w - pad_w) * inv_scale);
        r.y2 = std::min((float)(img_h - 1), (obj.y + obj.h - pad_h) * inv_scale);
        r.confidence = obj.prob;
        r.class_id = obj.label;
        results.push_back(r);
        count++;
    }

    return count;
}
