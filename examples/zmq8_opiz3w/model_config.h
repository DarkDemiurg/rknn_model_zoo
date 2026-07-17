/**
 * @file model_config.h
 * @brief Конфигурация модели YOLOv8/YOLOv11 для Orange Pi Zero 3W (Allwinner A733 NPU)
 */
#ifndef _MODEL_CONFIG_H_
#define _MODEL_CONFIG_H_

#include <iostream>
#include <vector>
#include <algorithm>

// COCO dataset, 80 классов
#define CLASS_NUM           80

// Размер входа модели (letterbox)
#define LETTERBOX_ROWS      640
#define LETTERBOX_COLS      640

// Пороги детекции
#define SCORE_THRESHOLD     0.45f
#define NMS_THRESHOLD       0.45f

/** Параметры letterbox для сопоставления координат NPU ↔ исходный кадр. */
struct LetterboxLayout {
    float scale;
    float pad_w;
    float pad_h;
    float content_w;
    float content_h;
};

inline LetterboxLayout letterbox_layout(int img_w, int img_h)
{
    LetterboxLayout lb;
    lb.scale = std::min((float)LETTERBOX_COLS / img_w, (float)LETTERBOX_ROWS / img_h);
    lb.content_w = lb.scale * img_w;
    lb.content_h = lb.scale * img_h;
    lb.pad_w = (LETTERBOX_COLS - lb.content_w) * 0.5f;
    lb.pad_h = (LETTERBOX_ROWS - lb.content_h) * 0.5f;
    return lb;
}

/** Центр bbox в координатах letterbox (640×640) внутри области изображения? */
inline bool letterbox_center_inside(float x, float y, float w, float h, const LetterboxLayout &lb)
{
    float cx = x + w * 0.5f;
    float cy = y + h * 0.5f;
    return cx >= lb.pad_w && cx <= lb.pad_w + lb.content_w &&
           cy >= lb.pad_h && cy <= lb.pad_h + lb.content_h;
}

static const std::vector<std::string> g_classes_name{
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic_light",
    "fire_hydrant", "stop_sign", "parking_meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports_ball", "kite", "baseball_bat", "baseball_glove", "skateboard", "surfboard",
    "tennis_racket", "bottle", "wine_glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot_dog", "pizza", "donut", "cake", "chair", "couch",
    "potted_plant", "bed", "dining_table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell_phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy_bear",
    "hair_drier", "toothbrush"
};

#endif
