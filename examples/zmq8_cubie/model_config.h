/**
 * @file model_config.h
 * @brief Конфигурация модели YOLOv8 для Radxa Cubie A7Z (Allwinner A733 NPU)
 */
#ifndef _MODEL_CONFIG_H_
#define _MODEL_CONFIG_H_

#include <iostream>
#include <vector>

// COCO dataset, 80 классов
#define CLASS_NUM           80

// Размер входа модели (letterbox)
#define LETTERBOX_ROWS      640
#define LETTERBOX_COLS      640

// Пороги детекции
#define SCORE_THRESHOLD     0.6f
#define NMS_THRESHOLD       0.45f

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
