/**
 * @file fisheye_preprocess.h
 * @brief Коррекция fisheye-кадра перед YOLO (crop центра + lens undistort).
 *
 * По опыту zmq_dragon: crop 0.5 обычно эффективнее lenscorrection.
 */
#ifndef _FISHEYE_PREPROCESS_H_
#define _FISHEYE_PREPROCESS_H_

#include <opencv2/core/core.hpp>

struct FisheyeConfig {
    float crop_ratio = 0.0f;       // 0 = выкл, 0.5 = центральные 50% кадра
    bool lens_correction = false;  // OpenCV fisheye undistort
    float k1 = -0.2f;
    float k2 = 0.0f;
};

/** Crop центра + опциональная коррекция дисторсии. Модифицирует frame in-place. */
void apply_fisheye_preprocess(cv::Mat &frame, const FisheyeConfig &cfg);

#endif
