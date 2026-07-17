/**
 * @file fisheye_preprocess.cpp
 */
#include "fisheye_preprocess.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <algorithm>
#include <cmath>

static void center_crop(cv::Mat &frame, float ratio)
{
    if (ratio <= 0.0f || ratio >= 1.0f)
        return;

    int cw = std::max(1, (int)(frame.cols * ratio));
    int ch = std::max(1, (int)(frame.rows * ratio));
    int x = (frame.cols - cw) / 2;
    int y = (frame.rows - ch) / 2;
    frame = frame(cv::Rect(x, y, cw, ch)).clone();
}

static void apply_lens_correction(cv::Mat &frame, float k1, float k2)
{
    if (frame.empty())
        return;

    const int w = frame.cols;
    const int h = frame.rows;

    static int cached_w = 0, cached_h = 0;
    static float cached_k1 = 0, cached_k2 = 0;
    static cv::Mat map1, map2;

    if (w != cached_w || h != cached_h || k1 != cached_k1 || k2 != cached_k2) {
        cv::Mat K = (cv::Mat_<double>(3, 3) <<
            (double)w, 0.0, (double)w * 0.5,
            0.0, (double)h, (double)h * 0.5,
            0.0, 0.0, 1.0);
        cv::Mat D = (cv::Mat_<double>(4, 1) << (double)k1, (double)k2, 0.0, 0.0);
        cv::Mat R = cv::Mat::eye(3, 3, CV_64F);

        cv::fisheye::initUndistortRectifyMap(
            K, D, R, K, cv::Size(w, h), CV_16SC2, map1, map2);

        cached_w = w;
        cached_h = h;
        cached_k1 = k1;
        cached_k2 = k2;
    }

    cv::Mat corrected;
    cv::remap(frame, corrected, map1, map2, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    frame = corrected;
}

void apply_fisheye_preprocess(cv::Mat &frame, const FisheyeConfig &cfg)
{
    if (frame.empty())
        return;

    center_crop(frame, cfg.crop_ratio);

    if (cfg.lens_correction)
        apply_lens_correction(frame, cfg.k1, cfg.k2);
}
