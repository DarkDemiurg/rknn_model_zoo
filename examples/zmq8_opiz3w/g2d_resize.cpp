/**
 * @file g2d_resize.cpp
 * @brief Оптимизированный letterbox resize для NPU input.
 *
 * G2D на A733 работает через DMA heap, но для масштабирования 960x720→640x640
 * CPU с INTER_NEAREST быстрее (~3ms vs G2D ~10ms+ с overhead копирования в DMA).
 * G2D выгоден для больших разрешений (4K) и поворотов.
 *
 * Используем оптимизированный CPU resize.
 */
#include "g2d_resize.h"
#include "model_config.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <string.h>

bool g2d_init()
{
    fprintf(stderr, "[Resize] Using optimized CPU letterbox (INTER_NEAREST)\n");
    return false;
}

void g2d_deinit() {}

void g2d_letterbox_resize(const cv::Mat &src, unsigned char *dst, int dst_w, int dst_h)
{
    float scale = std::min((float)dst_w / src.cols, (float)dst_h / src.rows);
    int resize_w = (int)(scale * src.cols);
    int resize_h = (int)(scale * src.rows);
    int top = (dst_h - resize_h) / 2;
    int left = (dst_w - resize_w) / 2;

    // Заполняем серым (letterbox padding)
    memset(dst, 114, dst_w * dst_h * 3);

    // Resize (INTER_NEAREST — самый быстрый)
    cv::Mat resized;
    cv::resize(src, resized, cv::Size(resize_w, resize_h), 0, 0, cv::INTER_NEAREST);

    // BGR→RGB
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

    // Копируем в центр
    cv::Mat out(dst_h, dst_w, CV_8UC3, dst);
    resized.copyTo(out(cv::Rect(left, top, resize_w, resize_h)));
}
