/**
 * @file g2d_resize.cpp
 * @brief Реализация аппаратного resize через Allwinner G2D.
 *        G2D выполняет масштабирование и конвертацию цвета аппаратно,
 *        разгружая CPU для других задач (постпроцессинг, ZMQ).
 *        При недоступности G2D — fallback на OpenCV.
 */
#include "g2d_resize.h"
#include "model_config.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/types.h>

// Подключаем заголовок G2D
#include "sunxi-g2d.h"

static int g2d_fd = -1;

bool g2d_init()
{
    g2d_fd = open("/dev/g2d", O_RDWR);
    if (g2d_fd < 0) {
        fprintf(stderr, "[G2D] Cannot open /dev/g2d, using OpenCV fallback\n");
        return false;
    }
    fprintf(stderr, "[G2D] Hardware 2D accelerator initialized\n");
    return true;
}

void g2d_deinit()
{
    if (g2d_fd >= 0) {
        close(g2d_fd);
        g2d_fd = -1;
    }
}

/**
 * Fallback: letterbox resize через OpenCV (CPU).
 */
static void opencv_letterbox_resize(const cv::Mat &src, unsigned char *dst, int dst_w, int dst_h)
{
    cv::Mat img;
    cv::cvtColor(src, img, cv::COLOR_BGR2RGB);

    float scale = std::min((float)dst_w / img.cols, (float)dst_h / img.rows);
    int resize_w = (int)(scale * img.cols);
    int resize_h = (int)(scale * img.rows);

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(resize_w, resize_h));

    // Заполняем выходной буфер серым (114)
    cv::Mat out(dst_h, dst_w, CV_8UC3, dst);
    out.setTo(cv::Scalar(114, 114, 114));

    int top = (dst_h - resize_h) / 2;
    int left = (dst_w - resize_w) / 2;

    // Копируем масштабированное изображение в центр
    resized.copyTo(out(cv::Rect(left, top, resize_w, resize_h)));
}

/**
 * Аппаратный resize через G2D с letterbox.
 * G2D делает масштабирование + конвертацию BGR→RGB аппаратно.
 * Паддинг заполняется программно (быстрая операция memset).
 */
static bool g2d_hw_letterbox_resize(const cv::Mat &src, unsigned char *dst, int dst_w, int dst_h)
{
    float scale = std::min((float)dst_w / src.cols, (float)dst_h / src.rows);
    int resize_w = (int)(scale * src.cols);
    int resize_h = (int)(scale * src.rows);
    int top = (dst_h - resize_h) / 2;
    int left = (dst_w - resize_w) / 2;

    // Заполняем весь выходной буфер серым (114) — паддинг letterbox
    memset(dst, 114, dst_w * dst_h * 3);

    // Настраиваем G2D blit с масштабированием
    g2d_blt_h blit;
    memset(&blit, 0, sizeof(blit));

    blit.flag_h = G2D_ROT_0; // Без поворота, только масштабирование

    // Источник: BGR изображение с камеры
    blit.src_image_h.format = G2D_FORMAT_BGR888;
    blit.src_image_h.width = src.cols;
    blit.src_image_h.height = src.rows;
    blit.src_image_h.align[0] = src.step[0]; // stride
    blit.src_image_h.clip_rect.x = 0;
    blit.src_image_h.clip_rect.y = 0;
    blit.src_image_h.clip_rect.w = src.cols;
    blit.src_image_h.clip_rect.h = src.rows;
    blit.src_image_h.laddr[0] = (__u32)(uintptr_t)src.data;
    blit.src_image_h.use_phy_addr = 0;
    blit.src_image_h.color_range = COLOR_RANGE_0_255;
    blit.src_image_h.alpha = 0xFF;
    blit.src_image_h.mode = G2D_GLOBAL_ALPHA;

    // Назначение: RGB буфер, смещённый на letterbox offset
    unsigned char *dst_offset = dst + (top * dst_w + left) * 3;
    blit.dst_image_h.format = G2D_FORMAT_RGB888;
    blit.dst_image_h.width = dst_w;
    blit.dst_image_h.height = dst_h;
    blit.dst_image_h.align[0] = dst_w * 3;
    blit.dst_image_h.clip_rect.x = left;
    blit.dst_image_h.clip_rect.y = top;
    blit.dst_image_h.clip_rect.w = resize_w;
    blit.dst_image_h.clip_rect.h = resize_h;
    blit.dst_image_h.laddr[0] = (__u32)(uintptr_t)dst;
    blit.dst_image_h.use_phy_addr = 0;
    blit.dst_image_h.color_range = COLOR_RANGE_0_255;
    blit.dst_image_h.alpha = 0xFF;
    blit.dst_image_h.mode = G2D_GLOBAL_ALPHA;

    int ret = ioctl(g2d_fd, G2D_CMD_BITBLT_H, &blit);
    if (ret < 0) {
        return false;
    }

    return true;
}

void g2d_letterbox_resize(const cv::Mat &src, unsigned char *dst, int dst_w, int dst_h)
{
    // Пробуем аппаратный G2D
    if (g2d_fd >= 0) {
        if (g2d_hw_letterbox_resize(src, dst, dst_w, dst_h)) {
            return;
        }
        // G2D не сработал — fallback
        fprintf(stderr, "[G2D] HW resize failed, falling back to OpenCV\n");
    }

    // OpenCV fallback
    opencv_letterbox_resize(src, dst, dst_w, dst_h);
}
