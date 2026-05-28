/**
 * @file g2d_resize.cpp
 * @brief Аппаратный letterbox resize через Allwinner G2D.
 *        G2D работает только с DMA буферами (/dev/dma_heap/system).
 *        Выполняет масштабирование + конвертацию BGR→RGB аппаратно.
 *        При недоступности G2D — fallback на OpenCV.
 *
 * Ref: https://docs.radxa.com/en/cubie/a7z/app-dev/npu-dev/g2d-usage-guide
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
// DMA heap ioctl — определяем вручную если заголовок недоступен
#ifndef DMA_HEAP_IOCTL_ALLOC
#include <linux/ioctl.h>
#include <stdint.h>
struct dma_heap_allocation_data {
    uint64_t len;
    uint32_t fd;
    uint32_t fd_flags;
    uint64_t heap_flags;
};
#define DMA_HEAP_IOCTL_ALLOC _IOWR('H', 0x0, struct dma_heap_allocation_data)
#endif

#include "sunxi-g2d.h"

static int g2d_fd = -1;

// DMA буфер: fd + mmap-ed виртуальный адрес
struct DmaBuf {
    int fd;
    void *vaddr;
    size_t size;
};

static DmaBuf src_dma = {-1, nullptr, 0};
static DmaBuf dst_dma = {-1, nullptr, 0};

// Размеры текущих DMA буферов
static int src_alloc_w = 0, src_alloc_h = 0;

/**
 * Аллокация DMA буфера через /dev/dma_heap/system
 */
static int alloc_dmabuf(DmaBuf *buf, size_t size)
{
    struct dma_heap_allocation_data alloc_data;
    memset(&alloc_data, 0, sizeof(alloc_data));
    alloc_data.len = size;
    alloc_data.fd_flags = O_RDWR | O_CLOEXEC;
    alloc_data.heap_flags = 0;

    int heap_fd = open("/dev/dma_heap/system", O_RDONLY);
    if (heap_fd < 0) {
        perror("[G2D] open /dev/dma_heap/system");
        return -1;
    }

    if (ioctl(heap_fd, DMA_HEAP_IOCTL_ALLOC, &alloc_data) < 0) {
        perror("[G2D] DMA_HEAP_IOCTL_ALLOC");
        close(heap_fd);
        return -1;
    }
    close(heap_fd);

    buf->fd = alloc_data.fd;
    buf->size = size;
    buf->vaddr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, buf->fd, 0);
    if (buf->vaddr == MAP_FAILED) {
        perror("[G2D] mmap");
        close(buf->fd);
        buf->fd = -1;
        buf->vaddr = nullptr;
        return -1;
    }
    return 0;
}

static void free_dmabuf(DmaBuf *buf)
{
    if (buf->vaddr && buf->vaddr != MAP_FAILED) {
        munmap(buf->vaddr, buf->size);
        buf->vaddr = nullptr;
    }
    if (buf->fd >= 0) {
        close(buf->fd);
        buf->fd = -1;
    }
    buf->size = 0;
}

bool g2d_init()
{
    g2d_fd = open("/dev/g2d", O_RDWR);
    if (g2d_fd < 0) {
        fprintf(stderr, "[G2D] Cannot open /dev/g2d, using OpenCV fallback\n");
        return false;
    }

    // Аллоцируем DMA буфер для выхода (640x640x3 RGB, фиксированный размер)
    size_t dst_size = LETTERBOX_COLS * LETTERBOX_ROWS * 3;
    if (alloc_dmabuf(&dst_dma, dst_size) < 0) {
        fprintf(stderr, "[G2D] Failed to alloc dst DMA buffer\n");
        close(g2d_fd);
        g2d_fd = -1;
        return false;
    }

    fprintf(stderr, "[G2D] Hardware 2D accelerator initialized (DMA heap)\n");
    return true;
}

void g2d_deinit()
{
    free_dmabuf(&src_dma);
    free_dmabuf(&dst_dma);
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
    float scale = std::min((float)dst_w / src.cols, (float)dst_h / src.rows);
    int resize_w = (int)(scale * src.cols);
    int resize_h = (int)(scale * src.rows);

    memset(dst, 114, dst_w * dst_h * 3);

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(resize_w, resize_h), 0, 0, cv::INTER_NEAREST);
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

    int top = (dst_h - resize_h) / 2;
    int left = (dst_w - resize_w) / 2;
    cv::Mat out(dst_h, dst_w, CV_8UC3, dst);
    resized.copyTo(out(cv::Rect(left, top, resize_w, resize_h)));
}

/**
 * Аппаратный G2D resize с DMA буферами.
 * G2D делает масштабирование + конвертацию BGR→RGB.
 */
static bool g2d_hw_resize(const cv::Mat &src, unsigned char *dst, int dst_w, int dst_h)
{
    float scale = std::min((float)dst_w / src.cols, (float)dst_h / src.rows);
    int resize_w = (int)(scale * src.cols);
    int resize_h = (int)(scale * src.rows);
    int top = (dst_h - resize_h) / 2;
    int left = (dst_w - resize_w) / 2;

    // Реаллоцируем src DMA буфер если размер изменился
    size_t src_size = src.cols * src.rows * 3;
    if (src_dma.fd < 0 || src_alloc_w != src.cols || src_alloc_h != src.rows) {
        free_dmabuf(&src_dma);
        if (alloc_dmabuf(&src_dma, src_size) < 0) return false;
        src_alloc_w = src.cols;
        src_alloc_h = src.rows;
    }

    // Копируем BGR данные в src DMA буфер
    memcpy(src_dma.vaddr, src.data, src_size);

    // Заполняем dst DMA буфер серым (letterbox padding)
    memset(dst_dma.vaddr, 114, dst_dma.size);

    // Настраиваем G2D blit: масштабирование + BGR→RGB
    g2d_blt_h blit;
    memset(&blit, 0, sizeof(blit));
    blit.flag_h = G2D_ROT_0;

    // Источник: BGR888
    blit.src_image_h.fd = src_dma.fd;
    blit.src_image_h.format = G2D_FORMAT_BGR888;
    blit.src_image_h.width = src.cols;
    blit.src_image_h.height = src.rows;
    blit.src_image_h.align[0] = src.cols * 3;
    blit.src_image_h.clip_rect.x = 0;
    blit.src_image_h.clip_rect.y = 0;
    blit.src_image_h.clip_rect.w = src.cols;
    blit.src_image_h.clip_rect.h = src.rows;
    blit.src_image_h.alpha = 0xFF;
    blit.src_image_h.mode = G2D_GLOBAL_ALPHA;
    blit.src_image_h.color_range = COLOR_RANGE_0_255;

    // Назначение: RGB888, масштабированное в центр
    blit.dst_image_h.fd = dst_dma.fd;
    blit.dst_image_h.format = G2D_FORMAT_RGB888;
    blit.dst_image_h.width = dst_w;
    blit.dst_image_h.height = dst_h;
    blit.dst_image_h.align[0] = dst_w * 3;
    blit.dst_image_h.clip_rect.x = left;
    blit.dst_image_h.clip_rect.y = top;
    blit.dst_image_h.clip_rect.w = resize_w;
    blit.dst_image_h.clip_rect.h = resize_h;
    blit.dst_image_h.alpha = 0xFF;
    blit.dst_image_h.mode = G2D_GLOBAL_ALPHA;
    blit.dst_image_h.color_range = COLOR_RANGE_0_255;

    int ret = ioctl(g2d_fd, G2D_CMD_BITBLT_H, (unsigned long)&blit);
    if (ret < 0) {
        perror("[G2D] G2D_CMD_BITBLT_H");
        return false;
    }

    // Копируем результат из DMA буфера в выходной буфер
    memcpy(dst, dst_dma.vaddr, dst_w * dst_h * 3);
    return true;
}

void g2d_letterbox_resize(const cv::Mat &src, unsigned char *dst, int dst_w, int dst_h)
{
    if (g2d_fd >= 0) {
        if (g2d_hw_resize(src, dst, dst_w, dst_h)) {
            return;
        }
        // Однократное предупреждение
        static bool warned = false;
        if (!warned) {
            fprintf(stderr, "[G2D] HW resize failed, falling back to OpenCV\n");
            warned = true;
        }
    }
    opencv_letterbox_resize(src, dst, dst_w, dst_h);
}
