/**
 * @file g2d_resize.h
 * @brief Аппаратный resize через Allwinner G2D для ускорения препроцессинга.
 *        Используется для масштабирования кадров камеры до размера входа модели (640x640).
 *        Если G2D недоступен — fallback на OpenCV resize.
 */
#ifndef _G2D_RESIZE_H_
#define _G2D_RESIZE_H_

#include <opencv2/core/core.hpp>

/**
 * Инициализация G2D устройства.
 * @return true если G2D доступен, false — будет использоваться OpenCV fallback
 */
bool g2d_init();

/**
 * Закрытие G2D устройства.
 */
void g2d_deinit();

/**
 * Аппаратный resize изображения через G2D (или OpenCV fallback).
 * Выполняет letterbox: масштабирует с сохранением пропорций и заполняет паддинг серым (114).
 *
 * @param src       Входное изображение (BGR, cv::Mat)
 * @param dst       Выходной буфер (RGB, 640x640, уже аллоцирован)
 * @param dst_w     Ширина выхода
 * @param dst_h     Высота выхода
 */
void g2d_letterbox_resize(const cv::Mat &src, unsigned char *dst, int dst_w, int dst_h);

#endif
