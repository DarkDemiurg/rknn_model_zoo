// Copyright (c) 2024 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <iostream>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>

#include <linux/videodev2.h>
#include <opencv2/opencv.hpp>
#include <thread>
#include <atomic>
#include <chrono>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <ctime>
#include <sstream>
#include <memory>
#include <cstring>
#include <csignal>  
#include <stdexcept>  
#include <cstdlib>    
#include <fstream>

#include "yolo11.h"
#include "image_utils.h"
#include "file_utils.h"
#include "image_drawing.h"

#include <RgaApi.h>
#include <RgaUtils.h>
#include <im2d.hpp>

#include "dma_alloc.h"
#include "queue.h"

constexpr const int width = 1920;
constexpr const int height = 1080;

using namespace cv;
using namespace std;

#define BUFFER_COUNT 4

atomic<bool> g_exit(false);

// V4L2 camera wrapper (mmap, DQ/QBUF, COPY plane[0] which contains continuous NV12 in your driver)
class V4L2Camera {
    int fd = -1;
    string dev;
    int width, height;
    int bufcount;
    vector<buffer> v4l2bufs;
public:
    V4L2Camera(const string &device, int w, int h, int count=BUFFER_COUNT)
        : dev(device), width(w), height(h), bufcount(count) {}

    bool init() {
        fd = ::open(dev.c_str(), O_RDWR | O_NONBLOCK);
        if (fd < 0) { perror("open dev"); return false; }

        v4l2_capability cap{};
        if (ioctl(fd, VIDIOC_QUERYCAP, &cap) < 0) { perror("VIDIOC_QUERYCAP"); close(fd); fd=-1; return false; }

        v4l2_format fmt{};
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        fmt.fmt.pix_mp.width = width;
        fmt.fmt.pix_mp.height = height;
        fmt.fmt.pix_mp.pixelformat = V4L2_PIX_FMT_NV12;
        fmt.fmt.pix_mp.field = V4L2_FIELD_NONE;
        fmt.fmt.pix_mp.num_planes = 2;
        if (ioctl(fd, VIDIOC_S_FMT, &fmt) < 0) { perror("VIDIOC_S_FMT"); close(fd); fd=-1; return false; }

        v4l2_requestbuffers req{};
        req.count = bufcount;
        req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        req.memory = V4L2_MEMORY_MMAP;
        if (ioctl(fd, VIDIOC_REQBUFS, &req) < 0) { perror("VIDIOC_REQBUFS"); close(fd); fd=-1; return false; }

        v4l2bufs.resize(req.count);
        for (unsigned i = 0; i < (unsigned)req.count; ++i) {
            v4l2_buffer buf{};
            v4l2_plane planes[2]{};
            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
            buf.memory = V4L2_MEMORY_MMAP;
            buf.index = i;
            buf.length = 2;
            buf.m.planes = planes;
            if (ioctl(fd, VIDIOC_QUERYBUF, &buf) < 0) { perror("VIDIOC_QUERYBUF"); cleanup(); return false; }

            // Many drivers present NV12 as single continuous in plane[0] (Y + UV),
            // therefore mapping plane[0] length is enough.
            v4l2bufs[i].length = buf.m.planes[0].length;
            v4l2bufs[i].start = mmap(nullptr, buf.m.planes[0].length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, buf.m.planes[0].m.mem_offset);
            if (v4l2bufs[i].start == MAP_FAILED) { perror("mmap"); cleanup(); return false; }

            // queue it
            if (ioctl(fd, VIDIOC_QBUF, &buf) < 0) { perror("VIDIOC_QBUF"); cleanup(); return false; }
        }

        // start stream
        v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        if (ioctl(fd, VIDIOC_STREAMON, &type) < 0) { perror("STREAMON"); cleanup(); return false; }

        return true;
    }

    void cleanup() {
        if (fd >= 0) {
            v4l2_buf_type t = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
            ioctl(fd, VIDIOC_STREAMOFF, &t);
        }
        for (auto &b : v4l2bufs) {
            if (b.start && b.length) munmap(b.start, b.length);
            b.start = nullptr; b.length = 0;
        }
        if (fd >= 0) { close(fd); fd = -1; }
    }

    ~V4L2Camera() { cleanup(); }

    // capture: copy the device plane0 content into provided NV12 DMABuffer
    bool capture_to_dmabuf(shared_ptr<NV12FrameDMABuf> &out) {
        if (!out || !out->buf) return false;
        v4l2_buffer buf{};
        v4l2_plane planes[2]{};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.length = 2;
        buf.m.planes = planes;

        int ret = ioctl(fd, VIDIOC_DQBUF, &buf);
        if (ret < 0) {
            if (errno == EAGAIN || errno == EINTR) return false;
            perror("VIDIOC_DQBUF");
            return false;
        }

        size_t bytes = buf.m.planes[0].bytesused ? buf.m.planes[0].bytesused : v4l2bufs[buf.index].length;
        // safety clamp
        bytes = min(bytes, out->buf->size);

        // copy from mmaped plane0 (driver gives continuous NV12) to DMA target buffer
        memcpy(out->buf->va, v4l2bufs[buf.index].start, bytes);

        // sync CPU->device if needed (we will import fd to RGA); keep for safety
        dma_sync_cpu_to_device(out->buf->fd);

        // requeue
        if (ioctl(fd, VIDIOC_QBUF, &buf) < 0) {
            perror("VIDIOC_QBUF");
        }
        return true;
    }
};

int rga_nv12_fd_to_bgr_fd(int src_fd, int src_size, int dst_fd, int dst_size, int width, int height) {
    // 计算对齐后的 wstride（水平 stride）
    int wstride = (width + 15) & ~15;  // 对齐到 16 像素

    rga_buffer_handle_t src_handle = importbuffer_fd(src_fd, src_size);
    rga_buffer_handle_t dst_handle = importbuffer_fd(dst_fd, dst_size);
    if (!src_handle || !dst_handle) {
        if (src_handle) releasebuffer_handle(src_handle);
        if (dst_handle) releasebuffer_handle(dst_handle);
        cerr << "importbuffer_fd failed\n";
        return -1;
    }

    // 使用 wrapbuffer_handle 初始化基本字段（width, height, fd, handle 等）
    rga_buffer_t src = wrapbuffer_handle(src_handle, width, height, RK_FORMAT_YCbCr_420_SP);
    rga_buffer_t dst = wrapbuffer_handle(dst_handle, width, height, RK_FORMAT_BGR_888);

    src.wstride = wstride;        // 水平行跨距（Y 和 UV 平面都用这个）
    src.hstride = height;         // 垂直 stride，通常等于 height（除非特殊对齐要求）

    dst.wstride = wstride;
    dst.hstride = height;

    // 设置色彩空间
    imsetColorSpace(&src, IM_YUV_BT709_LIMIT_RANGE);
    imsetColorSpace(&dst, IM_RGB_FULL);

    // 检查参数
    int ret = imcheck(src, dst, {}, {});
    if (IM_STATUS_NOERROR != ret) {
        cerr << "imcheck error: " << imStrError((IM_STATUS)ret) << endl;
        releasebuffer_handle(src_handle);
        releasebuffer_handle(dst_handle);
        return -1;
    }

    // 执行颜色转换
    ret = imcvtcolor(src, dst, RK_FORMAT_YCbCr_420_SP, RK_FORMAT_BGR_888);
    if (ret != IM_STATUS_SUCCESS) {
        cerr << "imcvtcolor failed: " << imStrError((IM_STATUS)ret) << endl;
        releasebuffer_handle(src_handle);
        releasebuffer_handle(dst_handle);
        return -1;
    }

    // 释放 handle（DMA-BUF 引用不受影响）
    releasebuffer_handle(src_handle);
    releasebuffer_handle(dst_handle);
    return 0;
}

void camera_thread_func(MemoryPool &pool, const char* cam_dev, int width, int height) {
    cout << "camera_thread started\n";
    V4L2Camera cam(cam_dev, width, height, BUFFER_COUNT);
    if (!cam.init()) {
        cerr << "camera init failed\n";
        return;
    }

    auto idleQ = pool.getIdleQueue(0);
    auto dataQ = pool.getDataQueue(0);
    if (!idleQ || !dataQ) {
        cerr << "bad queues\n";
        return;
    }

    int skip = 10;
    while (!g_exit.load(std::memory_order_relaxed)) {
        shared_ptr<NV12FrameDMABuf> frame;
        if (!idleQ->pop(frame)) { // non-blocking pop
            this_thread::sleep_for(chrono::milliseconds(1));
            continue;
        }
        if (skip > 0) {
            cam.capture_to_dmabuf(frame);
            idleQ->push(frame);   
            skip--;
            std::cout << "skip unstalbe packets, skip count = " << skip << std::endl;
            continue;        
        }
        if (!cam.capture_to_dmabuf(frame)) {
            // capture failed; return buffer to idle and continue
            idleQ->push(frame);
            this_thread::sleep_for(chrono::milliseconds(1));
            continue;
        }
        if (!dataQ->push(frame)) {
            // queue full, return to idle
            idleQ->push(frame);
            this_thread::sleep_for(chrono::milliseconds(1));
        }
    }
}

void signalHandler(int signum) {
    if (signum == SIGINT) {
        g_exit.store(true, std::memory_order_relaxed);  
        std::cout << "Good Bye!" << std::endl;
    }
    exit(signum);
}

bool mat_to_image_buffer(const cv::Mat& mat, image_buffer_t& out_buf, image_format_t fmt)
{
    if (!mat.data || mat.empty()) {
        return false;
    }

    if (!mat.isContinuous()) {
        // 不连续不能直接共享数据
        return false;
    }

    out_buf.width  = mat.cols;
    out_buf.height = mat.rows;
    out_buf.width_stride = static_cast<int>(mat.step); // 每行字节数
    out_buf.height_stride = mat.rows;
    out_buf.format = fmt;
    out_buf.virt_addr = mat.data;
    out_buf.size = mat.total() * mat.elemSize();
    out_buf.fd = -1;  // 如果有 fd，可单独赋值

    return true;
}

/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int main(int argc, char **argv)
{
    if (argc != 3)
    {
        printf("%s <cam_id> <model_path>\n", argv[0]);
        return -1;
    }

    int cam_id = std::stoi(argv[1]);
    const char* model_path = argv[2];

    std::signal(SIGINT, signalHandler);

    int ret;
    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

    init_post_process();

    ret = init_yolo11_model(model_path, &rknn_app_ctx);
    if (ret != 0)
    {
        printf("init_yolo11_model fail! ret=%d model_path=%s\n", ret, model_path);
        deinit_post_process();
        return -1;
    }

    try {
        MemoryPool pool(4, 1, width, height); // 4 preallocated NV12 dma buffers
        // create separate DMA buffer for RGA destination (BGR)
        DMABuffer dstBuf;
        size_t dst_size = size_t(width) * height * 3; // BGR888
        if (!dstBuf.alloc(DMA_HEAP_DMA32_UNCACHED_PATH, dst_size)) {
            cerr << "alloc dst dma failed\n";
            return -1;
        }

        std::string cam_dev = "/dev/video" + std::to_string(cam_id);
        // start camera thread
        thread cam_thread(camera_thread_func, std::ref(pool), cam_dev.c_str(), width, height);

        auto dataQ = pool.getDataQueue(0);
        auto idleQ = pool.getIdleQueue(0);
            
        image_buffer_t src_image;
        object_detect_result_list od_results;

    auto last_fps_time = std::chrono::steady_clock::now();
    int frame_count = 0;

    static double t = 0;
    static int f = 0;
//        string window_name = "RKNN Demo Window";
//        namedWindow(window_name);
        while (!g_exit.load(std::memory_order_relaxed)) 
        {
auto start = std::chrono::steady_clock::now();

            shared_ptr<NV12FrameDMABuf> frame;
            if (!dataQ->pop(frame)) 
	    {
                this_thread::sleep_for(chrono::milliseconds(1));
                continue;
            }

            // Run RGA: src = frame->buf->fd (NV12), dst = dstBuf.fd (BGR)
            int src_fd = frame->buf->fd;
            int src_size = frame->buf->size;
            int dst_fd = dstBuf.fd;
            int dst_size_int = int(dstBuf.size);

            int ret = rga_nv12_fd_to_bgr_fd(src_fd, src_size, dst_fd, dst_size_int, width, height);
            if (ret != 0) {
                cerr << "rga convert failed\n";
                idleQ->push(frame);
                continue;
            }

            // ensure device->cpu sync before reading dstBuf.va
            dma_sync_device_to_cpu(dst_fd);
            cv::Mat bgr(height, width, CV_8UC3, dstBuf.va);

            memset(&src_image, 0, sizeof(image_buffer_t));
            if (!mat_to_image_buffer(bgr, src_image, IMAGE_FORMAT_RGB888)) 
	    {
                printf("mat to image buffer failed\n");
                idleQ->push(frame);
                continue;
            }

            ret = inference_yolo11_model(&rknn_app_ctx, &src_image, &od_results);
            if (0 != ret) 
	    {
                idleQ->push(frame);
                continue;
            }

            // 画框和概率
            char text[256];
            for (int i = 0; i < od_results.count; i++)
            {
                object_detect_result *det_result = &(od_results.results[i]);
                printf("%s @ (%d %d %d %d) %.3f\n", coco_cls_to_name(det_result->cls_id),
                    det_result->box.left, det_result->box.top,
                    det_result->box.right, det_result->box.bottom,
                    det_result->prop);
                int x1 = det_result->box.left;
                int y1 = det_result->box.top;
                int x2 = det_result->box.right;
                int y2 = det_result->box.bottom;

//                draw_rectangle(&src_image, x1, y1, x2 - x1, y2 - y1, COLOR_BLUE, 3);

                sprintf(text, "%s %.1f%%", coco_cls_to_name(det_result->cls_id), det_result->prop * 100);

//                draw_text(&src_image, text, x1, y1 - 20, COLOR_RED, 10);
            }
		printf(text);
//            imshow("BGR", bgr);
//            imshow(window_name, bgr);
//            if (waitKey(1) == 27) {
//                cout << "exit\n";
//                break;
//            }

            // return frame to idle pool
            idleQ->push(frame);

	auto end = std::chrono::steady_clock::now();
        double cpu_time_used = std::chrono::duration<double>(end - start).count();
        t += cpu_time_used;
        f += 1;
        frame_count++;

        // Обновляем статистику FPS каждые 30 кадров
        if (frame_count >= 30)
        {
            auto current_time = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(current_time - last_fps_time).count();
            double fps = frame_count / elapsed;
            
            cout << "FPS: " << std::fixed << std::setw(6) << std::setprecision(2) 
                 << fps << " | Avg processing time: " << (t/f)*1000 << " ms" << endl;
                 
            t = 0;
            f = 0;
            frame_count = 0;
            last_fps_time = current_time;
        }
        }
        cam_thread.detach();
    }  catch (const exception &e) {
        cerr << "fatal: " << e.what() << endl;
    }

    deinit_post_process();

    

    release_yolo11_model(&rknn_app_ctx);

    return 0;
}
