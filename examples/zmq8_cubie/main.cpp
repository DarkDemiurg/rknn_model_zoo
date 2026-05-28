/**
 * @file main.cpp
 * @brief ZMQ-based YOLOv8 inference для Radxa Cubie A7Z (Allwinner A733 NPU).
 *
 * Архитектура: 3-стадийный конвейер (pipeline) для достижения ~90 FPS:
 *   1. Capture thread  — захват кадров с камеры (MJPEG USB)
 *   2. Preprocess thread — letterbox resize через G2D (аппаратный 2D ускоритель)
 *   3. Inference thread — NPU inference + постпроцессинг + отправка по ZMQ
 *
 * Между стадиями используются lock-free очереди для минимизации задержек.
 *
 * Аргументы командной строки:
 *   ./zmq8_cubie <model_path> <source> [-w WIDTH] [-h HEIGHT]
 *   model_path — путь к .nb файлу модели
 *   source     — номер камеры (0,1,...) или путь к видеофайлу
 *   -w WIDTH   — ширина захвата камеры (по умолчанию 960)
 *   -h HEIGHT  — высота захвата камеры (по умолчанию 720)
 */
#include <iostream>
#include <string>
#include <sstream>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <getopt.h>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <chrono>
#include <iomanip>

#include <opencv2/opencv.hpp>
#include <zmq.hpp>

#include "model_config.h"
#include "g2d_resize.h"
#include "npu_yolov8.h"
#include "postprocess.h"

using namespace cv;
using namespace std;

// Параметры по умолчанию
#define DEFAULT_WIDTH  960
#define DEFAULT_HEIGHT 720
#define FPS            90
#define ZMQ_ADDR       "tcp://127.0.0.1:5757"

// Размер очередей конвейера (кольцевой буфер)
#define QUEUE_SIZE     4

// ============================================================================
// Потокобезопасная очередь с ограниченным размером
// ============================================================================
template<typename T>
class BoundedQueue {
public:
    explicit BoundedQueue(size_t max_size) : max_size_(max_size), stopped_(false) {}

    // Помещает элемент в очередь. Если очередь полна — отбрасывает старый кадр.
    void push(T &&item) {
        std::unique_lock<std::mutex> lock(mtx_);
        if (queue_.size() >= max_size_) {
            queue_.pop(); // Отбрасываем старый кадр для минимизации latency
        }
        queue_.push(std::move(item));
        cv_.notify_one();
    }

    // Извлекает элемент. Блокируется если очередь пуста.
    bool pop(T &item) {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [this] { return !queue_.empty() || stopped_; });
        if (stopped_ && queue_.empty()) return false;
        item = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    void stop() {
        std::unique_lock<std::mutex> lock(mtx_);
        stopped_ = true;
        cv_.notify_all();
    }

private:
    std::queue<T> queue_;
    std::mutex mtx_;
    std::condition_variable cv_;
    size_t max_size_;
    bool stopped_;
};

// ============================================================================
// Структуры данных конвейера
// ============================================================================

// Кадр с камеры
struct CapturedFrame {
    cv::Mat img;
};

// Предобработанный кадр (готов для NPU)
struct PreprocessedFrame {
    std::vector<unsigned char> rgb_data; // 640x640x3 RGB
    int orig_w;
    int orig_h;
    cv::Mat orig_img; // Оригинал для отправки по ZMQ
};

// ============================================================================
// Глобальные переменные конвейера
// ============================================================================
static std::atomic<bool> g_running(true);
static BoundedQueue<CapturedFrame> g_capture_queue(QUEUE_SIZE);
static BoundedQueue<PreprocessedFrame> g_preprocess_queue(QUEUE_SIZE);

// ============================================================================
// Поток захвата кадров
// ============================================================================
static void capture_thread_func(VideoCapture &vid)
{
    while (g_running) {
        CapturedFrame frame;
        vid >> frame.img;
        if (frame.img.empty()) {
            fprintf(stderr, "[Capture] Empty frame, stopping\n");
            g_running = false;
            break;
        }
        g_capture_queue.push(std::move(frame));
    }
    g_capture_queue.stop();
}

// ============================================================================
// Поток препроцессинга (G2D letterbox resize)
// ============================================================================
static void preprocess_thread_func()
{
    while (g_running) {
        CapturedFrame frame;
        if (!g_capture_queue.pop(frame)) break;

        PreprocessedFrame pf;
        pf.orig_w = frame.img.cols;
        pf.orig_h = frame.img.rows;
        pf.orig_img = frame.img;
        pf.rgb_data.resize(LETTERBOX_COLS * LETTERBOX_ROWS * 3);

        // Аппаратный resize через G2D (или OpenCV fallback)
        g2d_letterbox_resize(frame.img, pf.rgb_data.data(), LETTERBOX_COLS, LETTERBOX_ROWS);

        g_preprocess_queue.push(std::move(pf));
    }
    g_preprocess_queue.stop();
}

// ============================================================================
// Поток inference + ZMQ
// ============================================================================
static void inference_thread_func(NpuYolov8 &npu, zmq::socket_t &sock)
{
    double total_time = 0;
    int frame_counter = 0;

    while (g_running) {
        PreprocessedFrame pf;
        if (!g_preprocess_queue.pop(pf)) break;

        auto start = std::chrono::steady_clock::now();

        // NPU inference
        std::vector<DetectResult> results;
        int ret = npu.infer(pf.rgb_data.data(), pf.orig_w, pf.orig_h, results);
        if (ret != 0) {
            fprintf(stderr, "[Inference] NPU inference failed\n");
            continue;
        }

        // Формируем текстовое сообщение с результатами детекции
        std::string msg;
        char text[256];
        for (auto &det : results) {
            const char *name = (det.class_id >= 0 && det.class_id < (int)g_classes_name.size())
                               ? g_classes_name[det.class_id].c_str() : "unknown";
            sprintf(text, "%s@%d,%d,%d,%d@%.2f;",
                    name,
                    (int)det.x1, (int)det.y1, (int)det.x2, (int)det.y2,
                    det.confidence);
            msg += text;
        }
        if (msg.empty()) msg = "empty";

        // Отправляем по ZMQ: [текст детекций] [RGB данные 640x640]
        sock.send(zmq::buffer(msg), zmq::send_flags::sndmore);
        sock.send(zmq::buffer(pf.rgb_data.data(), pf.rgb_data.size()), zmq::send_flags::none);

        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> diff = end - start;
        total_time += diff.count();
        frame_counter++;

        // Вывод FPS каждые 30 кадров
        if (frame_counter % 30 == 0) {
            double avg_fps = frame_counter / total_time;
            cout << "\t FPS: " << std::fixed << std::setw(11) << std::setprecision(6)
                 << avg_fps << " time: " << total_time << endl;
            total_time = 0;
            frame_counter = 0;
        }
    }
}

// ============================================================================
// Разбор аргументов командной строки
// ============================================================================
struct AppConfig {
    int width = DEFAULT_WIDTH;
    int height = DEFAULT_HEIGHT;
};

static void print_usage(const char *prog)
{
    cout << "Usage: " << prog << " <model_path> <source> [-w WIDTH] [-h HEIGHT]" << endl;
    cout << "  model_path  - path to .nb model file" << endl;
    cout << "  source      - camera index (0,1,...) or video file path" << endl;
    cout << "  -w WIDTH    - capture width (default " << DEFAULT_WIDTH << ")" << endl;
    cout << "  -h HEIGHT   - capture height (default " << DEFAULT_HEIGHT << ")" << endl;
}

static AppConfig parse_args(int argc, char *argv[])
{
    AppConfig cfg;
    int opt;
    while ((opt = getopt(argc, argv, "w:h:")) != -1) {
        switch (opt) {
            case 'w': cfg.width = atoi(optarg); break;
            case 'h': cfg.height = atoi(optarg); break;
        }
    }
    return cfg;
}

// ============================================================================
// main
// ============================================================================
int main(int argc, char **argv)
{
    if (argc < 3) {
        print_usage(argv[0]);
        return -1;
    }

    const char *model_path = argv[1];
    const char *source = argv[2];

    AppConfig cfg = parse_args(argc, argv);
    cout << "Camera resolution: [" << cfg.width << "x" << cfg.height << "]" << endl;

    // --- Инициализация G2D ---
    g2d_init();

    // --- Инициализация NPU ---
    NpuYolov8 npu;
    if (npu.init(model_path) != 0) {
        fprintf(stderr, "Failed to init NPU model: %s\n", model_path);
        g2d_deinit();
        return -1;
    }

    // --- Открытие камеры ---
    VideoCapture vid;
    if (isdigit(*source)) {
        int cam = (int)strtol(source, nullptr, 10);

        // GStreamer pipeline для MJPEG USB камеры
        stringstream pipeline_builder;
        pipeline_builder << "v4l2src device=/dev/video" << cam
                         << " ! image/jpeg, width=" << cfg.width
                         << ", height=" << cfg.height
                         << ", framerate=" << FPS << "/1"
                         << " ! jpegdec ! videoconvert"
                         << " ! video/x-raw, format=BGR"
                         << " ! appsink drop=true sync=false";
        string pipeline = pipeline_builder.str();

        cout << "GStreamer pipeline:\n\t" << pipeline << endl;
        vid.open(pipeline, CAP_GSTREAMER);

        if (!vid.isOpened()) {
            // Fallback: прямое открытие камеры
            cout << "GStreamer failed, trying direct V4L2..." << endl;
            vid.open(cam);
            if (vid.isOpened()) {
                vid.set(CAP_PROP_FRAME_WIDTH, cfg.width);
                vid.set(CAP_PROP_FRAME_HEIGHT, cfg.height);
                vid.set(CAP_PROP_FPS, FPS);
            } else {
                cerr << "Cannot open camera " << cam << endl;
                g2d_deinit();
                return -1;
            }
        }
        cout << "Camera opened." << endl;
    } else {
        cout << "Opening video: " << source << endl;
        vid.open(source);
        if (!vid.isOpened()) {
            cerr << "Cannot open video: " << source << endl;
            g2d_deinit();
            return -1;
        }
    }

    // Проверяем первый кадр
    Mat test_frame;
    vid >> test_frame;
    if (test_frame.empty()) {
        cerr << "First frame is empty!" << endl;
        g2d_deinit();
        return -1;
    }
    cout << "First frame: " << test_frame.cols << "x" << test_frame.rows << endl;

    // --- Инициализация ZMQ ---
    zmq::context_t zmq_ctx;
    zmq::socket_t sock(zmq_ctx, zmq::socket_type::pub);
    sock.set(zmq::sockopt::sndbuf, LETTERBOX_COLS * LETTERBOX_ROWS * 3 * 4);
    sock.bind(ZMQ_ADDR);
    cout << "ZMQ publisher bound to " << ZMQ_ADDR << endl;

    // --- Запуск конвейера ---
    std::thread capture_thread(capture_thread_func, std::ref(vid));
    std::thread preprocess_thread(preprocess_thread_func);
    std::thread inference_thread(inference_thread_func, std::ref(npu), std::ref(sock));

    // Ожидание завершения
    capture_thread.join();
    preprocess_thread.join();
    inference_thread.join();

    // --- Очистка ---
    sock.close();
    vid.release();
    g2d_deinit();

    cout << "Done." << endl;
    return 0;
}
