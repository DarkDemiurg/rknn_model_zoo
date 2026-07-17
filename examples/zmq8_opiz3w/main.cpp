/**
 * @file main.cpp
 * @brief ZMQ-based YOLOv8 / YOLOv11 inference для Orange Pi Zero 3W (Allwinner A733 NPU).
 *
 * Архитектура: 3-стадийный конвейер (pipeline) для достижения ~90 FPS:
 *   1. Capture thread  — захват кадров с камеры (MJPEG USB)
 *   2. Preprocess thread — letterbox resize через G2D (аппаратный 2D ускоритель)
 *   3. Inference thread — NPU inference + постпроцессинг + отправка по ZMQ
 *
 * Между стадиями используются lock-free очереди для минимизации задержек.
 *
 * Аргументы командной строки:
 *   ./zmq8_opiz3w <model_path> <source> [-v 8|11] [-w WIDTH] [-h HEIGHT] [-d STEP]
 *   model_path — путь к .nb файлу модели
 *   source     — номер камеры (0,1,...) или путь к видеофайлу
 *   -w WIDTH   — ширина захвата камеры (по умолчанию 960)
 *   -h HEIGHT  — высота захвата камеры (по умолчанию 720)
 *   -d STEP    — сохранять отладочные изображения каждые STEP кадров (по умолчанию выкл)
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
#include "fisheye_preprocess.h"
#include "g2d_resize.h"
#include "npu_yolo.h"
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
static std::atomic<double> g_capture_fps(0.0);
static BoundedQueue<CapturedFrame> g_capture_queue(QUEUE_SIZE);
static BoundedQueue<PreprocessedFrame> g_preprocess_queue(QUEUE_SIZE);
static FisheyeConfig g_fisheye_cfg;

// ============================================================================
// Поток захвата кадров
// ============================================================================
static void capture_thread_func(VideoCapture &vid)
{
    using clock = std::chrono::steady_clock;
    auto batch_start = clock::now();
    int batch_count = 0;

    while (g_running) {
        CapturedFrame frame;
        vid >> frame.img;
        if (frame.img.empty()) {
            fprintf(stderr, "[Capture] Empty frame, stopping\n");
            g_running = false;
            break;
        }

        batch_count++;
        if (batch_count >= 30) {
            auto now = clock::now();
            double sec = std::chrono::duration<double>(now - batch_start).count();
            if (sec > 0)
                g_capture_fps.store(batch_count / sec);
            batch_count = 0;
            batch_start = now;
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

        apply_fisheye_preprocess(frame.img, g_fisheye_cfg);

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
static void inference_thread_func(NpuYolo &npu, zmq::socket_t &sock, int debug_step,
                                  const std::vector<int> &class_filter)
{
    double total_infer_time = 0;
    double total_loop_time = 0;
    int frame_counter = 0;
    int total_frames = 0;
    auto batch_wall_start = std::chrono::steady_clock::now();

    while (g_running) {
        PreprocessedFrame pf;
        if (!g_preprocess_queue.pop(pf)) break;

        auto loop_start = std::chrono::steady_clock::now();

        // NPU inference (только это время считаем для FPS)
        auto infer_start = std::chrono::steady_clock::now();

        std::vector<DetectResult> results;
        int ret = npu.infer(pf.rgb_data.data(), pf.orig_w, pf.orig_h, results, class_filter);
        if (ret != 0) {
            fprintf(stderr, "[Inference] NPU inference failed\n");
            continue;
        }

        auto infer_end = std::chrono::steady_clock::now();
        std::chrono::duration<double> infer_diff = infer_end - infer_start;
        total_infer_time += infer_diff.count();

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

        // Сохранение отладочного изображения с рамками
        total_frames++;
        if (debug_step > 0 && (total_frames % debug_step == 0)) {
            cv::Mat dbg_img = pf.orig_img.clone();
            for (auto &det : results) {
                const char *name = (det.class_id >= 0 && det.class_id < (int)g_classes_name.size())
                                   ? g_classes_name[det.class_id].c_str() : "?";
                cv::rectangle(dbg_img, cv::Point((int)det.x1, (int)det.y1),
                              cv::Point((int)det.x2, (int)det.y2), cv::Scalar(0, 255, 0), 2);
                sprintf(text, "%s %.0f%%", name, det.confidence * 100);
                int label_y = std::max((int)det.y1 + 15, 15);
                cv::putText(dbg_img, text, cv::Point((int)det.x1, label_y),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
            }
            std::string filename = "debug_" + std::to_string(total_frames) + ".jpg";
            cv::imwrite(filename, dbg_img);
            fprintf(stderr, "[Debug] Saved %s (%d detections)\n", filename.c_str(), (int)results.size());
        }

        auto loop_end = std::chrono::steady_clock::now();
        std::chrono::duration<double> loop_diff = loop_end - loop_start;
        total_loop_time += loop_diff.count();
        frame_counter++;

        // Вывод FPS каждые 30 кадров
        if (frame_counter % 30 == 0) {
            auto batch_wall_end = std::chrono::steady_clock::now();
            double wall_sec = std::chrono::duration<double>(batch_wall_end - batch_wall_start).count();
            double capture_fps = g_capture_fps.load();
            double pipeline_fps = wall_sec > 0 ? frame_counter / wall_sec : 0;
            double infer_fps = frame_counter / total_infer_time;
            double loop_fps = frame_counter / total_loop_time;
            cout << "\t Capture FPS: " << std::fixed << std::setprecision(1) << capture_fps
                 << "  Pipeline FPS: " << pipeline_fps
                 << "  Infer FPS: " << infer_fps
                 << "  Loop FPS: " << loop_fps
                 << "  (infer: " << std::setprecision(1) << (total_infer_time / frame_counter * 1000) << " ms)" << endl;
            total_infer_time = 0;
            total_loop_time = 0;
            frame_counter = 0;
            batch_wall_start = batch_wall_end;
        }
    }
}

// ============================================================================
// Разбор аргументов командной строки
// ============================================================================
struct AppConfig {
    int width = DEFAULT_WIDTH;
    int height = DEFAULT_HEIGHT;
    int debug_step = 0;
    int yolo_version = 8;
    FisheyeConfig fisheye;
    std::vector<int> class_filter;
};

static void print_usage(const char *prog)
{
    cout << "Usage: " << prog << " <model_path> <source> [-v 8|11] [-w WIDTH] [-h HEIGHT] [-d STEP] [-c CLASSES]" << endl;
    cout << "  model_path  - path to .nb model file" << endl;
    cout << "  source      - camera index (0,1,...) or video file path" << endl;
    cout << "  -v VERSION  - YOLO version: 8 or 11 (default 8)" << endl;
    cout << "  -w WIDTH    - capture width (default " << DEFAULT_WIDTH << ")" << endl;
    cout << "  -h HEIGHT   - capture height (default " << DEFAULT_HEIGHT << ")" << endl;
    cout << "  -d STEP     - save debug images every STEP frames (default: off)" << endl;
    cout << "  -c CLASSES  - comma-separated class filter (names or IDs, e.g. \"person,car\" or \"0,2\")" << endl;
    cout << "  -r RATIO    - center crop before detect (0.5 = central 50%%, recommended for fisheye)" << endl;
    cout << "  -f K1:K2    - fisheye lens correction (e.g. -f -0.2:0.0, try after -r)" << endl;
}

/**
 * Парсит строку классов: "person,car,0,15" → вектор индексов.
 * Поддерживает имена классов и числовые ID.
 */
static std::vector<int> parse_classes(const char *str)
{
    std::vector<int> result;
    std::string s(str);
    size_t pos = 0;
    while (pos < s.size()) {
        size_t comma = s.find(',', pos);
        if (comma == std::string::npos) comma = s.size();
        std::string token = s.substr(pos, comma - pos);
        pos = comma + 1;

        // Пробуем как число
        char *end;
        long id = strtol(token.c_str(), &end, 10);
        if (*end == '\0' && id >= 0 && id < CLASS_NUM) {
            result.push_back((int)id);
        } else {
            // Ищем по имени
            for (int i = 0; i < (int)g_classes_name.size(); i++) {
                if (g_classes_name[i] == token) {
                    result.push_back(i);
                    break;
                }
            }
        }
    }
    return result;
}

static AppConfig parse_args(int argc, char *argv[])
{
    AppConfig cfg;
    int opt;
    while ((opt = getopt(argc, argv, "w:h:d:c:v:r:f:")) != -1) {
        switch (opt) {
            case 'w': cfg.width = atoi(optarg); break;
            case 'h': cfg.height = atoi(optarg); break;
            case 'd': cfg.debug_step = atoi(optarg); break;
            case 'c': cfg.class_filter = parse_classes(optarg); break;
            case 'r':
                cfg.fisheye.crop_ratio = (float)atof(optarg);
                break;
            case 'f': {
                cfg.fisheye.lens_correction = true;
                float k1 = -0.2f, k2 = 0.0f;
                if (optarg && sscanf(optarg, "%f:%f", &k1, &k2) >= 1) {
                    cfg.fisheye.k1 = k1;
                    cfg.fisheye.k2 = k2;
                }
                break;
            }
            case 'v':
                cfg.yolo_version = atoi(optarg);
                if (cfg.yolo_version != 8 && cfg.yolo_version != 11) {
                    cerr << "Invalid -v value, use 8 or 11" << endl;
                    cfg.yolo_version = 8;
                }
                break;
        }
    }
    return cfg;
}

static std::string fourcc_to_string(int fourcc)
{
    char s[5];
    s[0] = (char)(fourcc & 0xFF);
    s[1] = (char)((fourcc >> 8) & 0xFF);
    s[2] = (char)((fourcc >> 16) & 0xFF);
    s[3] = (char)((fourcc >> 24) & 0xFF);
    s[4] = '\0';
    return std::string(s);
}

static void configure_camera(VideoCapture &vid, int width, int height)
{
    // Один кадр в буфере — чтение блокируется до нового кадра с камеры
    vid.set(CAP_PROP_BUFFERSIZE, 1);
    vid.set(CAP_PROP_FRAME_WIDTH, width);
    vid.set(CAP_PROP_FRAME_HEIGHT, height);
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

    AppConfig cfg = parse_args(argc, argv);

    const char *model_path = nullptr;
    const char *source = nullptr;
    for (int i = optind; i < argc; i++) {
        if (!model_path) model_path = argv[i];
        else if (!source) { source = argv[i]; break; }
    }

    if (!model_path || !source) {
        print_usage(argv[0]);
        return -1;
    }

    cout << "YOLO version: v" << cfg.yolo_version << endl;
    cout << "Camera resolution: [" << cfg.width << "x" << cfg.height << "]" << endl;
    if (cfg.fisheye.crop_ratio > 0.0f)
        cout << "Fisheye crop ratio: " << cfg.fisheye.crop_ratio << endl;
    if (cfg.fisheye.lens_correction)
        cout << "Fisheye lens correction: k1=" << cfg.fisheye.k1 << " k2=" << cfg.fisheye.k2 << endl;
    g_fisheye_cfg = cfg.fisheye;
    if (!cfg.class_filter.empty()) {
        cout << "Class filter: ";
        for (int id : cfg.class_filter)
            cout << g_classes_name[id] << "(" << id << ") ";
        cout << endl;
    }

    // --- Инициализация G2D ---
    g2d_init();

    // --- Инициализация NPU ---
    NpuYolo npu;
    YoloModelVersion version = (cfg.yolo_version == 11) ? YOLO_V11 : YOLO_V8;
    if (npu.init(model_path, version) != 0) {
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
            // Fallback: V4L2 с MJPEG fourcc
            cout << "GStreamer failed, trying V4L2 MJPEG..." << endl;
            vid.open(cam, CAP_V4L2);
            if (vid.isOpened()) {
                vid.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M','J','P','G'));
                configure_camera(vid, cfg.width, cfg.height);
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
    if (isdigit(*source)) {
        int fourcc = (int)vid.get(CAP_PROP_FOURCC);
        double reported_fps = vid.get(CAP_PROP_FPS);
        int buf_size = (int)vid.get(CAP_PROP_BUFFERSIZE);
        cout << "Camera fourcc: " << fourcc_to_string(fourcc)
             << "  driver FPS: " << reported_fps
             << "  buffer: " << buf_size << endl;
    }

    // --- Инициализация ZMQ ---
    zmq::context_t zmq_ctx;
    zmq::socket_t sock(zmq_ctx, zmq::socket_type::pub);
    sock.set(zmq::sockopt::sndbuf, LETTERBOX_COLS * LETTERBOX_ROWS * 3 * 4);
    sock.bind(ZMQ_ADDR);
    cout << "ZMQ publisher bound to " << ZMQ_ADDR << endl;

    // --- Запуск конвейера ---
    std::thread capture_thread(capture_thread_func, std::ref(vid));
    std::thread preprocess_thread(preprocess_thread_func);
    std::thread inference_thread(inference_thread_func, std::ref(npu), std::ref(sock),
                                  cfg.debug_step, std::cref(cfg.class_filter));

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
