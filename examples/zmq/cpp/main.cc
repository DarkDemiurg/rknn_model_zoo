#include <cctype>
#include <ctime>
#include <iostream>
#include <cstdlib>
#include <getopt.h>
#include <string>
#include <zmq.hpp>
#include <opencv2/opencv.hpp>

#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <chrono>
#include <iomanip>

#include "yolov5.h"
#include "image_utils.h"
#include "file_utils.h"
#include "image_drawing.h"

using namespace cv;
using namespace std;

#define DEFAULT_WIDTH  960
#define DEFAULT_HEIGHT 720

struct Resolution
{
    int width;
    int height;
};

// ============================================================================
// Bounded frame queue: drops oldest frame when full to minimize latency
// ============================================================================
template<typename T>
class BoundedQueue {
public:
    explicit BoundedQueue(size_t max_size) : max_size_(max_size), stopped_(false) {}

    void push(T &&item) {
        std::unique_lock<std::mutex> lock(mtx_);
        if (queue_.size() >= max_size_)
            queue_.pop();  // drop oldest frame
        queue_.push(std::move(item));
        cv_.notify_one();
    }

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

static BoundedQueue<cv::Mat> g_frame_queue(3);
static std::atomic<bool> g_running(true);

// Capture thread: only grabs frames, puts into queue
static void capture_thread_func(VideoCapture &vid)
{
    while (g_running) {
        cv::Mat frame;
        vid >> frame;
        if (frame.empty()) {
            cerr << "[Capture] Empty frame, stopping" << endl;
            g_running = false;
            break;
        }
        g_frame_queue.push(std::move(frame));
    }
    g_frame_queue.stop();
}

bool mat_to_image_buffer(const cv::Mat &mat, image_buffer_t &out_buf, image_format_t fmt)
{
    if (!mat.data || mat.empty())
        return false;
    if (!mat.isContinuous())
        return false;

    out_buf.width = mat.cols;
    out_buf.height = mat.rows;
    out_buf.width_stride = static_cast<int>(mat.step);
    out_buf.height_stride = mat.rows;
    out_buf.format = fmt;
    out_buf.virt_addr = mat.data;
    out_buf.size = mat.total() * mat.elemSize();
    out_buf.fd = -1;
    return true;
}

void print_usage(const char *program_name)
{
    cout << "Usage: " << program_name << " <model_path> <source> [-w WIDTH] [-h HEIGHT]" << endl;
}

Resolution parseArguments(int argc, char *argv[])
{
    Resolution res = {DEFAULT_WIDTH, DEFAULT_HEIGHT};
    int opt;
    while ((opt = getopt(argc, argv, "w:h:")) != -1) {
        switch (opt) {
            case 'w':
                res.width = atoi(optarg);
                if (res.width <= 0) {
                    cerr << "Warning: WIDTH must be > 0. Using default." << endl;
                    res.width = DEFAULT_WIDTH;
                }
                break;
            case 'h':
                res.height = atoi(optarg);
                if (res.height <= 0) {
                    cerr << "Warning: HEIGHT must be > 0. Using default." << endl;
                    res.height = DEFAULT_HEIGHT;
                }
                break;
            case '?':
                print_usage(argv[0]);
                break;
        }
    }
    return res;
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        print_usage(argv[0]);
        return -1;
    }

    const char *model_path = argv[1];
    const char *source = argv[2];

    Resolution resolution = parseArguments(argc, argv);
    cout << "Camera resolution: [" << resolution.width << "x" << resolution.height << "]" << endl;

    VideoCapture vid;

    if (isdigit(*source)) {
        int cam = static_cast<int>(strtol(source, nullptr, 10));

        stringstream pipeline_builder;
        pipeline_builder << "v4l2src device=/dev/video" << cam
                         << " ! image/jpeg, width=" << resolution.width
                         << ", height=" << resolution.height
                         << " ! jpegdec ! videoconvert"
                         << " ! video/x-raw, format=BGR"
                         << " ! appsink drop=true sync=false";
        string pipeline = pipeline_builder.str();

        cout << "Try open USB camera <" << cam << "> with GStreamer pipeline:\n\t" << pipeline << endl;
        vid.open(pipeline, CAP_GSTREAMER);

        if (!vid.isOpened()) {
            cout << "GStreamer failed, trying direct open <" << cam << ">" << endl;
            vid.open(cam);
            if (vid.isOpened()) {
                vid.set(CAP_PROP_FRAME_WIDTH, resolution.width);
                vid.set(CAP_PROP_FRAME_HEIGHT, resolution.height);
                cout << "Camera opened." << endl;
            } else {
                cerr << "Can't open camera!" << endl;
                return 1;
            }
        } else {
            cout << "USB camera opened." << endl;
        }
    } else {
        cout << "Opening video: " << source << endl;
        vid.open(source);
        if (!vid.isOpened()) {
            cerr << "ERROR: Could not open video" << endl;
            return 1;
        }
        cout << "Video opened." << endl;
    }

    // Check first frame
    Mat m;
    vid >> m;
    if (m.empty()) {
        cerr << "ERROR: First frame is empty" << endl;
        return 2;
    }

    int ret;
    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

    init_post_process();

    ret = init_yolov5_model(model_path, &rknn_app_ctx);
    if (ret != 0) {
        cerr << "init_yolov5_model fail! ret=" << ret << endl;
        vid.release();
        deinit_post_process();
        release_yolov5_model(&rknn_app_ctx);
        return -1;
    }

    zmq::context_t zmq_ctx;
    zmq::socket_t sock(zmq_ctx, zmq::socket_type::pub);
    sock.set(zmq::sockopt::sndbuf, rknn_app_ctx.model_width * rknn_app_ctx.model_height * 3 * 4);
    sock.set(zmq::sockopt::sndhwm, 2);
    sock.bind("tcp://127.0.0.1:5757");

    // Start capture thread
    std::thread cap_thread(capture_thread_func, std::ref(vid));

    double total_processing_time = 0;
    int frame_counter = 0;

    while (g_running) {
        auto start = std::chrono::steady_clock::now();

        cv::Mat frame;
        if (!g_frame_queue.pop(frame))
            break;

        image_buffer_t src_image;
        memset(&src_image, 0, sizeof(image_buffer_t));

        // Pass original frame directly — inference does letterbox internally (no double resize)
        if (!mat_to_image_buffer(frame, src_image, IMAGE_FORMAT_RGB888)) {
            cerr << "mat_to_image_buffer failed" << endl;
            continue;
        }

        object_detect_result_list od_results;
        ret = inference_yolov5_model(&rknn_app_ctx, &src_image, &od_results);
        if (ret != 0) {
            cerr << "inference_yolov5_model fail! ret=" << ret << endl;
            break;
        }

        string msg;
        for (int i = 0; i < od_results.count; i++) {
            char text[256];
            object_detect_result *det_result = &(od_results.results[i]);
            sprintf(text, "%s@%d,%d,%d,%d@%.2f;",
                    coco_cls_to_name(det_result->cls_id),
                    det_result->box.left, det_result->box.top,
                    det_result->box.right, det_result->box.bottom,
                    det_result->prop);
            msg += text;
        }
        if (msg.empty())
            msg = "empty";

        // Send detections text + pre-processed 640x640 image
        sock.send(zmq::buffer(msg), zmq::send_flags::sndmore);
        sock.send(zmq::buffer(rknn_app_ctx.input_buf, rknn_app_ctx.input_buf_size),
                  zmq::send_flags::none);

        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> diff = end - start;
        total_processing_time += diff.count();
        frame_counter++;

        if (frame_counter % 30 == 0) {
            double avg_fps = frame_counter / total_processing_time;
            cout << "\t FPS: " << std::fixed << std::setw(7) << std::setprecision(2) << avg_fps
                 << "  avg_ms: " << std::setprecision(1) << (total_processing_time / frame_counter * 1000)
                 << endl;
            total_processing_time = 0;
            frame_counter = 0;
        }
    }

    g_running = false;
    g_frame_queue.stop();
    cap_thread.join();

    sock.close();
    vid.release();

    deinit_post_process();

    ret = release_yolov5_model(&rknn_app_ctx);
    if (ret != 0)
        cerr << "release_yolov5_model fail! ret=" << ret << endl;

    return 0;
}
