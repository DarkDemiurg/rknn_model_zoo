#include <cctype>
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

#include "yolov8.h"
#include "rknn_api.h"
#include "image_utils.h"
#include "file_utils.h"
#include "image_drawing.h"

using namespace cv;
using namespace std;

#define DEFAULT_WIDTH  960
#define DEFAULT_HEIGHT 720

// Model input size — must match the .rknn model
#define MODEL_WIDTH  640
#define MODEL_HEIGHT 640

struct Resolution { int width; int height; };

// ============================================================================
// Bounded frame queue: holds pre-scaled 640x640 frames ready for NPU.
// Drops oldest when full to minimize latency.
// ============================================================================
template<typename T>
class BoundedQueue {
public:
    explicit BoundedQueue(size_t max_size) : max_size_(max_size), stopped_(false) {}

    void push(T &&item) {
        std::unique_lock<std::mutex> lock(mtx_);
        if (queue_.size() >= max_size_)
            queue_.pop();
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

// Frame ready for NPU: already letterboxed to 640x640 RGB
struct PreparedFrame {
    cv::Mat scaled; // 640x640, RGB
};

static BoundedQueue<PreparedFrame> g_frame_queue(3);
static std::atomic<bool> g_running(true);

// ============================================================================
// Capture + preprocess thread:
//   1. Grab frame from camera (BGR)
//   2. Letterbox resize to 640x640 via OpenCV (NEON-accelerated, no RGA overhead)
//   3. Convert BGR -> RGB
//   4. Push to queue
//
// Doing letterbox here in a separate thread hides its cost behind NPU inference.
// OpenCV resize uses NEON intrinsics and is much faster than the RGA
// virtualaddr path (which does mmap/munmap per call).
// ============================================================================
static void capture_thread_func(VideoCapture &vid)
{
    const int W = MODEL_WIDTH;
    const int H = MODEL_HEIGHT;

    while (g_running) {
        cv::Mat frame;
        vid >> frame;
        if (frame.empty()) {
            cerr << "[Capture] Empty frame, stopping" << endl;
            g_running = false;
            break;
        }

        // Letterbox: scale keeping aspect ratio, pad with 114 (grey)
        int fw = frame.cols, fh = frame.rows;
        float scale = std::min((float)W / fw, (float)H / fh);
        int rw = (int)(fw * scale);
        int rh = (int)(fh * scale);
        int pad_x = (W - rw) / 2;
        int pad_y = (H - rh) / 2;

        cv::Mat resized;
        cv::resize(frame, resized, cv::Size(rw, rh), 0, 0, cv::INTER_LINEAR);

        cv::Mat canvas(H, W, CV_8UC3, cv::Scalar(114, 114, 114));
        resized.copyTo(canvas(cv::Rect(pad_x, pad_y, rw, rh)));

        // BGR -> RGB
        cv::Mat rgb;
        cv::cvtColor(canvas, rgb, cv::COLOR_BGR2RGB);

        PreparedFrame pf;
        pf.scaled = std::move(rgb);
        g_frame_queue.push(std::move(pf));
    }
    g_frame_queue.stop();
}

void print_usage(const char *prog)
{
    cout << "Usage: " << prog << " <model_path> <source> [-w WIDTH] [-h HEIGHT]" << endl;
}

Resolution parseArguments(int argc, char *argv[])
{
    Resolution res = {DEFAULT_WIDTH, DEFAULT_HEIGHT};
    int opt;
    while ((opt = getopt(argc, argv, "w:h:")) != -1) {
        switch (opt) {
            case 'w': res.width  = atoi(optarg) > 0 ? atoi(optarg) : DEFAULT_WIDTH;  break;
            case 'h': res.height = atoi(optarg) > 0 ? atoi(optarg) : DEFAULT_HEIGHT; break;
        }
    }
    return res;
}

int main(int argc, char **argv)
{
    if (argc < 3) { print_usage(argv[0]); return -1; }

    const char *model_path = argv[1];
    const char *source     = argv[2];

    Resolution res = parseArguments(argc, argv);
    cout << "Camera resolution: [" << res.width << "x" << res.height << "]" << endl;

    VideoCapture vid;
    if (isdigit(*source)) {
        int cam = (int)strtol(source, nullptr, 10);
        stringstream ss;
        ss << "v4l2src device=/dev/video" << cam
           << " ! image/jpeg, width=" << res.width << ", height=" << res.height
           << " ! jpegdec ! videoconvert"
           << " ! video/x-raw, format=BGR"
           << " ! appsink drop=true sync=false";
        cout << "GStreamer pipeline:\n\t" << ss.str() << endl;
        vid.open(ss.str(), CAP_GSTREAMER);
        if (!vid.isOpened()) {
            cout << "GStreamer failed, trying direct V4L2" << endl;
            vid.open(cam);
            if (!vid.isOpened()) { cerr << "Can't open camera!" << endl; return 1; }
            vid.set(CAP_PROP_FRAME_WIDTH, res.width);
            vid.set(CAP_PROP_FRAME_HEIGHT, res.height);
        }
    } else {
        vid.open(source);
        if (!vid.isOpened()) { cerr << "Can't open video: " << source << endl; return 1; }
    }
    cout << "Source opened." << endl;

    Mat probe; vid >> probe;
    if (probe.empty()) { cerr << "First frame empty" << endl; return 2; }
    // Store actual camera frame dimensions for letterbox calculation
    const int cam_w = probe.cols;
    const int cam_h = probe.rows;
    cout << "First frame: " << cam_w << "x" << cam_h << endl;

    // Init model
    int ret;
    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));
    init_post_process();

    ret = init_yolov8_model(model_path, &rknn_app_ctx);
    if (ret != 0) {
        cerr << "init_yolov8_model fail! ret=" << ret << endl;
        deinit_post_process();
        return -1;
    }

    // Init ZMQ
    zmq::context_t zmq_ctx;
    zmq::socket_t sock(zmq_ctx, zmq::socket_type::pub);
    sock.set(zmq::sockopt::sndbuf, MODEL_WIDTH * MODEL_HEIGHT * 3 * 4);
    sock.set(zmq::sockopt::sndhwm, 2);
    sock.bind("tcp://127.0.0.1:5757");

    // Pre-compute letterbox params (constant for a fixed camera resolution)
    const float lb_scale = std::min((float)MODEL_WIDTH / cam_w, (float)MODEL_HEIGHT / cam_h);
    letterbox_t letter_box;
    letter_box.scale = lb_scale;
    letter_box.x_pad = (MODEL_WIDTH  - (int)(cam_w * lb_scale)) / 2;
    letter_box.y_pad = (MODEL_HEIGHT - (int)(cam_h * lb_scale)) / 2;

    // Start capture+preprocess thread
    std::thread cap_thread(capture_thread_func, std::ref(vid));

    double total_time = 0;
    int frame_counter = 0;

    while (g_running) {
        auto t0 = std::chrono::steady_clock::now();

        PreparedFrame pf;
        if (!g_frame_queue.pop(pf)) break;

        // Feed the pre-scaled RGB frame directly to RKNN —
        // no letterbox needed here, capture thread already did it.
        // We reuse the pre-allocated input_buf for the RKNN copy.
        size_t frame_bytes = pf.scaled.total() * pf.scaled.elemSize();
        if (!pf.scaled.isContinuous() || (int)frame_bytes != rknn_app_ctx.input_buf_size) {
            cerr << "Frame size mismatch: " << frame_bytes << " vs " << rknn_app_ctx.input_buf_size << endl;
            continue;
        }
        memcpy(rknn_app_ctx.input_buf, pf.scaled.data, rknn_app_ctx.input_buf_size);

        // Run inference directly on pre-allocated buffer (skip letterbox in inference_*)
        rknn_input inputs[1];
        memset(inputs, 0, sizeof(inputs));
        inputs[0].index = 0;
        inputs[0].type  = RKNN_TENSOR_UINT8;
        inputs[0].fmt   = RKNN_TENSOR_NHWC;
        inputs[0].size  = rknn_app_ctx.input_buf_size;
        inputs[0].buf   = rknn_app_ctx.input_buf;

        ret = rknn_inputs_set(rknn_app_ctx.rknn_ctx, rknn_app_ctx.io_num.n_input, inputs);
        if (ret < 0) { cerr << "rknn_inputs_set fail ret=" << ret << endl; break; }

        ret = rknn_run(rknn_app_ctx.rknn_ctx, nullptr);
        if (ret < 0) { cerr << "rknn_run fail ret=" << ret << endl; break; }

        rknn_output outputs[rknn_app_ctx.io_num.n_output];
        memset(outputs, 0, sizeof(outputs));
        for (int i = 0; i < (int)rknn_app_ctx.io_num.n_output; i++) {
            outputs[i].index      = i;
            outputs[i].want_float = (!rknn_app_ctx.is_quant);
        }
        ret = rknn_outputs_get(rknn_app_ctx.rknn_ctx, rknn_app_ctx.io_num.n_output, outputs, NULL);
        if (ret < 0) { cerr << "rknn_outputs_get fail ret=" << ret << endl; break; }

        object_detect_result_list od_results;
        post_process(&rknn_app_ctx, outputs, &letter_box, BOX_THRESH, NMS_THRESH, &od_results);
        rknn_outputs_release(rknn_app_ctx.rknn_ctx, rknn_app_ctx.io_num.n_output, outputs);

        string msg;
        for (int i = 0; i < od_results.count; i++) {
            char text[256];
            object_detect_result *d = &od_results.results[i];
            sprintf(text, "%s@%d,%d,%d,%d@%.2f;",
                    coco_cls_to_name(d->cls_id),
                    d->box.left, d->box.top, d->box.right, d->box.bottom,
                    d->prop);
            msg += text;
        }
        if (msg.empty()) msg = "empty";

        sock.send(zmq::buffer(msg), zmq::send_flags::sndmore);
        sock.send(zmq::buffer(rknn_app_ctx.input_buf, rknn_app_ctx.input_buf_size),
                  zmq::send_flags::none);

        auto t1 = std::chrono::steady_clock::now();
        total_time += std::chrono::duration<double>(t1 - t0).count();
        frame_counter++;

        if (frame_counter % 30 == 0) {
            cout << "\t FPS: " << std::fixed << std::setprecision(2)
                 << (frame_counter / total_time)
                 << "  avg_ms: " << std::setprecision(1)
                 << (total_time / frame_counter * 1000) << endl;
            total_time   = 0;
            frame_counter = 0;
        }
    }

    g_running = false;
    g_frame_queue.stop();
    cap_thread.join();

    sock.close();
    vid.release();
    deinit_post_process();
    release_yolov8_model(&rknn_app_ctx);
    return 0;
}
