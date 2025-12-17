#include <cctype>
#include <ctime>
#include <iostream>
#include <string>
#include <zmq.hpp>
#include <opencv2/opencv.hpp>

#include <chrono>
#include <iomanip>

#include "yolov5.h"
#include "image_utils.h"
#include "file_utils.h"
#include "image_drawing.h"

using namespace cv;
using namespace std;

#define SRC_WIDTH 960
#define SRC_HEIGHT 720
#define FPS 30

#define WIDTH 640
#define HEIGHT 640

static const unsigned char base64_table[65] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/**
 * base64_encode - Base64 encode
 * @src: Data to be encoded
 * @len: Length of the data to be encoded
 * @out_len: Pointer to output length variable, or %NULL if not used
 * Returns: Allocated buffer of out_len bytes of encoded data,
 * or empty string on failure
 */
std::string base64_encode(const unsigned char *src, size_t len)
{
    unsigned char *out, *pos;
    const unsigned char *end, *in;

    size_t olen;

    olen = 4 * ((len + 2) / 3); /* 3-byte blocks to 4-byte */

    if (olen < len)
        return std::string(); /* integer overflow */

    std::string outStr;
    outStr.resize(olen);
    out = (unsigned char *)&outStr[0];

    end = src + len;
    in = src;
    pos = out;
    while (end - in >= 3)
    {
        *pos++ = base64_table[in[0] >> 2];
        *pos++ = base64_table[((in[0] & 0x03) << 4) | (in[1] >> 4)];
        *pos++ = base64_table[((in[1] & 0x0f) << 2) | (in[2] >> 6)];
        *pos++ = base64_table[in[2] & 0x3f];
        in += 3;
    }

    if (end - in)
    {
        *pos++ = base64_table[in[0] >> 2];
        if (end - in == 1)
        {
            *pos++ = base64_table[(in[0] & 0x03) << 4];
            *pos++ = '=';
        }
        else
        {
            *pos++ = base64_table[((in[0] & 0x03) << 4) |
                                  (in[1] >> 4)];
            *pos++ = base64_table[(in[1] & 0x0f) << 2];
        }
        *pos++ = '=';
    }

    return outStr;
}

bool mat_to_image_buffer(const cv::Mat &mat, image_buffer_t &out_buf, image_format_t fmt)
{
    if (!mat.data || mat.empty())
    {
        return false;
    }

    if (!mat.isContinuous())
    {
        // Неоднородные данные не могут передаваться напрямую
        return false;
    }

    out_buf.width = mat.cols;
    out_buf.height = mat.rows;
    out_buf.width_stride = static_cast<int>(mat.step); // Байтов на строку
    out_buf.height_stride = mat.rows;
    out_buf.format = fmt;
    out_buf.virt_addr = mat.data;
    out_buf.size = mat.total() * mat.elemSize();
    out_buf.fd = -1; // Если есть fd, его можно назначить отдельно

    return true;
}

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        cout << argv[0] << " <model_path> <source>" << endl;
        return -1;
    }

    const char *model_path = argv[1];
    const char *source = argv[2];

    VideoCapture vid;

    if (isdigit(*source))
    {
        int cam = static_cast<int>(strtol(source, nullptr, 10));

        stringstream pipeline_builder;
        pipeline_builder << "v4l2src device=/dev/video" << cam
                 << " ! image/jpeg, width=" << SRC_WIDTH
                 << ", height=" << SRC_HEIGHT
                 << ", framerate=" << FPS << "/1"
                 << " ! jpegdec ! videoconvert"
                 << " ! video/x-raw, format=BGR"
                 << " ! appsink drop=true sync=false";
        string pipeline = pipeline_builder.str();

        cout << "Try open USB camera <" << cam << "> with custom GStreamer pipeline:\n\t" << pipeline << endl;

        vid.open(pipeline, CAP_GSTREAMER);

        if (!vid.isOpened())
        {
            cout << "Can't open USB camera! Try open CSI camera <" << cam << ">" << endl;
            vid.open(cam);

            if (vid.isOpened())
            {
                vid.set(CAP_PROP_FRAME_WIDTH, SRC_WIDTH);
                vid.set(CAP_PROP_FRAME_HEIGHT, SRC_HEIGHT);
                vid.set(CAP_PROP_FPS, FPS);
                cout << "CSI camera opened.";
            }
            else
            {
                cout << "Can't open CSI camera!" << endl;
                return 1;
            }
        }
        else
        {
            cout << "USB camera opened.";
        }
    }
    else
    {
        cout << "Before open video: " << source << endl;
        vid.open(source);
        cout << "After open video" << endl;

        if (!vid.isOpened())
        {
            cerr << "ERROR: Could not open video" << endl;
            return 1;
        }
        else
        {
            cout << "Video opened.";
        }
    }

    Mat m;
    vid >> m;
    if (m.empty())
    {
        cerr << "ERROR: Image is empty" << endl;
        return 2;
    }

    int ret;
    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

    init_post_process();

    ret = init_yolov5_model(model_path, &rknn_app_ctx);
    if (ret != 0)
    {
        cerr << "init_yolov5_model fail! ret = " << ret << " model_path = " << model_path << endl;
        vid.release();

        deinit_post_process();

        ret = release_yolov5_model(&rknn_app_ctx);
        if (ret != 0)
        {
            cout << "release_yolov5_model fail! ret = " << ret << endl;
        }
    }

    image_buffer_t src_image, dst_image;

    memset(&dst_image, 0, sizeof(image_buffer_t));
    dst_image.width = WIDTH;
    dst_image.height = HEIGHT;
    dst_image.format = IMAGE_FORMAT_RGB888;
    dst_image.size = get_image_size(&dst_image);
    dst_image.virt_addr = (unsigned char *)malloc(dst_image.size);
    if (dst_image.virt_addr == NULL)
    {
        printf("malloc buffer size:%d fail!\n", dst_image.size);
        return -1;
    }

    zmq::context_t ctx;
    zmq::socket_t sock(ctx, zmq::socket_type::pub);
    // sock.setsockopt(ZMQ_SNDBUF, WIDTH * HEIGHT * 8 * 3 * 2);
    sock.set(zmq::sockopt::sndbuf, WIDTH * HEIGHT * 8 * 3 * 2);
    sock.bind("tcp://127.0.0.1:5757");

    double total_processing_time = 0;
    int frame_counter = 0;

    while (true)
    {
	auto start = std::chrono::steady_clock::now();

        vid >> m;
        if (m.empty())
        {
            cerr << "ERROR: Image is empty" << endl;
            break;
        }

        string msg;

        memset(&src_image, 0, sizeof(image_buffer_t));
        if (!mat_to_image_buffer(m, src_image, IMAGE_FORMAT_RGB888))
        {
            cerr << "Mat to image buffer failed" << endl;
            continue;
        }

        convert_image(&src_image, &dst_image, NULL, NULL, 0);

        object_detect_result_list od_results;

        ret = inference_yolov5_model(&rknn_app_ctx, &dst_image, &od_results);
        if (ret != 0)
        {
            cerr << "init_yolov5_model fail! ret = " << ret << endl;
            break;
        }

        for (int i = 0; i < od_results.count; i++)
        {
            char text[256];
            object_detect_result *det_result = &(od_results.results[i]);
            //            printf("%s @ (%d %d %d %d) %.3f\n", coco_cls_to_name(det_result->cls_id),
            //                   det_result->box.left, det_result->box.top,
            //                   det_result->box.right, det_result->box.bottom,
            //                   det_result->prop);
            int x1 = det_result->box.left;
            int y1 = det_result->box.top;
            int x2 = det_result->box.right;
            int y2 = det_result->box.bottom;

            // draw_rectangle(&src_image, x1, y1, x2 - x1, y2 - y1, COLOR_BLUE, 3);
            // sprintf(text, "%s %.1f%%", coco_cls_to_name(det_result->cls_id), det_result->prop * 100);
            // draw_text(&src_image, text, x1, y1 - 20, COLOR_RED, 10);

            sprintf(text, "%s@%d,%d,%d,%d@%.2f;", coco_cls_to_name(det_result->cls_id),
                    det_result->box.left, det_result->box.top, det_result->box.right, det_result->box.bottom, det_result->prop);

            msg += text;
        }

        if (msg.length() == 0)
            msg = "empty";

        sock.send(zmq::buffer(msg), zmq::send_flags::sndmore);
        sock.send(zmq::buffer(dst_image.virt_addr, dst_image.size), zmq::send_flags::none);

	auto end = std::chrono::steady_clock::now();
	std::chrono::duration<double> diff = end - start;

	total_processing_time += diff.count();
	frame_counter++;

        if (frame_counter % 30 == 0)
        {
	    double avg_fps = frame_counter / total_processing_time;

	    cout << "\t FPS: " << std::fixed << std::setw(11) << std::setprecision(6) << avg_fps << " time: " << total_processing_time << std::endl;

	    total_processing_time = 0;
	    frame_counter = 0;
        }
    }

    if (dst_image.virt_addr != NULL)
    {
        free(dst_image.virt_addr);
    }

    sock.close();
    vid.release();

    deinit_post_process();

    ret = release_yolov5_model(&rknn_app_ctx);
    if (ret != 0)
    {
        cout << "release_yolov5_model fail! ret = " << ret << endl;
    }

    if (src_image.virt_addr != NULL)
    {
        free(src_image.virt_addr);
    }

    return 0;
}
