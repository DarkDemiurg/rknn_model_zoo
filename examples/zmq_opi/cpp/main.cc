#include <cctype>
#include <ctime>
#include <iostream>
#include <cstdlib>
#include <getopt.h>
#include <string>
#include <zmq.hpp>
#include <opencv2/opencv.hpp>

#include <chrono>
#include <iomanip>

#include "image_utils.h"
#include "file_utils.h"
#include "image_drawing.h"

#include "rkYolov5s.hpp"
#include "rknnPool.hpp"

using namespace cv;
using namespace std;

#define DEFAULT_WIDTH 960
#define DEFAULT_HEIGHT 720
#define FPS 90

struct Resolution 
{
    int width;
    int height;
};

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

void print_usage(const char* program_name)
{
    cout << "Usage: " << program_name << " <model_path> <source> [-w WIDTH] [-h HEIGHT]" << endl;
}

Resolution parseArguments(int argc, char* argv[])
{
    Resolution res = {DEFAULT_WIDTH, DEFAULT_HEIGHT};
    int opt;

    while ((opt = getopt(argc, argv, "w:h:")) != -1) 
    {
        switch (opt) 
        {
            case 'w':
                res.width = atoi(optarg);
                if (res.width <= 0) 
                {
                    cerr << "Warning: WIDTH must be greater than 0. Use default value." << endl;
                    res.width = DEFAULT_WIDTH;
                }
                break;
            case 'h':
                res.height = atoi(optarg);
                if (res.height <= 0) 
                {
                    cerr << "Warning: HEIGHT must be greater than 0. Use default value." << endl;
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
    if (argc < 3)
    {
        print_usage(argv[0]);
        return -1;
    }

    const char *model_path = argv[1];
    const char *source = argv[2];

    Resolution resolution = parseArguments(argc, argv);
    cout << "Camera resolution: [" << resolution.width << "x" << resolution.height << "]" << endl;

///////////////////////////////////////////////////////////////////////
  
    int threadNum = 3;

    rknnPool<rkYolov5s, cv::Mat, cv::Mat> testPool(model_path, threadNum);

    if (testPool.init() != 0)
    {
        cerr << "rknnPool init fail!" << endl;
        return -1;
    }

///////////////////////////////////////////////////////////////////////

    zmq::context_t ctx;
    zmq::socket_t sock(ctx, zmq::socket_type::pub);
    sock.set(zmq::sockopt::sndbuf, WIDTH * HEIGHT * 8 * 3 * 2);
    sock.bind("tcp://127.0.0.1:5757");

    VideoCapture vid;

    if (isdigit(*source))
    {
        int cam = static_cast<int>(strtol(source, nullptr, 10));

        stringstream pipeline_builder;
        pipeline_builder << "v4l2src device=/dev/video" << cam
                 << " ! image/jpeg, width=" << resolution.width
                 << ", height=" << resolution.height
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
                vid.set(CAP_PROP_FRAME_WIDTH, resolution.width);
                vid.set(CAP_PROP_FRAME_HEIGHT, resolution.height);
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

    Mat m, img;
    vid >> m;
    if (m.empty())
    {
        cerr << "ERROR: Image is empty" << endl;
        return 2;
    }

    double total_processing_time = 0;
    int frame_counter = 0;
    int frames = 0;

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

        if (testPool.put(m) != 0)
        {
            cerr << "Can't put to pool" << endl;
            break;
        }

        if (frames >= threadNum && testPool.get(img) != 0)
        {
            cerr << "Can't get from pool" << endl;
            break;
        }
/*
        if (!img.empty())
        {
            std::ostringstream filename;
            filename << "img_" << frames << ".png";
            cv::imwrite(filename.str(), img);
        }
*/
        if (msg.length() == 0)
            msg = "empty";

//        sock.send(zmq::buffer(msg), zmq::send_flags::sndmore);
//        sock.send(zmq::buffer(dst_image.virt_addr, dst_image.size), zmq::send_flags::none); // TODO: convert 

        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> diff = end - start;

        total_processing_time += diff.count();
        frame_counter++;
        frames++;

        if (frame_counter % 30 == 0)
        {
            double avg_fps = frame_counter / total_processing_time;

            cout << "\t FPS: " << std::fixed << std::setw(11) << std::setprecision(6) << avg_fps << " time: " << total_processing_time << std::endl;

            total_processing_time = 0;
            frame_counter = 0;
        }
    }

    while (true)
    {
        cv::Mat img;
        if (testPool.get(img) != 0)
            break;
    }

    sock.close();
    vid.release();

    return 0;
}
