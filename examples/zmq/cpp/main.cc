#include <cctype>
#include <ctime>
#include <iostream>
#include <zmq.hpp>
#include <opencv2/opencv.hpp>

#include "yolov5.h"
#include "image_utils.h"
#include "file_utils.h"
#include "image_drawing.h"


using namespace cv;
using namespace std;

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
        cout << "Before open camera: " << cam << endl;
        vid.open(cam);
        vid.set(CAP_PROP_FRAME_WIDTH, 640);
        vid.set(CAP_PROP_FRAME_HEIGHT, 480);
        vid.set(CAP_PROP_FPS, 30);
        cout << "After open camera" << endl;
    }
    else
    {
        cout << "Before open video: " << source << endl;
        vid.open(source);
        cout << "After open video" << endl;
    }

    if (!vid.isOpened())
    {
        cerr << "ERROR: Could not open camera" << endl;
        return 1;
    }

    Mat m, mat;
    vid >> m;
    if (m.empty())
    {
        cerr << "ERROR: Image is empty" << endl;
        return 2;
    }

    resize(m, mat, Size(640, 640), 0, 0, INTER_AREA);
    size_t mat_len = mat.total() * mat.elemSize();
    cout << "Type: " << mat.type() << endl;
    cout << "Size: " << mat_len << endl;

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
		if (ret != 0) { cout << "release_yolov5_model fail! ret = " << ret << endl; }
    }

    image_buffer_t src_image;
    memset(&src_image, 0, sizeof(image_buffer_t));


    zmq::context_t ctx;
    zmq::socket_t sock(ctx, zmq::socket_type::pub);
    sock.bind("tcp://localhost:5555");

    static clock_t start, end;
    static double t = 0;
    static int f = 0;
    static int cnt = 0;

    while(true)
    {
        start = clock();

        vid >> m; if (m.empty()) { cerr << "ERROR: Image is empty" << endl; break; }

        char text[256];

        resize(m, mat, Size(640, 640), 0, 0, INTER_AREA);
        mat_len = mat.total() * mat.elemSize();

        src_image.width = 640;
        src_image.height = 640;
        src_image.size = mat_len;
        src_image.virt_addr = mat.data;
        src_image.format = IMAGE_FORMAT_RGB888;

        object_detect_result_list od_results;

        ret = inference_yolov5_model(&rknn_app_ctx, &src_image, &od_results);
        if (ret != 0)
        {
            cerr << "init_yolov5_model fail! ret = " << ret << endl;
	    	break;
        }

        for (int i = 0; i < od_results.count; i++)
        {
            object_detect_result *det_result = &(od_results.results[i]);
//            printf("%s @ (%d %d %d %d) %.3f\n", coco_cls_to_name(det_result->cls_id),
//                   det_result->box.left, det_result->box.top,
//                   det_result->box.right, det_result->box.bottom,
//                   det_result->prop);
            int x1 = det_result->box.left;
            int y1 = det_result->box.top;
            int x2 = det_result->box.right;
            int y2 = det_result->box.bottom;

//            draw_rectangle(&src_image, x1, y1, x2 - x1, y2 - y1, COLOR_BLUE, 3);

//            sprintf(text, "%s %.1f%%", coco_cls_to_name(det_result->cls_id), det_result->prop * 100);
//            draw_text(&src_image, text, x1, y1 - 20, COLOR_RED, 10);
			sprintf(text, "%d %s@%d,%d,%d,%d@%.2f\n", cnt++, coco_cls_to_name(det_result->cls_id),
	     	   		det_result->box.left, det_result->box.top, det_result->box.right, det_result->box.bottom, det_result->prop);

	    	sock.send(zmq::buffer(text), zmq::send_flags::dontwait);
        }

        
        end = clock();
        double cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        t += cpu_time_used;
        f += 1;

        if (f % 30 == 0)
        {
            cout << "\t FPS: " << std::fixed << std::setw(11) << std::setprecision(6) << 30.0 / t << " time = " << t << endl;
            t = 0; f = 0;
        }
    }

    sock.close();
    vid.release();

    deinit_post_process();

    ret = release_yolov5_model(&rknn_app_ctx);
    if (ret != 0) { cout << "release_yolov5_model fail! ret = " << ret << endl; }

    if (src_image.virt_addr != NULL) { free(src_image.virt_addr); }

    return 0;
}
