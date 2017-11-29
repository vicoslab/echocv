#include <unistd.h>

#include <iostream>
#include <fstream>
#include <memory>
#include <chrono>
#include <opencv2/opencv.hpp>

#include <echolib/client.h>
#include <echolib/datatypes.h>
#include <echolib/helpers.h>
#include <echolib/loop.h>

#include "echolib/opencv.h"

using namespace std;
using namespace echolib;
using namespace cv;

CameraIntrinsics parameters;

#define READ_MATX(N, M) { Mat tmp; (N) >> tmp; (M) = tmp; }

int main(int argc, char** argv) {

    int cameraid = (argc < 2 ? 0 : atoi(argv[1]));

    SharedClient client = echolib::connect();

    VideoCapture device;
    Mat frame;

    device.open(cameraid);

    if (!device.isOpened()) {
        cerr << "Cannot open camera device " << cameraid << endl;
        return -1;
    }

    device >> frame;

    FileStorage fsc("calibration.xml", FileStorage::READ);
    if (fsc.isOpened()) {
        READ_MATX(fsc["intrinsic"], parameters.intrinsics);
        fsc["distortion"] >> parameters.distortion;
    } else {
        parameters.intrinsics(0, 0) = 700;
        parameters.intrinsics(1, 1) = 700;
        parameters.intrinsics(0, 2) = (float)(frame.cols) / 2;
        parameters.intrinsics(1, 2) = (float)(frame.rows) / 2;
        parameters.intrinsics(2, 2) = 1;
        parameters.distortion = (Mat_<float>(1,5) << 0, 0, 0, 0, 0);
    }

    parameters.width = frame.cols;
    parameters.height = frame.rows;

    SharedImagePublisher image_publisher = make_shared<ImagePublisher>(client, "camera");

    StaticPublisher<CameraIntrinsics> intrinsics_publisher = StaticPublisher<CameraIntrinsics>(client, "intrinsics", parameters);

    double fps = 30;
    if (getenv("LIMIT_FPS")) {
        fps = min(1000.0, max(0.1, atof(getenv("LIMIT_FPS"))));
    }

    std::chrono::system_clock::time_point a = std::chrono::system_clock::now();
    std::chrono::system_clock::time_point b = std::chrono::system_clock::now();

    while (true) {
        device >> frame;

        a = std::chrono::system_clock::now();
        std::chrono::duration<double, std::milli> work_time = a - b;

        b = a;

        if (frame.empty()) return -1;

        if (image_publisher->get_subscribers() > 0) {
            image_publisher->send(frame);
        }

        std::chrono::duration<double, std::milli> delta_ms(max(0.0, 1000.0 / fps - (double)work_time.count()));
        
        auto delta_ms_duration = std::chrono::duration_cast<std::chrono::milliseconds>(delta_ms);

        if (!echolib::wait(max(10, (int)delta_ms_duration.count()))) break;
    }

    exit(0);
}
