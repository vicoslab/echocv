#include <unistd.h>

#include <iostream>
#include <fstream>
#include <memory>
#include <opencv2/opencv.hpp>

#include <echolib/client.h>
#include <echolib/datatypes.h>
#include <echolib/helpers.h>

#include "echolib/opencv.h"

using namespace std;
using namespace echolib;
using namespace cv;

CameraIntrinsics parameters;
bool autoset_parameters = true;

#define READ_MATX(N, M) { Mat tmp; (N) >> tmp; (M) = tmp; }

int main(int argc, char** argv) {

    int cameraid = (argc < 2 ? 0 : atoi(argv[1]));

    IOLoop loop;
    SharedClient client = connect(loop);

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
        autoset_parameters = false;
    } else {
        parameters.intrinsics(0, 0) = 700;
        parameters.intrinsics(1, 1) = 700;
        parameters.intrinsics(0, 2) = (float)(frame.cols) / 2;
        parameters.intrinsics(1, 2) = (float)(frame.rows) / 2;
        parameters.distortion = (Mat_<double>(1,5) << 0, 0, 0, 0, 0);
        autoset_parameters = false;
    }

    parameters.width = frame.cols;
    parameters.height = frame.rows;

    SharedImagePublisher image_publisher = make_shared<ImagePublisher>(client, "camera");

    StaticPublisher<CameraIntrinsics> intrinsics_publisher = StaticPublisher<CameraIntrinsics>(client, "intrinsics", parameters);

    while (true) {
        device >> frame;

        if (frame.empty()) return -1;

        if (image_publisher->get_subscribers() > 0) {
            image_publisher->send(frame);
        }
        if (!loop.wait(10)) break;
    }

    exit(0);
}
