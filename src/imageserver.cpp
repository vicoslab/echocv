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

#define READ_MATX(N, M) { Mat tmp; (N) >> tmp; (M) = tmp; }

int main(int argc, char** argv) {

    string filename(argv[1]);

    SharedClient client = echolib::connect();

    Mat frame = imread(filename);

    if (frame.empty()) {
        cerr << "Cannot open image file " << filename << endl;
        return -1;
    }

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
        parameters.distortion = (Mat_<double>(1,5) << 0, 0, 0, 0, 0);
    }

    parameters.width = frame.cols;
    parameters.height = frame.rows;

    SharedImagePublisher image_publisher = make_shared<ImagePublisher>(client, "camera");

    StaticPublisher<CameraIntrinsics> intrinsics_publisher = StaticPublisher<CameraIntrinsics>(client, "intrinsics", parameters);

    while (true) {
        if (image_publisher->get_subscribers() > 0) {
            image_publisher->send(frame);
        }
        if (!echolib::wait(30)) break;
    }

    exit(0);
}
