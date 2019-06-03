#include <unistd.h>

#include <iostream>
#include <fstream>
#include <memory>
#include <opencv2/opencv.hpp>

#include <echolib/client.h>
#include <echolib/datatypes.h>
#include <echolib/helpers.h>

#include <echolib/opencv.h>

using namespace std;
using namespace echolib;
using namespace cv;

CameraIntrinsics parameters;

#define READ_MATX(N, M) { Mat tmp; (N) >> tmp; (M) = tmp; }

int main(int argc, char** argv) {

    string filename(argv[1]);

    SharedClient client = echolib::connect(string(), "imageserver");

    Mat image = imread(filename);

    if (image.empty()) {
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
        parameters.intrinsics(0, 2) = (float)(image.cols) / 2;
        parameters.intrinsics(1, 2) = (float)(image.rows) / 2;
        parameters.intrinsics(2, 2) = 1;
        parameters.distortion = (Mat_<float>(1,5) << 0, 0, 0, 0, 0);
    }

    parameters.width = image.cols;
    parameters.height = image.rows;

    SharedTypedPublisher<Frame> frame_publisher = make_shared<TypedPublisher<Frame> >(client, "camera", 1);

    SubscriptionWatcher watcher(client, "camera");

    StaticPublisher<CameraIntrinsics> intrinsics_publisher = StaticPublisher<CameraIntrinsics>(client, "intrinsics", parameters);

    while (true) {
        
        if (watcher.get_subscribers() > 0) {
            Frame frame(Header("image"), image);

            frame_publisher->send(frame);

        }
        if (!echolib::wait(30)) break;
    }

    exit(0);
}
