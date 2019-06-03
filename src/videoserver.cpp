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

    SharedClient client = echolib::connect(string(), "videoserver");

    Mat image;

    VideoCapture video(filename);

    if (!video.isOpened()) {
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
        parameters.intrinsics(0, 2) = video.get(CV_CAP_PROP_FRAME_WIDTH) / 2;
        parameters.intrinsics(1, 2) = video.get(CV_CAP_PROP_FRAME_HEIGHT) / 2;
        parameters.intrinsics(2, 2) = 1;
        parameters.distortion = (Mat_<float>(1,5) << 0, 0, 0, 0, 0);
    }

    parameters.width = video.get(CV_CAP_PROP_FRAME_WIDTH);
    parameters.height = video.get(CV_CAP_PROP_FRAME_HEIGHT);

    SharedTypedPublisher<Frame> frame_publisher = make_shared<TypedPublisher<Frame> >(client, "camera", 1);

    SubscriptionWatcher watcher(client, "camera");

    StaticPublisher<CameraIntrinsics> intrinsics_publisher = StaticPublisher<CameraIntrinsics>(client, "intrinsics", parameters);

    while (true) {
        video >> image;
        if (image.empty()) {
            video.set(CV_CAP_PROP_POS_FRAMES, 0);
            continue;
        }

        std::chrono::system_clock::time_point a = std::chrono::system_clock::now();

        if (watcher.get_subscribers() > 0) {
            Frame frame(Header("video", a), image);

            frame_publisher->send(frame);
        }
        if (!echolib::wait(30)) break;
    }

    exit(0);
}
