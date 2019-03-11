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

int main(int argc, char** argv) {

    SharedClient client = echolib::connect();

    Mat frame;
    bool incoming = false;

    function<void(Mat&)> image_callback = [&](Mat& m) {

        frame = m;
        incoming = true;

    };

    SharedImageSubscriber image_subscriber = make_shared<ImageSubscriber>(client, "camera", image_callback);

    while (true) {

        if (incoming) {
            imshow("EchoCV demo", frame);
            incoming = false;
        }

        if (!echolib::wait(20)) break;
        if (!waitKey(1)) break;
    }

    exit(0);
}
