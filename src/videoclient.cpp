#include <unistd.h>

#include <iostream>
#include <fstream>
#include <memory>
#include <ctime>
#include <chrono>
#include <opencv2/opencv.hpp>

#include <echolib/client.h>
#include <echolib/datatypes.h>
#include <echolib/helpers.h>

#include <echolib/opencv.h>

using namespace std;
using namespace echolib;
using namespace cv;

int main(int argc, char** argv) {

    SharedClient client = echolib::connect();

    shared_ptr<Frame> frame;
    bool incoming = false;

    bool headless = (NULL == getenv("DISPLAY"));

    function<void(shared_ptr<Frame>)> frame_callback = [&](shared_ptr<Frame> m) {

        frame = m;
        incoming = true;

    };

    SharedTypedSubscriber<Frame> frame_subscriber = make_shared<TypedSubscriber<Frame> >(client, "camera", frame_callback);

    while (true) {

        if (incoming) {

            std::time_t tt = std::chrono::system_clock::to_time_t ( frame->header.timestamp );

            

            if (!headless) {
                cv::putText(frame->image, ctime(&tt), Point(10, 50), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 0, 0), 3);
                imshow("EchoCV demo", frame->image);
            }
            else {
                cout << "Frame received, timestamp = " << ctime(&tt) << endl;
            }

            incoming = false;
        }

        if (!echolib::wait(20)) break;
        if (!waitKey(1) && !headless) break;
    }

    exit(0);
}
