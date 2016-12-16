


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

Mat frame;

// set all mat values at given channel to given value
void set_channel(Mat &mat, unsigned int channel, unsigned char value)
{
    // make sure have enough channels
    if (mat.channels() < channel + 1)
        return;

    const int cols = mat.cols;
    const int step = mat.channels();
    const int rows = mat.rows;
    for (int y = 0; y < rows; y++) {
        // get pointer to the first byte to be changed in this row
        unsigned char *p_row = mat.ptr(y) + channel; 
        unsigned char *row_end = p_row + cols*step;
        for (; p_row != row_end; p_row += step)
            *p_row = value;
    }
}

void handle_frame(Mat& out) {

	imwrite("out.png", out);

	exit(0);

}

int main(int argc, char** argv) {

    IOLoop loop;

    SharedClient client = connect(loop);

    frame = Mat(100, 100, CV_8UC3);

	set_channel(frame, 0, 100);
	set_channel(frame, 1, 150);
	set_channel(frame, 2, 200);

    SharedImagePublisher image_publisher = make_shared<ImagePublisher>(client, "camera");
	ImageSubscriber sub(client, "camera", handle_frame);

	imwrite("in.png", frame);

    while (true) {
        if (!frame.empty() && image_publisher->get_subscribers() > 0) {
            image_publisher->send(frame);
        }
        if (!loop.wait(10)) break;
    }

    exit(0);
}
