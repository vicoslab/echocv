#include <iostream>
#include <thread>
#include <string>
#include <iostream>
#include <fstream>
#include <cmath>
#include <type_traits>

#include <echolib/opencv.h>

using namespace std;
using namespace echolib;
using namespace cv;

namespace echolib {

MatBuffer::MatBuffer(const Mat& mat, function<void()> complete) : mat(mat), complete(complete) {
    mat_length =  mat.cols * mat.rows * mat.elemSize();
}

MatBuffer::~MatBuffer() {
    if (complete)
        complete();
}

size_t MatBuffer::get_length() const {
    return mat_length;
}

size_t MatBuffer::copy_data(size_t position, uchar* buffer, size_t length) const {
    length = min(length, mat_length - position);
    if (length < 1) return 0;

    memcpy(buffer, &(mat.data[position]), length);
    return length;
}

Frame::Frame(Header header, Mat image): header(header), image(image) {}

Frame::~Frame() {}

}
