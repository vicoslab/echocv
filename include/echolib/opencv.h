#ifndef __ECHOCV_MESSAGES_H
#define __ECHOCV_MESSAGES_H

#include <opencv2/opencv.hpp>
#include <echolib/client.h>
#include <echolib/datatypes.h>

using namespace std;
using namespace echolib;
using namespace cv;

namespace echolib {

typedef struct CameraIntrinsics {
    int width;
    int height;
    Matx33f intrinsics;
    Mat distortion; 
} CameraIntrinsics;

typedef struct CameraExtrinsics {
    Header header;
    Matx33f rotation;
    Matx31f translation;
} CameraExtrinsics;

class Frame {
public:
    Frame(Header header = Header(), Mat image = Mat());
    ~Frame();

    Header header;
    Mat image;
};

class MatBuffer : public Buffer {
public:
    MatBuffer(const Mat& mat, function<void()> complete = NULL);

    virtual ~MatBuffer();

    virtual ssize_t get_length() const;

    virtual ssize_t copy_data(ssize_t position, uchar* buffer, ssize_t length) const;

private:

    const Mat mat;
    ssize_t mat_length;
    function<void()> complete;

};

template<> inline void read(MessageReader& reader, Mat& dst) {

    int cols = reader.read<ushort>();
    int rows = reader.read<ushort>(); 
    int type = reader.read<int>();

    dst.create(rows, cols, type);

    reader.copy_data(dst.data, dst.cols * dst.rows * dst.elemSize());

}

template<> void write(MessageWriter& writer, const Mat& src) {
    
    writer.write<ushort>(src.cols);
    writer.write<ushort>(src.rows);
    writer.write<int>(src.type());

    writer.write_buffer(src.data, src.cols * src.rows * src.elemSize());


}

template<typename T, int m, int n> void write(MessageWriter& writer, const Matx<T, m, n>& src) {
    
    assert(m == src.rows && n == src.cols);

    writer.write<ushort>(src.rows);
    writer.write<ushort>(src.cols);
    writer.write<int>(CV_MAKETYPE(DataType<T>::depth, 1));

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            writer.write<T>(src(i, j));
        }
    }

}

template<typename T, int m, int n> void read(MessageReader& reader, Matx<T, m, n>& dst) {
    
    ushort h = reader.read<ushort>();
    ushort w = reader.read<ushort>();
    int t = reader.read<int>();

    assert(m == h && n == w && t == CV_MAKETYPE(DataType<T>::depth, 1));

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            dst(i, j) = reader.read<T>();
        }
    }

}

template <> inline string get_type_identifier<CameraExtrinsics>() { return string("camera extrinsics"); }

template<> inline shared_ptr<Message> echolib::Message::pack<CameraExtrinsics>(const CameraExtrinsics &data)
{
    MessageWriter writer(12 * sizeof(float));

    write(writer, data.header);
    write(writer, data.rotation);
    write(writer, data.translation);

    return make_shared<BufferedMessage>(writer);
}

template<> inline shared_ptr<CameraExtrinsics> echolib::Message::unpack<CameraExtrinsics>(SharedMessage message)
{
    MessageReader reader(message);

    shared_ptr<CameraExtrinsics> result(new CameraExtrinsics());
    read(reader, result->header);
    read(reader, result->rotation);
    read(reader, result->translation);
    return result;
}

template <> inline string get_type_identifier<CameraIntrinsics>() { return string("camera intrinsics"); }

template<> inline shared_ptr<Message> echolib::Message::pack<CameraIntrinsics>(const CameraIntrinsics &data)
{
    MessageWriter writer(12 * sizeof(float));

    writer.write<int>(data.width);
    writer.write<int>(data.height);
    write(writer, data.intrinsics);
    write(writer, data.distortion);

    return make_shared<BufferedMessage>(writer);
}

template<> inline shared_ptr<CameraIntrinsics> echolib::Message::unpack<CameraIntrinsics>(SharedMessage message)
{
    MessageReader reader(message);

    shared_ptr<CameraIntrinsics> result(new CameraIntrinsics());
    result->width = reader.read<int>();
    result->height = reader.read<int>();
    read(reader, result->intrinsics);
    read(reader, result->distortion);
    return result;
}

template <> inline string get_type_identifier<Frame>() { return string("camera frame"); }


template<> inline shared_ptr<Message> echolib::Message::pack<Frame>(const Frame &data) {

    MessageWriter writer;
    write(writer, data.header);
    writer.write<ushort>(data.image.cols);
    writer.write<ushort>(data.image.rows);
    writer.write<int>(data.image.type());

    vector<SharedBuffer> buffers;
    buffers.push_back(make_shared<BufferedMessage>(writer));
    buffers.push_back(make_shared<MatBuffer>(data.image));

    return make_shared<MultiBufferMessage>(buffers);
}

template<> inline shared_ptr<Frame> echolib::Message::unpack<Frame>(SharedMessage message) {
    MessageReader reader(message);

    shared_ptr<Frame> result(new Frame());
    read(reader, result->header);
    read(reader, result->image);
    return result;
}

}

#endif