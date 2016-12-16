#ifndef __FINGERSCAPE_MSGS_H
#define __FINGERSCAPE_MSGS_H

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
    Matx33f rotation;
    Matx31f translation;
} CameraExtrinsics;

template<> inline void MessageWriter::write<Mat>(const Mat& src) {

    write<ushort>(src.rows);
    write<ushort>(src.cols);
    write<int>(src.type());

    for (int i = 0; i < src.rows; i++) {

        for (int j = 0; j < src.cols; j++) {

            write<float>(src.at<float>(i, j));

        }

    }

}

template<> inline Mat MessageReader::read<Mat>() {

    ushort h = read<ushort>();
    ushort w = read<ushort>();
    int t = read<int>();

    Mat data(h, w, t);

    for (int i = 0; i < data.rows; i++) {

        for (int j = 0; j < data.cols; j++) {

            data.at<float>(i, j) = read<float>();

        }

    }

    return data;
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

    write(writer, data.rotation);
    write(writer, data.translation);

    return make_shared<BufferedMessage>(writer);
}

template<> inline shared_ptr<CameraExtrinsics> echolib::Message::unpack<CameraExtrinsics>(SharedMessage message)
{
    MessageReader reader(message);

    shared_ptr<CameraExtrinsics> result(new CameraExtrinsics());
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
    writer.write<Mat>(data.distortion);

    return make_shared<BufferedMessage>(writer);
}

template<> inline shared_ptr<CameraIntrinsics> echolib::Message::unpack<CameraIntrinsics>(SharedMessage message)
{
    MessageReader reader(message);

    shared_ptr<CameraIntrinsics> result(new CameraIntrinsics());
    result->width = reader.read<int>();
    result->height = reader.read<int>();
    read(reader, result->intrinsics);
    result->distortion = reader.read<Mat>();
    return result;
}

class ImageSubscriber : public ChunkedSubscriber, public std::enable_shared_from_this<ImageSubscriber> {
public:
    ImageSubscriber(SharedClient client, const string &alias, function<void(Mat&)> callback);

    virtual ~ImageSubscriber();
    
    virtual void on_message(SharedMessage message);

protected:

    virtual void on_ready();

private:

    function<void(Mat&)> callback;

};

class ImagePublisher : public ChunkedPublisher, public Watcher, public std::enable_shared_from_this<ImagePublisher> {
public:
    ImagePublisher(SharedClient client, const string &alias, int queue_size = 1);

    virtual ~ImagePublisher();
    
    bool send(Mat &mat);

    virtual void on_event(SharedDictionary message);

    int get_subscribers();

private:

	static void throttle_callback(shared_ptr<ImagePublisher> publisher);

	int pending_images;

    int subscribers;

    int queue_size;

};

typedef shared_ptr<ImageSubscriber> SharedImageSubscriber;
typedef shared_ptr<ImagePublisher> SharedImagePublisher;

}

#endif