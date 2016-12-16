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

ImageSubscriber::ImageSubscriber(SharedClient client, const string &alias, function<void(Mat&)> callback) : 
    ChunkedSubscriber(client, alias, string("opencv image")), callback(callback) {

}

ImageSubscriber::~ImageSubscriber() {

}

void ImageSubscriber::on_message(SharedMessage message) {

    try {

        MessageReader reader(message);
        int cols = reader.read<ushort>();
        int rows = reader.read<ushort>(); 
        int type = reader.read<int>();

        Mat data(rows, cols, type);

        reader.copy_data(data.data, data.cols * data.rows * data.elemSize());

        callback(data);

    } catch (echolib::ParseException &e) {
        Subscriber::on_error(e);
    }

}

void ImageSubscriber::on_ready() { 
 
}

class MatBuffer : public Buffer {
public:
    MatBuffer(Mat& mat, function<void()> complete = NULL) : mat(mat), complete(complete) {
        mat_length =  mat.cols * mat.rows * mat.elemSize();
    }
    virtual ~MatBuffer() {
        if (complete)
            complete();
    };

    virtual ssize_t get_length() const
    {
        return mat_length;
    }

    virtual ssize_t copy_data(ssize_t position, uchar* buffer, ssize_t length) const
    {
        length = min(length, mat_length - position);
        if (length < 1) return 0;

        memcpy(buffer, &(mat.data[position]), length);
        return length;
    }

private:

    Mat mat;
    ssize_t mat_length;
    function<void()> complete;
};


ImagePublisher::ImagePublisher(SharedClient client, const string &alias, int queue_size) : 
    ChunkedPublisher(client, alias, string("opencv image")), Watcher(client, alias),
    pending_images(0), queue_size(queue_size) {

}

ImagePublisher::~ImagePublisher() {

}

bool ImagePublisher::send(Mat &mat) {

    if (pending_images > queue_size + 1)
        return false;

    shared_ptr<MemoryBuffer> header = make_shared<MemoryBuffer>(sizeof(int) + 2 * sizeof(ushort));
    MessageWriter writer(header->get_buffer(), header->get_length());
    //writer.write_integer(mat.elemSize());
    writer.write<ushort>(mat.cols);
    writer.write<ushort>(mat.rows);
    writer.write<int>(mat.type());

    vector<SharedBuffer> buffers;
    buffers.push_back(header);
    buffers.push_back(make_shared<MatBuffer>(mat, bind(&ImagePublisher::throttle_callback, shared_from_this())));

    shared_ptr<Message> message = make_shared<MultiBufferMessage>(buffers);
    MessageHandler::set_channel(message, get_channel_id());

    pending_images++;

    return send_message_internal(message);

}

void ImagePublisher::throttle_callback(shared_ptr<ImagePublisher> publisher)
{
    // Called when a mat buffer is freed ... the image was sent by then
    publisher->pending_images--;
}

void ImagePublisher::on_event(SharedDictionary message)
{

    string type = message->get<string>("type", "");

    if (type == "subscribe" || type == "unsubscribe" || type == "summary") {
        subscribers = message->get<int>("subscribers", 0);

    }

}

int ImagePublisher::get_subscribers()
{
    return subscribers;
}

}
