#include "echolib/opencv.h"
#include "conversion.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>

#include <opencv2/core/core.hpp>

namespace py = pybind11;

namespace pybind11 {
namespace detail {

template <> class type_caster<Mat> {
    typedef Mat type;
public:
    bool load(py::handle src, bool) {
        //if (!src || src == Py_None || !PyArray_Check(src)) { return false; }
        NDArrayConverter cvt;
        value = cvt.toMat(src.ptr());
        return true;
    }
    static py::handle cast(const Mat &src, return_value_policy policy, py::handle parent) {
        py::gil_scoped_acquire gil;
        NDArrayConverter cvt;
        py::handle result(cvt.toNDArray(src));
        return result;
    }
    PYBIND11_TYPE_CASTER(cv::Mat, _("cv::Mat"));
};

}
}

class PyImageSubscriber : public ImageSubscriber {
public:
    PyImageSubscriber(SharedClient client, const string &alias, 
        function<void(Mat&)> callback) : ImageSubscriber(client, alias, std::bind(&PyImageSubscriber::internal_callback, 
            this, std::placeholders::_1)), callback(callback) {
    }

    virtual ~PyImageSubscriber() {
        unsubscribe();
    }

    using Subscriber::subscribe;
    using Subscriber::unsubscribe;

private:

    void internal_callback(cv::Mat& mat) {
        py::gil_scoped_acquire gil; // acquire GIL lock
        callback(mat);
    }

    function<void(Mat&)> callback;

};

void write_mat(MessageWriter& writer, const Mat& src) {

    writer.write<Mat>(src);

}

Mat read_mat(MessageReader& reader) {

    return reader.read<Mat>();

}

PYBIND11_PLUGIN(pyechocv) {
        py::module m("pyechocv", "OpenCV messaging support for Echo IPC library");

        py::class_<ImagePublisher, std::shared_ptr<ImagePublisher> >(m, "ImagePublisher")
        .def(py::init<SharedClient, string>())
        .def("send", (bool (ImagePublisher::*)(Mat)) &ImagePublisher::send, "Send an image")
        .def("getSubscribers", &ImagePublisher::get_subscribers, "Get the number of subscribers");

        py::class_<ImageSubscriber, PyImageSubscriber, std::shared_ptr<ImageSubscriber> >(m, "ImageSubscriber")
        .def(py::init<SharedClient, string, function<void(Mat)> >())
        .def("subscribe", [](PyImageSubscriber &a) {
            py::gil_scoped_release gil; // release GIL lock
            return a.subscribe();
        }, "Start receiving")
        .def("unsubscribe", [](PyImageSubscriber &a) {
            py::gil_scoped_release gil; // release GIL lock
            return a.unsubscribe();
        }, "Stop receiving");

        m.def("readMat", &read_mat, "Read numpy array from message");
        m.def("writeMat", &write_mat, "Write numpy array to message");
        return m.ptr();

}
