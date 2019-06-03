
#include <echolib/opencv.h>

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

void write_mat(MessageWriter& writer, const Mat& src) {

    write(writer, src);

}

Mat read_mat(MessageReader& reader) {

    Mat mat;

    read(reader, mat);

    return mat;

}

PYBIND11_MODULE(pyechocv, m) {

        m.def("readMat", &read_mat, "Read numpy array from message");
        m.def("writeMat", &write_mat, "Write numpy array to message");

}
