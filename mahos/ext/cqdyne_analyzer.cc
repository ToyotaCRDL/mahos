#include <iostream>
#include <chrono>

#include <pybind11/numpy.h>

namespace py = pybind11;

using std::chrono::seconds;
using std::chrono::system_clock;

uint64_t
search_head(py::array_t<uint64_t>& raw_data, uint64_t start_idx, uint64_t head)
{
    for (int i = start_idx; i < raw_data.size(); i++) {
        if (head <= raw_data.at(i)) {
            return i;
        }
    }
    return raw_data.size();
}

uint64_t
search_tail(py::array_t<uint64_t>& raw_data, uint64_t start_idx, uint64_t tail)
{
    for (int i = start_idx; i < raw_data.size(); i++) {
        if (tail < raw_data.at(i)) {
            return i;
        }
    }
    return start_idx;
}

void
analyze(py::array_t<uint64_t> raw_data,
        py::array_t<uint64_t> xdata,
        py::array_t<uint64_t> data,
        uint64_t signal_head,
        uint64_t signal_tail,
        int print_each)
{
    uint64_t idx = 0, head_idx = 0, tail_idx = 0;
    system_clock::time_point prev = system_clock::now();
    for (int i = 0; i < xdata.size(); i++) {
        uint64_t t = xdata.at(i);
        uint64_t head = t + signal_head;
        uint64_t tail = t + signal_tail;

        head_idx = search_head(raw_data, idx, head);
        if (head_idx == static_cast<uint64_t>(raw_data.size())) {
            std::cout << "[WARN] head index got out of bounds" << std::endl;
            return;
        }
        tail_idx = search_tail(raw_data, head_idx, tail);

        /* std::cout << "head " << head << " tail " << tail << std::endl; */
        /* std::cout << "hi " << head_idx << " ti " << tail_idx << std::endl; */

        data.mutable_at(i) = tail_idx - head_idx;
        idx = tail_idx;

        if (i > 0 && i % print_each == 0) {
            system_clock::time_point now = system_clock::now();
            double r = static_cast<double>(i) / xdata.size() * 100.0;
            std::cout << i << "/" << xdata.size() << "( " << r << "%) : "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(now - prev).count()
                      << " ms" << std::endl;
            prev = now;
        }
    }
}

PYBIND11_MODULE(cqdyne_analyzer, m)
{
    m.doc() = "qdyne analyzer functions";
    m.def("analyze",
          &analyze,
          "analyze qdyne raw data",
          py::arg("raw_data"),
          py::arg("xdata"),
          py::arg("data"),
          py::arg("signal_head"),
          py::arg("signal_tail"),
          py::arg("print_each") = 10000000);
}
