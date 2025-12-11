#include <array>
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "catalog.hpp"

namespace py = pybind11;

PYBIND11_MODULE(pm_core, m) {
  m.doc() = "PolyMine core (mmap-backed)";

  py::class_<Catalog>(m, "Catalog")
    .def_static("mmap_all", &Catalog::mmap_all,
                py::arg("meta_bin"), py::arg("ids_bin"), py::arg("cif_blob"))

    .def_property_readonly("n", [](const Catalog& c){ return c.n; })

    .def("id_at", [](const Catalog& c, uint64_t i){
        auto sv = c.id_at(i);
        return py::str(sv.data(), sv.size());
    })

    // Return compressed CIF frame as Python bytes; user can zstd-decompress in Python
    .def("get_cif_bytes", [](const Catalog& c, uint64_t row){
        if (row >= c.n) throw std::out_of_range("row");
        uint64_t off = c.cif_off[row];
        uint64_t ln  = c.cif_len[row];
        if (off + ln > c.cif_blob_size) throw std::runtime_error("cif.blob bounds");
        const char* p = reinterpret_cast<const char*>(c.cif_blob + off);
        return py::bytes(p, ln);
    }, py::arg("row"))

    // ---- zero-copy numpy views ----
    .def_property_readonly("sgnum", [](const Catalog& c){
        return py::array_t<uint16_t>(c.n, c.sgnum, py::none());
    })
    .def_property_readonly("density", [](const Catalog& c){
        return py::array_t<float>(c.n, c.density, py::none());
    })
    .def_property_readonly("flags", [](const Catalog& c){
        return py::array_t<uint8_t>(c.n, c.flags, py::none());
    })
    .def_property_readonly("elem_lo", [](const Catalog& c){
        return py::array_t<uint64_t>(c.n, c.elem_lo, py::none());
    })
    .def_property_readonly("elem_hi", [](const Catalog& c){
        return py::array_t<uint64_t>(c.n, c.elem_hi, py::none());
    })
    .def_property_readonly("cif_off", [](const Catalog& c){
        return py::array_t<uint64_t>(c.n, c.cif_off, py::none());
    })
    .def_property_readonly("cif_len", [](const Catalog& c){
        return py::array_t<uint64_t>(c.n, c.cif_len, py::none());
    })
    .def_property_readonly("cell", [](const Catalog& c){
        const py::ssize_t n = static_cast<py::ssize_t>(c.n);
        const std::array<py::ssize_t, 2> shape   = { n, 6 };
        const std::array<py::ssize_t, 2> strides = {
            static_cast<py::ssize_t>(6 * sizeof(float)),
            static_cast<py::ssize_t>(sizeof(float))
        };
        return py::array(py::dtype::of<float>(), shape, strides,
                         static_cast<const void*>(c.cell), py::none());
    });
}

