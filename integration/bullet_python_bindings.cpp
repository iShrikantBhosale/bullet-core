#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "../bullet-core.cpp"
#include "../bullet-builder.cpp"

namespace py = pybind11;

PYBIND11_MODULE(bullet_bindings, m) {
    fprintf(stderr, "[DEBUG] bullet_bindings module loaded.\n");
    m.doc() = "Bullet OS Core Python Bindings";

    py::class_<bullet::core::BulletModel>(m, "BulletModel")
        .def(py::init<const char*>(), "Load model from .bullet file")
        .def("generate", &bullet::core::BulletModel::generate, 
             py::arg("prompt"), py::arg("max_tokens") = 128, 
             "Generate text from prompt")
        .def("ner", &bullet::core::BulletModel::ner, "Named Entity Recognition")
        .def("pos", &bullet::core::BulletModel::pos, "Part of Speech Tagging")
        .def("sentiment", &bullet::core::BulletModel::sentiment, "Sentiment Analysis")
        .def("classify", &bullet::core::BulletModel::classify, "Text Classification");

    py::class_<bullet::builder::BulletBuilder>(m, "BulletBuilder")
        .def(py::init<const std::string&>(), "Create builder for output file")
        .def("load_vocab", &bullet::builder::BulletBuilder::load_vocab, "Load vocab from file")
        .def("set_metadata", &bullet::builder::BulletBuilder::set_metadata, "Set model metadata")
        .def("add_tensor", [](bullet::builder::BulletBuilder& self, const std::string& name, std::vector<uint16_t> shape, py::array_t<float> data) {
            py::buffer_info buf = data.request();
            float* ptr = static_cast<float*>(buf.ptr);
            std::vector<float> vec(ptr, ptr + buf.size);
            self.add_tensor(name, shape, vec);
        }, "Add tensor to model")
        .def("build", &bullet::builder::BulletBuilder::build, "Finalize and write .bullet file");
}
