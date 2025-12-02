// pybind11 bindings for Bullet-Core CPU kernels
// Exposes C++ functions to Python with NumPy array support

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "utils.h"

namespace py = pybind11;

// Forward declarations of kernel functions
namespace bullet {
    void gemm_f32(const float*, const float*, float*, int, int, int, bool, bool);
    void softmax_f32(const float*, float*, int, int);
    void rmsnorm_f32(const float*, const float*, float*, int, int, float);
    void layernorm_f32(const float*, const float*, const float*, float*, int, int, float);
    void embedding_lookup_f32(const float*, const int*, float*, int, int);
    void vector_add_f32(const float*, const float*, float*, int);
    void vector_mul_f32(const float*, const float*, float*, int);
    void vector_scale_f32(const float*, float, float*, int);
    void vector_sub_f32(const float*, const float*, float*, int);
    void vector_relu_f32(const float*, float*, int);
}

// Python wrapper for GEMM
void gemm_f32_wrapper(
    py::array_t<float> A,
    py::array_t<float> B,
    py::array_t<float> C,
    int M, int N, int K,
    bool transpose_A = false,
    bool transpose_B = false
) {
    auto a_buf = A.request();
    auto b_buf = B.request();
    auto c_buf = C.request();
    
    bullet::gemm_f32(
        static_cast<float*>(a_buf.ptr),
        static_cast<float*>(b_buf.ptr),
        static_cast<float*>(c_buf.ptr),
        M, N, K, transpose_A, transpose_B
    );
}

// Python wrapper for Softmax
void softmax_f32_wrapper(
    py::array_t<float> input,
    py::array_t<float> output,
    int batch, int dim
) {
    auto in_buf = input.request();
    auto out_buf = output.request();
    
    bullet::softmax_f32(
        static_cast<float*>(in_buf.ptr),
        static_cast<float*>(out_buf.ptr),
        batch, dim
    );
}

// Python wrapper for RMSNorm
void rmsnorm_f32_wrapper(
    py::array_t<float> input,
    py::array_t<float> weight,
    py::array_t<float> output,
    int batch, int dim,
    float eps = 1e-6f
) {
    auto in_buf = input.request();
    auto w_buf = weight.request();
    auto out_buf = output.request();
    
    bullet::rmsnorm_f32(
        static_cast<float*>(in_buf.ptr),
        static_cast<float*>(w_buf.ptr),
        static_cast<float*>(out_buf.ptr),
        batch, dim, eps
    );
}

// Python wrapper for LayerNorm
void layernorm_f32_wrapper(
    py::array_t<float> input,
    py::array_t<float> weight,
    py::array_t<float> bias,
    py::array_t<float> output,
    int batch, int dim,
    float eps = 1e-6f
) {
    auto in_buf = input.request();
    auto w_buf = weight.request();
    auto b_buf = bias.request();
    auto out_buf = output.request();
    
    bullet::layernorm_f32(
        static_cast<float*>(in_buf.ptr),
        static_cast<float*>(w_buf.ptr),
        static_cast<float*>(b_buf.ptr),
        static_cast<float*>(out_buf.ptr),
        batch, dim, eps
    );
}

// Python wrapper for Embedding Lookup
void embedding_lookup_f32_wrapper(
    py::array_t<float> table,
    py::array_t<int> indices,
    py::array_t<float> output,
    int num_indices, int embedding_dim
) {
    auto t_buf = table.request();
    auto i_buf = indices.request();
    auto o_buf = output.request();
    
    bullet::embedding_lookup_f32(
        static_cast<float*>(t_buf.ptr),
        static_cast<int*>(i_buf.ptr),
        static_cast<float*>(o_buf.ptr),
        num_indices, embedding_dim
    );
}

// Python wrappers for vector operations
void vector_add_f32_wrapper(
    py::array_t<float> a, py::array_t<float> b,
    py::array_t<float> out, int size
) {
    bullet::vector_add_f32(
        static_cast<float*>(a.request().ptr),
        static_cast<float*>(b.request().ptr),
        static_cast<float*>(out.request().ptr),
        size
    );
}

void vector_mul_f32_wrapper(
    py::array_t<float> a, py::array_t<float> b,
    py::array_t<float> out, int size
) {
    bullet::vector_mul_f32(
        static_cast<float*>(a.request().ptr),
        static_cast<float*>(b.request().ptr),
        static_cast<float*>(out.request().ptr),
        size
    );
}

void vector_scale_f32_wrapper(
    py::array_t<float> a, float scale,
    py::array_t<float> out, int size
) {
    bullet::vector_scale_f32(
        static_cast<float*>(a.request().ptr),
        scale,
        static_cast<float*>(out.request().ptr),
        size
    );
}

// pybind11 module definition
PYBIND11_MODULE(bullet_core_cpp, m) {
    m.doc() = "Bullet-Core CPU kernels - SIMD-optimized operations for deep learning";
    
    // GEMM
    m.def("gemm_f32", &gemm_f32_wrapper,
          "Matrix multiplication (fp32, SIMD-optimized)",
          py::arg("A"), py::arg("B"), py::arg("C"),
          py::arg("M"), py::arg("N"), py::arg("K"),
          py::arg("transpose_A") = false,
          py::arg("transpose_B") = false);
    
    // Softmax
    m.def("softmax_f32", &softmax_f32_wrapper,
          "Softmax activation (fp32, numerically stable)",
          py::arg("input"), py::arg("output"),
          py::arg("batch"), py::arg("dim"));
    
    // RMSNorm
    m.def("rmsnorm_f32", &rmsnorm_f32_wrapper,
          "RMS Normalization (fp32, for LLaMA/BULLET models)",
          py::arg("input"), py::arg("weight"), py::arg("output"),
          py::arg("batch"), py::arg("dim"), py::arg("eps") = 1e-6f);
    
    // LayerNorm
    m.def("layernorm_f32", &layernorm_f32_wrapper,
          "Layer Normalization (fp32)",
          py::arg("input"), py::arg("weight"), py::arg("bias"), py::arg("output"),
          py::arg("batch"), py::arg("dim"), py::arg("eps") = 1e-6f);
    
    // Embedding
    m.def("embedding_lookup_f32", &embedding_lookup_f32_wrapper,
          "Embedding table lookup (fp32, with prefetching)",
          py::arg("table"), py::arg("indices"), py::arg("output"),
          py::arg("num_indices"), py::arg("embedding_dim"));
    
    // Vector operations
    m.def("vector_add_f32", &vector_add_f32_wrapper,
          "Element-wise addition (fp32, SIMD)",
          py::arg("a"), py::arg("b"), py::arg("out"), py::arg("size"));
    
    m.def("vector_mul_f32", &vector_mul_f32_wrapper,
          "Element-wise multiplication (fp32, SIMD)",
          py::arg("a"), py::arg("b"), py::arg("out"), py::arg("size"));
    
    m.def("vector_scale_f32", &vector_scale_f32_wrapper,
          "Scalar multiplication (fp32, SIMD)",
          py::arg("a"), py::arg("scale"), py::arg("out"), py::arg("size"));
}
