// pybind11 bindings for CUDA operations
// Exposes CUDA kernels to Python

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// Forward declarations of CUDA functions
extern "C" {
    // Device management
    bool cuda_is_available();
    const char* cuda_get_device_name();
    int cuda_get_device_id();
    int cuda_get_compute_capability();
    size_t cuda_get_total_memory();
    void cuda_synchronize();
    
    // Memory management
    void* cuda_malloc(size_t bytes);
    void cuda_free(void* ptr);
    size_t cuda_get_used_vram();
    size_t cuda_get_free_vram();
    float cuda_get_usage_percent();
    
    // GEMM kernels
    void gemm_cuda_f32(const float*, const float*, float*, int, int, int);
    void gemm_tn_cuda_f32(const float*, const float*, float*, int, int, int);
    void gemm_nt_cuda_f32(const float*, const float*, float*, int, int, int);
    
    // Other kernels
    void softmax_cuda_f32(const float*, float*, int, int);
    void rmsnorm_cuda_f32(const float*, const float*, float*, int, int, float);
    void layernorm_cuda_f32(const float*, const float*, const float*, float*, int, int, float);
    void vector_add_cuda_f32(const float*, const float*, float*, int);
    void vector_mul_cuda_f32(const float*, const float*, float*, int);
    void vector_scale_cuda_f32(const float*, float, float*, int);
    void vector_relu_cuda_f32(const float*, float*, int);
}

// Helper: Copy numpy array to GPU
void* to_cuda(py::array_t<float> arr) {
    auto buf = arr.request();
    size_t bytes = buf.size * sizeof(float);
    
    void* d_ptr = cuda_malloc(bytes);
    cudaMemcpy(d_ptr, buf.ptr, bytes, cudaMemcpyHostToDevice);
    
    return d_ptr;
}

// Helper: Copy from GPU to numpy array
py::array_t<float> from_cuda(void* d_ptr, py::tuple shape) {
    std::vector<ssize_t> shape_vec;
    size_t total_size = 1;
    
    for (auto item : shape) {
        ssize_t dim = item.cast<ssize_t>();
        shape_vec.push_back(dim);
        total_size *= dim;
    }
    
    auto result = py::array_t<float>(shape_vec);
    auto buf = result.request();
    
    cudaMemcpy(buf.ptr, d_ptr, total_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    return result;
}

// Python wrappers
void gemm_cuda_wrapper(
    uintptr_t A_ptr, uintptr_t B_ptr, uintptr_t C_ptr,
    int M, int N, int K,
    bool transpose_A = false,
    bool transpose_B = false
) {
    const float* A = reinterpret_cast<const float*>(A_ptr);
    const float* B = reinterpret_cast<const float*>(B_ptr);
    float* C = reinterpret_cast<float*>(C_ptr);
    
    if (!transpose_A && !transpose_B) {
        gemm_cuda_f32(A, B, C, M, N, K);
    } else if (transpose_A && !transpose_B) {
        gemm_tn_cuda_f32(A, B, C, M, N, K);
    } else if (!transpose_A && transpose_B) {
        gemm_nt_cuda_f32(A, B, C, M, N, K);
    }
    
    cuda_synchronize();
}

void softmax_cuda_wrapper(uintptr_t in_ptr, uintptr_t out_ptr, int batch, int dim) {
    softmax_cuda_f32(
        reinterpret_cast<const float*>(in_ptr),
        reinterpret_cast<float*>(out_ptr),
        batch, dim
    );
    cuda_synchronize();
}

void rmsnorm_cuda_wrapper(
    uintptr_t in_ptr, uintptr_t weight_ptr, uintptr_t out_ptr,
    int batch, int dim, float eps
) {
    rmsnorm_cuda_f32(
        reinterpret_cast<const float*>(in_ptr),
        reinterpret_cast<const float*>(weight_ptr),
        reinterpret_cast<float*>(out_ptr),
        batch, dim, eps
    );
    cuda_synchronize();
}

// pybind11 module definition
PYBIND11_MODULE(bullet_core_cuda, m) {
    m.doc() = "Bullet-Core CUDA kernels - GPU acceleration for GT 730";
    
    // Device management
    m.def("is_available", &cuda_is_available, "Check if CUDA is available");
    m.def("get_device_name", &cuda_get_device_name, "Get GPU device name");
    m.def("get_device_id", &cuda_get_device_id, "Get device ID");
    m.def("get_compute_capability", &cuda_get_compute_capability, "Get compute capability");
    m.def("get_total_memory", &cuda_get_total_memory, "Get total GPU memory");
    m.def("synchronize", &cuda_synchronize, "Synchronize device");
    
    // Memory management
    m.def("malloc", &cuda_malloc, "Allocate GPU memory", py::arg("bytes"));
    m.def("free", &cuda_free, "Free GPU memory", py::arg("ptr"));
    m.def("get_used_vram", &cuda_get_used_vram, "Get used VRAM");
    m.def("get_free_vram", &cuda_get_free_vram, "Get free VRAM");
    m.def("get_usage_percent", &cuda_get_usage_percent, "Get VRAM usage %");
    
    // Data transfer
    m.def("to_cuda", &to_cuda, "Copy numpy array to GPU", py::arg("array"));
    m.def("from_cuda", &from_cuda, "Copy from GPU to numpy", 
          py::arg("ptr"), py::arg("shape"));
    
    // GEMM
    m.def("gemm", &gemm_cuda_wrapper,
          "Matrix multiplication on GPU",
          py::arg("A_ptr"), py::arg("B_ptr"), py::arg("C_ptr"),
          py::arg("M"), py::arg("N"), py::arg("K"),
          py::arg("transpose_A") = false,
          py::arg("transpose_B") = false);
    
    // Softmax
    m.def("softmax", &softmax_cuda_wrapper,
          "Softmax on GPU",
          py::arg("in_ptr"), py::arg("out_ptr"),
          py::arg("batch"), py::arg("dim"));
    
    // RMSNorm
    m.def("rmsnorm", &rmsnorm_cuda_wrapper,
          "RMSNorm on GPU",
          py::arg("in_ptr"), py::arg("weight_ptr"), py::arg("out_ptr"),
          py::arg("batch"), py::arg("dim"), py::arg("eps") = 1e-6f);
}
