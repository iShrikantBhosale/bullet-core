// CUDA Device Management
// Auto-detect GPU and handle device selection

#include "cuda_utils.cuh"
#include <string>

class CudaDevice {
private:
    int device_id = -1;
    bool available = false;
    cudaDeviceProp properties;
    
    static CudaDevice* instance;
    
    CudaDevice() {
        initialize();
    }
    
    void initialize() {
        int device_count;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        
        if (err != cudaSuccess || device_count == 0) {
            available = false;
            return;
        }
        
        // Find best device (prefer CC 3.5+)
        int best_device = -1;
        int best_compute = 0;
        
        for (int i = 0; i < device_count; ++i) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);
            
            int compute = prop.major * 10 + prop.minor;
            
            // GT 730 is CC 3.5
            if (compute >= 35 && compute > best_compute) {
                best_device = i;
                best_compute = compute;
                properties = prop;
            }
        }
        
        if (best_device >= 0) {
            device_id = best_device;
            available = true;
            cudaSetDevice(device_id);
        }
    }
    
public:
    static CudaDevice& get_instance() {
        if (!instance) {
            instance = new CudaDevice();
        }
        return *instance;
    }
    
    bool is_available() const { return available; }
    int get_device_id() const { return device_id; }
    
    std::string get_device_name() const {
        if (!available) return "No CUDA device";
        return std::string(properties.name);
    }
    
    int get_compute_capability() const {
        if (!available) return 0;
        return properties.major * 10 + properties.minor;
    }
    
    size_t get_total_memory() const {
        if (!available) return 0;
        return properties.totalGlobalMem;
    }
    
    int get_multiprocessor_count() const {
        if (!available) return 0;
        return properties.multiProcessorCount;
    }
    
    void synchronize() {
        if (available) {
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }
    
    void reset() {
        if (available) {
            cudaDeviceReset();
        }
    }
};

CudaDevice* CudaDevice::instance = nullptr;

// C API for Python bindings
extern "C" {
    bool cuda_is_available() {
        return CudaDevice::get_instance().is_available();
    }
    
    const char* cuda_get_device_name() {
        static std::string name;
        name = CudaDevice::get_instance().get_device_name();
        return name.c_str();
    }
    
    int cuda_get_device_id() {
        return CudaDevice::get_instance().get_device_id();
    }
    
    int cuda_get_compute_capability() {
        return CudaDevice::get_instance().get_compute_capability();
    }
    
    size_t cuda_get_total_memory() {
        return CudaDevice::get_instance().get_total_memory();
    }
    
    void cuda_synchronize() {
        CudaDevice::get_instance().synchronize();
    }
}
