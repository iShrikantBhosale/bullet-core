// CUDA Memory Management
// Tracks VRAM usage and handles automatic CPU fallback

#include "cuda_utils.cuh"
#include <unordered_map>
#include <mutex>
#include <stdexcept>

class CudaMemoryManager {
private:
    static constexpr size_t VRAM_LIMIT = 2ULL * 1024 * 1024 * 1024;  // 2GB
    static constexpr float USAGE_THRESHOLD = 0.90f;  // 90% max usage
    
    size_t used_vram = 0;
    std::unordered_map<void*, size_t> allocations;
    std::mutex mutex;
    
    static CudaMemoryManager* instance;
    
    CudaMemoryManager() = default;
    
public:
    static CudaMemoryManager& get_instance() {
        if (!instance) {
            instance = new CudaMemoryManager();
        }
        return *instance;
    }
    
    void* allocate(size_t bytes) {
        std::lock_guard<std::mutex> lock(mutex);
        
        // Check if we have enough VRAM
        if (used_vram + bytes > VRAM_LIMIT * USAGE_THRESHOLD) {
            throw std::runtime_error("CUDA OOM: VRAM limit exceeded");
        }
        
        void* ptr;
        cudaError_t err = cudaMalloc(&ptr, bytes);
        
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("CUDA allocation failed: ") + 
                                   cudaGetErrorString(err));
        }
        
        allocations[ptr] = bytes;
        used_vram += bytes;
        
        return ptr;
    }
    
    void free(void* ptr) {
        std::lock_guard<std::mutex> lock(mutex);
        
        if (allocations.count(ptr)) {
            cudaFree(ptr);
            used_vram -= allocations[ptr];
            allocations.erase(ptr);
        }
    }
    
    size_t get_used_vram() const {
        return used_vram;
    }
    
    size_t get_free_vram() const {
        return VRAM_LIMIT - used_vram;
    }
    
    float get_usage_percent() const {
        return (float)used_vram / VRAM_LIMIT * 100.0f;
    }
    
    void clear() {
        std::lock_guard<std::mutex> lock(mutex);
        
        for (auto& pair : allocations) {
            cudaFree(pair.first);
        }
        
        allocations.clear();
        used_vram = 0;
    }
};

CudaMemoryManager* CudaMemoryManager::instance = nullptr;

// C API for Python bindings
extern "C" {
    void* cuda_malloc(size_t bytes) {
        return CudaMemoryManager::get_instance().allocate(bytes);
    }
    
    void cuda_free(void* ptr) {
        CudaMemoryManager::get_instance().free(ptr);
    }
    
    size_t cuda_get_used_vram() {
        return CudaMemoryManager::get_instance().get_used_vram();
    }
    
    size_t cuda_get_free_vram() {
        return CudaMemoryManager::get_instance().get_free_vram();
    }
    
    float cuda_get_usage_percent() {
        return CudaMemoryManager::get_instance().get_usage_percent();
    }
}
