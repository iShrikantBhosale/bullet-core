// bq4_loader.h
// BQ4 File Loader - Loads .bq4 quantized models
// Bullet OS - 4-bit Quantization System

#ifndef BQ4_LOADER_H
#define BQ4_LOADER_H

#include "bq4_format.h"
#include "bq4_kernels.h"
#include <fstream>
#include <iostream>
#include <cstring>

namespace bullet {
namespace bq4 {

class Loader {
private:
    std::string filepath;
    Header header;
    std::vector<TensorEntry> directory;
    std::ifstream file;
    
public:
    Loader(const std::string& path) : filepath(path) {}
    
    ~Loader() {
        if (file.is_open()) file.close();
    }
    
    // Load BQ4 file header and directory
    bool load_metadata() {
        file.open(filepath, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Failed to open " << filepath << std::endl;
            return false;
        }
        
        // Read header
        file.read(header.magic, 4);
        if (strncmp(header.magic, "BQ4F", 4) != 0) {
            std::cerr << "Invalid BQ4 file: bad magic" << std::endl;
            return false;
        }
        
        file.read(reinterpret_cast<char*>(&header.version), 4);
        file.read(reinterpret_cast<char*>(&header.num_tensors), 4);
        
        std::cout << "BQ4 File: version=" << header.version 
                  << ", tensors=" << header.num_tensors << std::endl;
        
        // Read tensor directory
        directory.resize(header.num_tensors);
        
        for (uint32_t i = 0; i < header.num_tensors; i++) {
            TensorEntry& entry = directory[i];
            
            // Name
            uint32_t name_len;
            file.read(reinterpret_cast<char*>(&name_len), 4);
            
            char* name_buf = new char[name_len + 1];
            file.read(name_buf, name_len);
            name_buf[name_len] = '\0';
            entry.name = std::string(name_buf);
            delete[] name_buf;
            
            // Dims
            uint32_t ndims;
            file.read(reinterpret_cast<char*>(&ndims), 4);
            entry.dims.resize(ndims);
            
            for (uint32_t d = 0; d < ndims; d++) {
                file.read(reinterpret_cast<char*>(&entry.dims[d]), 4);
            }
            
            // Block info
            file.read(reinterpret_cast<char*>(&entry.block_size), 4);
            file.read(reinterpret_cast<char*>(&entry.num_blocks), 4);
            file.read(reinterpret_cast<char*>(&entry.data_offset), 8);
            
            std::cout << "  Tensor " << i << ": " << entry.name 
                      << " [";
            for (size_t d = 0; d < entry.dims.size(); d++) {
                std::cout << entry.dims[d];
                if (d < entry.dims.size() - 1) std::cout << "x";
            }
            std::cout << "] blocks=" << entry.num_blocks << std::endl;
        }
        
        return true;
    }
    
    // Load a specific tensor by index
    Tensor load_tensor(uint32_t tensor_idx) {
        if (tensor_idx >= directory.size()) {
            std::cerr << "Invalid tensor index: " << tensor_idx << std::endl;
            return Tensor();
        }
        
        const TensorEntry& entry = directory[tensor_idx];
        Tensor tensor;
        tensor.name = entry.name;
        tensor.dims = entry.dims;
        tensor.block_size = entry.block_size;
        tensor.blocks.resize(entry.num_blocks);
        
        // Seek to tensor data
        file.seekg(entry.data_offset);
        
        // Read all blocks
        for (uint32_t b = 0; b < entry.num_blocks; b++) {
            Block& block = tensor.blocks[b];
            
            // Read packed data
            uint32_t packed_size = entry.block_size / 2;
            block.data.resize(packed_size);
            file.read(reinterpret_cast<char*>(block.data.data()), packed_size);
            
            // Read scale
            file.read(reinterpret_cast<char*>(&block.scale), 4);
        }
        
        return tensor;
    }
    
    // Load tensor by name
    Tensor load_tensor(const std::string& name) {
        for (uint32_t i = 0; i < directory.size(); i++) {
            if (directory[i].name == name) {
                return load_tensor(i);
            }
        }
        std::cerr << "Tensor not found: " << name << std::endl;
        return Tensor();
    }
    
    // Dequantize tensor to FP32
    std::vector<float> dequantize_tensor(const Tensor& tensor) {
        uint32_t total_weights = 1;
        for (auto d : tensor.dims) total_weights *= d;
        
        std::vector<float> output(total_weights);
        
        for (size_t b = 0; b < tensor.blocks.size(); b++) {
            const Block& block = tensor.blocks[b];
            uint32_t offset = b * tensor.block_size;
            
            dequantize(
                block.data.data(),
                block.scale,
                output.data() + offset,
                tensor.block_size
            );
        }
        
        return output;
    }
    
    uint32_t num_tensors() const { return header.num_tensors; }
    const TensorEntry& get_entry(uint32_t idx) const { return directory[idx]; }
};

} // namespace bq4
} // namespace bullet

#endif // BQ4_LOADER_H
