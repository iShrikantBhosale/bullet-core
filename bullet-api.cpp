// bullet-api.cpp
// C API implementation

#include "bullet-api.h"
#include "bullet-core.cpp"
#include <cstring>
#include <cstdlib>

using namespace bullet::core;

extern "C" {

BULLET_API BulletModelHandle bullet_load_model(const char* model_path) {
    try {
        BulletModel* model = new BulletModel(model_path);
        return static_cast<BulletModelHandle>(model);
    } catch (...) {
        return nullptr;
    }
}

BULLET_API char* bullet_generate(BulletModelHandle handle, const char* prompt, int max_tokens) {
    if (!handle || !prompt) return nullptr;
    
    try {
        BulletModel* model = static_cast<BulletModel*>(handle);
        std::string result = model->generate(std::string(prompt), max_tokens);
        
        // Allocate and copy
        char* output = (char*)malloc(result.size() + 1);
        if (output) {
            strcpy(output, result.c_str());
        }
        return output;
    } catch (...) {
        return nullptr;
    }
}

BULLET_API void bullet_free_model(BulletModelHandle handle) {
    if (handle) {
        delete static_cast<BulletModel*>(handle);
    }
}

BULLET_API void bullet_free_string(char* str) {
    if (str) {
        free(str);
    }
}

} // extern "C"
