// bullet-api.h
// C API for Bullet-Core (for WASM and mobile bindings)

#ifndef BULLET_API_H
#define BULLET_API_H

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#define BULLET_API EMSCRIPTEN_KEEPALIVE
#else
#define BULLET_API
#endif

// Opaque handle to model
typedef void* BulletModelHandle;

// Initialize model from file path
BULLET_API BulletModelHandle bullet_load_model(const char* model_path);

// Generate text from prompt
// Returns malloc'd string (caller must free)
BULLET_API char* bullet_generate(BulletModelHandle model, const char* prompt, int max_tokens);

// Free model
BULLET_API void bullet_free_model(BulletModelHandle model);

// Free generated string
BULLET_API void bullet_free_string(char* str);

#ifdef __cplusplus
}
#endif

#endif // BULLET_API_H
