// Embedding Lookup - Fast table lookup with prefetching
// output[i] = table[indices[i]]

#include "../utils.h"
#include <omp.h>
#include <cstring>

namespace bullet {

// Embedding lookup (single precision)
void embedding_lookup_f32(
    const float* table, const int* indices,
    float* output, int num_indices, int embedding_dim
) {
    #pragma omp parallel for
    for (int i = 0; i < num_indices; ++i) {
        int idx = indices[i];
        const float* src = table + idx * embedding_dim;
        float* dst = output + i * embedding_dim;
        
        // Prefetch next embedding
        if (i + 1 < num_indices) {
            int next_idx = indices[i + 1];
            __builtin_prefetch(table + next_idx * embedding_dim, 0, 3);
        }
        
        // Copy embedding (SIMD-optimized memcpy)
        std::memcpy(dst, src, embedding_dim * sizeof(float));
    }
}

} // namespace bullet
