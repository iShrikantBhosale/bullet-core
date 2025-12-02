#include "test_harness.hpp"
// bullet-core.cpp included via test_main or separately?
// We need BulletLoader.
// Since we are compiling multiple files, we should be careful.
// Let's assume we include bullet-core.cpp here.
#include "../bullet-core.cpp"

TEST(TestFNV1a) {
    ASSERT_EQ(hash_name("test"), fnv1a64("test", 4));
    // Known hash check if possible, or just consistency
    ASSERT_EQ(hash_name("layers.0.attention.query"), hash_name("layers.0.attention.query"));
    ASSERT_TRUE(hash_name("a") != hash_name("b"));
    return true;
}

TEST(TestTensorView) {
    TensorView tv = {}; // Zero init
    tv.rank = 2;
    tv.shape[0] = 10;
    tv.shape[1] = 20;
    ASSERT_EQ(tv.numel(), 200);
    
    tv.rank = 4;
    tv.shape[2] = 5;
    tv.shape[3] = 2;
    ASSERT_EQ(tv.numel(), 2000);
    return true;
}
