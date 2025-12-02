#include "test_harness.hpp" // Include harness definitions
#include "../bullet-core.cpp" // Include core implementation for testing internals

TEST(TestHeaderParsing) {
    // Create a dummy header JSON
    std::string json = R"({
        "bullet_version": "1.0",
        "model_name": "test-model",
        "architecture": "bullet-hybrid-transformer",
        "hidden_size": 256,
        "num_layers": 2,
        "num_heads": 4,
        "vocab_size": 100,
        "max_context": 128,
        "tokenizer_start": 4096,
        "weights_start": 8192
    })";
    
    ASSERT_EQ(json_get_string(json, "bullet_version"), "1.0");
    ASSERT_EQ(json_get_string(json, "model_name"), "test-model");
    ASSERT_EQ(json_get_int(json, "hidden_size"), 256);
    ASSERT_EQ(json_get_int(json, "num_layers"), 2);
    ASSERT_EQ(json_get_int(json, "tokenizer_start"), 4096);
    
    return true;
}

TEST(TestMagicConstants) {
    // Check Endianness of Magic
    // "BULK" -> 0x4B4C5542
    ASSERT_EQ(MAGIC_TOKENIZER, 0x4B4C5542);
    // "BWT0" -> 0x30545742
    ASSERT_EQ(MAGIC_WEIGHTS, 0x30545742);
    return true;
}
