cmake_minimum_required(VERSION 3.10)
project(BulletTests)

set(CMAKE_CXX_STANDARD 17)

enable_testing()

# We include the source files directly in tests because they are single-file libs
# and we want to test internals.
# Note: We must ensure we don't link multiple definitions.
# The tests include "test_main.cpp" which includes the harness.
# They also include "../bullet-core.cpp" etc.
# This is messy if we compile them as separate objects and link.
# Better: Compile ONE test binary that includes all test files.

add_executable(run_tests
    test_runner.cpp
    test_header.cpp
    test_tokenizer.cpp
    test_bq4_quant.cpp
    test_tensor_loader.cpp
    test_transformer_small.cpp
    test_end_to_end.cpp
)

# Define BULLET_TEST to disable main() in core/builder
target_compile_definitions(run_tests PRIVATE BULLET_TEST)

add_test(NAME AllTests COMMAND run_tests)
