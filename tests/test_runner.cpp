#include "test_harness.hpp"

int main() {
    std::cout << "Running Bullet Test Suite...\n";
    int passed = 0;
    int failed = 0;
    
    for (const auto& test : get_tests()) {
        std::cout << "[ RUN      ] " << test.name << "\n";
        if (test.func()) {
            std::cout << "[       OK ] " << test.name << "\n";
            passed++;
        } else {
            std::cout << "[  FAILED  ] " << test.name << "\n";
            failed++;
        }
    }
    
    std::cout << "\nResults: " << passed << " passed, " << failed << " failed.\n";
    return (failed == 0) ? 0 : 1;
}
