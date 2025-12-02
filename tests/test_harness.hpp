#ifndef TEST_HARNESS_HPP
#define TEST_HARNESS_HPP

#include <iostream>
#include <vector>
#include <string>
#include <functional>
#include <cmath>
#include <iomanip>

// Minimal Test Harness
#define ASSERT_TRUE(x) \
    if (!(x)) { \
        std::cerr << "FAIL: " << #x << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
        return false; \
    }

#define ASSERT_EQ(a, b) \
    if ((a) != (b)) { \
        std::cerr << "FAIL: " << #a << " != " << #b << " (" << (a) << " vs " << (b) << ") at " << __FILE__ << ":" << __LINE__ << "\n"; \
        return false; \
    }

#define ASSERT_NEAR(a, b, tol) \
    if (std::abs((a) - (b)) > (tol)) { \
        std::cerr << "FAIL: " << #a << " !~= " << #b << " (" << (a) << " vs " << (b) << ") at " << __FILE__ << ":" << __LINE__ << "\n"; \
        return false; \
    }

#define TEST(name) \
    bool name(); \
    static bool name##_registered = register_test(#name, name); \
    bool name()

struct TestEntry {
    std::string name;
    std::function<bool()> func;
};

inline std::vector<TestEntry>& get_tests() {
    static std::vector<TestEntry> tests;
    return tests;
}

inline bool register_test(const std::string& name, std::function<bool()> func) {
    get_tests().push_back({name, func});
    return true;
}

#endif
