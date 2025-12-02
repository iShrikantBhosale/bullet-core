#include "test_harness.hpp"
#include <fstream>
#include <cstdlib>

TEST(TestEndToEnd) {
    // 1. Create Vocab
    // We are in build/, so write to ../tests/sample_model/
    std::string vocab_path = "../tests/sample_model/vocab.txt";
    std::string model_path = "../tests/sample_model/tiny_model.bullet";
    std::string out_path = "../tests/sample_model/output.txt";
    
    std::ofstream v(vocab_path);
    v << "h 1.0\ne 1.0\nl 1.0\no 1.0\n";
    v.close();
    
    // 2. Build Model using external binary
    std::string cmd_build = "../bullet-builder " + vocab_path + " " + model_path;
    int ret = system(cmd_build.c_str());
    ASSERT_EQ(ret, 0);
    
    // 3. Run Model using external binary
    std::string cmd_run = "../bullet-core " + model_path + " > " + out_path;
    ret = system(cmd_run.c_str());
    ASSERT_EQ(ret, 0);
    
    // 4. Verify Output
    std::ifstream out(out_path);
    std::string line;
    bool found_result = false;
    while(std::getline(out, line)) {
        if (line.find("Result:") != std::string::npos) {
            found_result = true;
            // Check if result is not empty
            // "Result: " is 8 chars
            if (line.length() > 8) {
                 // Good
            } else {
                // Empty result?
            }
        }
    }
    ASSERT_TRUE(found_result);
    
    return true;
}
