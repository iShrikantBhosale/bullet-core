#include <iostream>
#include <string>
#include <vector>
#include "../bullet-core.cpp"

void print_usage() {
    std::cout << "Usage: bullet <command> <model.bullet> <input>\n";
    std::cout << "Commands:\n";
    std::cout << "  run   : Generate text\n";
    std::cout << "  ner   : Named Entity Recognition\n";
    std::cout << "  pos   : Part of Speech Tagging\n";
    std::cout << "  sent  : Sentiment Analysis\n";
    std::cout << "  cls   : Classification\n";
}

int main(int argc, char** argv) {
    if (argc < 4) {
        print_usage();
        return 1;
    }
    
    std::string cmd = argv[1];
    std::string model_path = argv[2];
    std::string input = argv[3];
    
    try {
        BulletModel model(model_path);
        
        if (cmd == "run") {
            std::cout << model.generate(input) << "\n";
        } else if (cmd == "ner") {
            auto res = model.ner(input);
            for (const auto& s : res) std::cout << s << "\n";
        } else if (cmd == "pos") {
            auto res = model.pos(input);
            for (const auto& s : res) std::cout << s << "\n";
        } else if (cmd == "sent") {
            std::cout << model.sentiment(input) << "\n";
        } else if (cmd == "cls") {
            std::cout << model.classify(input) << "\n";
        } else {
            std::cerr << "Unknown command: " << cmd << "\n";
            return 1;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}
