// bq4_tokenizer.h
// BPE Tokenizer Integration
// Bullet OS - Load and use BPE tokenizer

#ifndef BQ4_TOKENIZER_H
#define BQ4_TOKENIZER_H

#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <sstream>

namespace bullet {
namespace bq4 {

// ============================================================================
// Simple BPE Tokenizer (loads from JSON)
// ============================================================================

class Tokenizer {
private:
    std::unordered_map<int, std::string> id_to_token;
    std::unordered_map<std::string, int> token_to_id;
    int vocab_size;
    
public:
    Tokenizer() : vocab_size(0) {}
    
    // Load from vocab.txt (simple format: id token)
    bool load_vocab(const std::string& vocab_path) {
        std::ifstream file(vocab_path);
        if (!file.is_open()) {
            std::cerr << "Failed to open vocab file: " << vocab_path << std::endl;
            return false;
        }
        
        std::string line;
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            int id;
            std::string token;
            
            if (iss >> id) {
                std::getline(iss, token);
                // Trim leading space
                if (!token.empty() && token[0] == ' ') {
                    token = token.substr(1);
                }
                
                id_to_token[id] = token;
                token_to_id[token] = id;
            }
        }
        
        vocab_size = id_to_token.size();
        std::cout << "Loaded vocabulary: " << vocab_size << " tokens" << std::endl;
        return true;
    }
    
    // Simple encode (space-split for now)
    std::vector<int> encode(const std::string& text) const {
        std::vector<int> tokens;
        std::istringstream iss(text);
        std::string word;
        
        while (iss >> word) {
            auto it = token_to_id.find(word);
            if (it != token_to_id.end()) {
                tokens.push_back(it->second);
            } else {
                // Unknown token - use ID 0 or special UNK token
                tokens.push_back(0);
            }
        }
        
        return tokens;
    }
    
    // Decode tokens to text
    std::string decode(const std::vector<int>& tokens) {
        std::string result;
        
        for (int id : tokens) {
            auto it = id_to_token.find(id);
            if (it != id_to_token.end()) {
                if (!result.empty()) result += " ";
                result += it->second;
            }
        }
        
        return result;
    }
    
    int get_vocab_size() const { return vocab_size; }
};

} // namespace bq4
} // namespace bullet

#endif // BQ4_TOKENIZER_H
