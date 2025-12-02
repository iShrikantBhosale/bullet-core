#include "test_harness.hpp"
#include "../bullet-core.cpp"

TEST(TestTokenizerLoading) {
    BulletTokenizer tokenizer;
    // Mock data: Magic(4) + Size(4) + [Score(4)+Len(2)+Text(Len)]...
    std::vector<uint8_t> data;
    
    // Magic "BULK"
    data.push_back('B'); data.push_back('U'); data.push_back('L'); data.push_back('K');
    
    // Size (ignored by load, but we put placeholder)
    for(int i=0; i<4; ++i) data.push_back(0);
    
    // Token 1: "a", score 1.0
    float score = 1.0f;
    uint8_t* s_ptr = reinterpret_cast<uint8_t*>(&score);
    for(int i=0; i<4; ++i) data.push_back(s_ptr[i]);
    
    uint16_t len = 1;
    uint8_t* l_ptr = reinterpret_cast<uint8_t*>(&len);
    data.push_back(l_ptr[0]); data.push_back(l_ptr[1]);
    data.push_back('a');
    
    // Token 2: "b", score 2.0
    score = 2.0f;
    s_ptr = reinterpret_cast<uint8_t*>(&score);
    for(int i=0; i<4; ++i) data.push_back(s_ptr[i]); 
    
    len = 1;
    data.push_back(l_ptr[0]); data.push_back(l_ptr[1]);
    data.push_back('b');
    
    tokenizer.load(data.data(), 2);
    
    ASSERT_EQ(tokenizer.size(), 2);
    ASSERT_EQ(tokenizer.decode(0), "a");
    ASSERT_EQ(tokenizer.decode(1), "b");
    
    return true;
}

TEST(TestTokenizerEncode) {
    BulletTokenizer tokenizer;
    
    // Helper to build binary buffer
    auto add_token = [](std::vector<uint8_t>& d, std::string t, float s) {
        uint8_t* sp = reinterpret_cast<uint8_t*>(&s);
        for(int i=0; i<4; ++i) d.push_back(sp[i]);
        uint16_t l = t.size();
        uint8_t* lp = reinterpret_cast<uint8_t*>(&l);
        d.push_back(lp[0]); d.push_back(lp[1]);
        for(char c : t) d.push_back(c);
    };
    
    std::vector<uint8_t> data;
    data.push_back('B'); data.push_back('U'); data.push_back('L'); data.push_back('K');
    for(int i=0; i<4; ++i) data.push_back(0); // Size placeholder
    
    add_token(data, "h", 1.0f); // 0
    add_token(data, "e", 1.0f); // 1
    add_token(data, "l", 1.0f); // 2
    add_token(data, "o", 1.0f); // 3
    add_token(data, "he", 5.0f); // 4
    add_token(data, "ll", 5.0f); // 5
    add_token(data, "hell", 7.0f); // 6
    add_token(data, "hello", 10.0f); // 7
    
    tokenizer.load(data.data(), 8);
    
    // "hello" -> should be single token 7
    std::vector<int> ids = tokenizer.encode("hello");
    ASSERT_EQ(ids.size(), 1);
    ASSERT_EQ(ids[0], 7);
    
    // "hel" -> "he" + "l" (indices 4, 2)
    ids = tokenizer.encode("hel");
    ASSERT_EQ(ids.size(), 2);
    ASSERT_EQ(ids[0], 4);
    ASSERT_EQ(ids[1], 2);
    
    return true;
}
