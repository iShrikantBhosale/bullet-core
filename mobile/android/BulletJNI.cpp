// BulletJNI.cpp
// Android JNI Wrapper for Bullet OS
// Provides Java interface to BQ4 engine

#include <jni.h>
#include "bq4/bq4_transformer.h"
#include "bq4/bq4_generation.h"
#include "bq4/bq4_sentiment.h"
#include "bq4/bq4_tokenizer.h"
#include <string>

using namespace bullet::bq4;

// Global model and tokenizer (loaded once)
static BulletModel* g_model = nullptr;
static Tokenizer* g_tokenizer = nullptr;
static SentimentHead* g_sentiment_head = nullptr;

extern "C" {

// Load model from assets
JNIEXPORT jboolean JNICALL
Java_com_bullet_BulletModel_loadModel(
    JNIEnv* env,
    jobject /* this */,
    jstring model_path,
    jstring vocab_path
) {
    const char* model_path_str = env->GetStringUTFChars(model_path, nullptr);
    const char* vocab_path_str = env->GetStringUTFChars(vocab_path, nullptr);
    
    // Load tokenizer
    g_tokenizer = new Tokenizer();
    if (!g_tokenizer->load_vocab(vocab_path_str)) {
        env->ReleaseStringUTFChars(vocab_path, vocab_path_str);
        env->ReleaseStringUTFChars(model_path, model_path_str);
        return JNI_FALSE;
    }
    
    // Load model (placeholder - needs full implementation)
    g_model = new BulletModel();
    // TODO: Load BQ4 model and map tensors
    
    env->ReleaseStringUTFChars(vocab_path, vocab_path_str);
    env->ReleaseStringUTFChars(model_path, model_path_str);
    
    return JNI_TRUE;
}

// Generate text
JNIEXPORT jstring JNICALL
Java_com_bullet_BulletModel_generate(
    JNIEnv* env,
    jobject /* this */,
    jstring prompt
) {
    if (!g_model || !g_tokenizer) {
        return env->NewStringUTF("Error: Model not loaded");
    }
    
    const char* prompt_str = env->GetStringUTFChars(prompt, nullptr);
    
    // Encode prompt
    std::vector<int> tokens = g_tokenizer->encode(prompt_str);
    
    // Generate (placeholder)
    // std::vector<int> output_tokens = generate(tokens, *g_model);
    
    // Decode
    // std::string output = g_tokenizer->decode(output_tokens);
    
    std::string output = std::string(prompt_str) + " [generation ready]";
    
    env->ReleaseStringUTFChars(prompt, prompt_str);
    
    return env->NewStringUTF(output.c_str());
}

// Sentiment analysis
JNIEXPORT jstring JNICALL
Java_com_bullet_BulletModel_sentiment(
    JNIEnv* env,
    jobject /* this */,
    jstring text
) {
    if (!g_model || !g_tokenizer || !g_sentiment_head) {
        return env->NewStringUTF("UNKNOWN");
    }
    
    const char* text_str = env->GetStringUTFChars(text, nullptr);
    
    std::string result = sentiment_analysis(
        text_str, *g_model, *g_sentiment_head, *g_tokenizer
    );
    
    env->ReleaseStringUTFChars(text, text_str);
    
    return env->NewStringUTF(result.c_str());
}

// Cleanup
JNIEXPORT void JNICALL
Java_com_bullet_BulletModel_unload(
    JNIEnv* env,
    jobject /* this */
) {
    delete g_model;
    delete g_tokenizer;
    delete g_sentiment_head;
    
    g_model = nullptr;
    g_tokenizer = nullptr;
    g_sentiment_head = nullptr;
}

} // extern "C"
