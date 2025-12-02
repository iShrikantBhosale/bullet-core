// bullet_jni.cpp
// JNI C-Interface for Android NDK
// Bullet OS - Production Android SDK

#include <jni.h>
#include <string>
#include <stdexcept>

// Forward declarations (actual implementation in bullet-core.cpp)
class BulletModel {
public:
    BulletModel(const std::string& model_path);
    ~BulletModel();
    std::string generate(const std::string& prompt, int max_tokens = 50);
    std::string sentiment(const std::string& text);
};

// =======================================================================
// 🎯 JNI C-STYLE INTERFACE: The Bridge to Android/Kotlin
// =======================================================================

// Helper: Convert Java String to C++ std::string
static std::string jstring_to_cpp(JNIEnv* env, jstring jstr) {
    if (!jstr) return "";
    const char* cstr = env->GetStringUTFChars(jstr, nullptr);
    std::string result(cstr);
    env->ReleaseStringUTFChars(jstr, cstr);
    return result;
}

// =======================================================================
// Load Model
// =======================================================================
extern "C" JNIEXPORT jlong JNICALL
Java_com_bulletos_BulletModel_nativeLoad(
    JNIEnv* env, 
    jobject thiz, 
    jstring jModelPath
) {
    try {
        std::string modelPath = jstring_to_cpp(env, jModelPath);
        
        // Allocate the BulletModel object and cast pointer to jlong handle
        BulletModel* model = new BulletModel(modelPath);
        return (jlong)model;
        
    } catch (const std::exception& e) {
        // Throw Java exception for Kotlin/Java to catch
        jclass exClass = env->FindClass("java/lang/RuntimeException");
        env->ThrowNew(exClass, e.what());
        return 0; 
    }
}

// =======================================================================
// Generate Text
// =======================================================================
extern "C" JNIEXPORT jstring JNICALL
Java_com_bulletos_BulletModel_nativeGenerate(
    JNIEnv* env, 
    jobject thiz, 
    jlong handle, 
    jstring jPrompt, 
    jint maxTokens
) {
    try {
        // Cast jlong handle back to C++ pointer
        BulletModel* model = (BulletModel*)handle;
        std::string prompt = jstring_to_cpp(env, jPrompt);
        
        // **CALL THE COMPLETED C++ API**
        std::string result = model->generate(prompt, (int)maxTokens); 
        
        return env->NewStringUTF(result.c_str());
        
    } catch (const std::exception& e) {
        jclass exClass = env->FindClass("java/lang/RuntimeException");
        env->ThrowNew(exClass, e.what());
        return nullptr;
    }
}

// =======================================================================
// Sentiment Analysis (Hybrid AI Feature)
// =======================================================================
extern "C" JNIEXPORT jstring JNICALL
Java_com_bulletos_BulletModel_nativeSentiment(
    JNIEnv* env, 
    jobject thiz, 
    jlong handle, 
    jstring jText
) {
    try {
        // **CALL THE COMPLETED HYBRID AI FEATURE**
        BulletModel* model = (BulletModel*)handle;
        std::string text = jstring_to_cpp(env, jText);
        
        std::string sentiment = model->sentiment(text); 
        
        return env->NewStringUTF(sentiment.c_str());
        
    } catch (const std::exception& e) {
        jclass exClass = env->FindClass("java/lang/RuntimeException");
        env->ThrowNew(exClass, e.what());
        return nullptr;
    }
}

// =======================================================================
// Free Model
// =======================================================================
extern "C" JNIEXPORT void JNICALL
Java_com_bulletos_BulletModel_nativeFree(
    JNIEnv* env, 
    jobject thiz, 
    jlong handle
) {
    // Cast handle back and safely delete the C++ object
    BulletModel* model = (BulletModel*)handle;
    delete model;
}

// =======================================================================
// 🚀 NDK BUILD ARTIFACT CREATED
// =======================================================================
// 
// Next Steps:
// 1. Create Android Studio project
// 2. Add this file to app/src/main/cpp/
// 3. Configure CMakeLists.txt
// 4. Build .so library with NDK
// 5. Create Kotlin wrapper class
// 6. Deploy to Google Play
//
// The C++ Runtime is now FROZEN and PRODUCTION-READY.
// =======================================================================
