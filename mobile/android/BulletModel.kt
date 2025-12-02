// BulletModel.kt
// Kotlin Wrapper for Bullet OS Android SDK
// Package: com.bulletos

package com.bulletos

class BulletModel(modelPath: String) {
    
    private var nativeHandle: Long = 0
    
    init {
        System.loadLibrary("bulletos")
        nativeHandle = nativeLoad(modelPath)
        if (nativeHandle == 0L) {
            throw RuntimeException("Failed to load Bullet model")
        }
    }
    
    /**
     * Generate text from a prompt
     * @param prompt Input text prompt
     * @param maxTokens Maximum tokens to generate (default: 50)
     * @return Generated text
     */
    fun generate(prompt: String, maxTokens: Int = 50): String {
        if (nativeHandle == 0L) {
            throw IllegalStateException("Model not loaded")
        }
        return nativeGenerate(nativeHandle, prompt, maxTokens)
    }
    
    /**
     * Analyze sentiment of text (Hybrid AI)
     * @param text Input text to analyze
     * @return Sentiment label: "POSITIVE", "NEGATIVE", or "NEUTRAL"
     */
    fun sentiment(text: String): String {
        if (nativeHandle == 0L) {
            throw IllegalStateException("Model not loaded")
        }
        return nativeSentiment(nativeHandle, text)
    }
    
    /**
     * Free native resources
     */
    fun close() {
        if (nativeHandle != 0L) {
            nativeFree(nativeHandle)
            nativeHandle = 0
        }
    }
    
    protected fun finalize() {
        close()
    }
    
    // Native method declarations
    private external fun nativeLoad(modelPath: String): Long
    private external fun nativeGenerate(handle: Long, prompt: String, maxTokens: Int): String
    private external fun nativeSentiment(handle: Long, text: String): String
    private external fun nativeFree(handle: Long)
    
    companion object {
        /**
         * Example usage:
         * 
         * val model = BulletModel("/sdcard/marathi_v3.bq4")
         * val output = model.generate("जीवन", maxTokens = 100)
         * val sentiment = model.sentiment("हे खूप छान आहे")
         * model.close()
         */
    }
}
