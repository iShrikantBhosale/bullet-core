import os
import sys

# Try to import the native extension
try:
    import bullet_bindings
except ImportError:
    print("Error: Could not import bullet_bindings. Make sure it is installed or in PYTHONPATH.")
    sys.exit(1)

class Bullet:
    """
    High-level Python wrapper for Bullet OS models.
    """
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            self.model = bullet_bindings.BulletModel(model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load Bullet model: {e}")

    def chat(self, text, max_tokens=128):
        """Generate text response."""
        return self.model.generate(text, max_tokens)

    def ner(self, text):
        """Extract Named Entities."""
        return self.model.ner(text)

    def pos(self, text):
        """Get Part-of-Speech tags."""
        return self.model.pos(text)

    def sentiment(self, text):
        """Analyze sentiment (0.0 to 1.0)."""
        return self.model.sentiment(text)

    def classify(self, text):
        """Classify text into categories."""
        return self.model.classify(text)
