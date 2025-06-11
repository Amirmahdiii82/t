import os

class VLMConfig:
    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY", "")
        self.model = "gemini-2.0-flash-exp-image-generation"
        self.text_model = "gemini-2.0-flash-exp-image-generation"  
        self.temperature = 1.0
        self.top_p = 0.95
        self.top_k = 40
        self.max_output_tokens = 8192
        
    def update(self, **kwargs):
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self