import os

class LLMConfig:
    def __init__(self):
        self.api_key = os.environ.get("GROQ_API_KEY", "")
        self.model = "llama-3.3-70b-versatile"
        self.temperature = 0.7
        self.max_tokens = 1024
        self.top_p = 1.0
        
    def update(self, **kwargs):
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self