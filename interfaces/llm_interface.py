from groq import Groq
from config.llm_config import LLMConfig
from utils.prompt_utils import render_prompt

class LLMInterface:
    def __init__(self, config=None):
        self.config = config or LLMConfig()
        self.client = Groq(api_key=self.config.api_key)
    
    def generate(self, phase, prompt_name, data=None, **kwargs):
        """Generate text using the LLM with a prompt template."""
        temperature = kwargs.get('temperature', self.config.temperature)
        max_tokens = kwargs.get('max_tokens', self.config.max_tokens)
        top_p = kwargs.get('top_p', self.config.top_p)
        
        if data:
            content = render_prompt(phase, prompt_name, data)
        else:
            content = prompt_name 
        
        completion = self.client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "user", "content": content}],
            temperature=temperature,
            max_completion_tokens=max_tokens,
            top_p=top_p,
            stream=False,
        )
        
        return completion.choices[0].message.content