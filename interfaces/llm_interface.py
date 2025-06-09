from groq import Groq
from config.llm_config import LLMConfig
from utils.prompt_utils import render_prompt
import os

class LLMInterface:
    def __init__(self, config=None):
        self.config = config or LLMConfig()
        self.client = Groq(api_key=self.config.api_key)
    
    def generate(self, phase, prompt_name, data=None, **kwargs):
        """Generate text using the LLM with flexible prompt handling."""
        temperature = kwargs.get('temperature', self.config.temperature)
        max_tokens = kwargs.get('max_tokens', self.config.max_tokens)
        top_p = kwargs.get('top_p', self.config.top_p)
        
        # Determine the content based on input parameters
        if phase and prompt_name and data is not None:
            # Try to load from template
            try:
                content = render_prompt(phase, prompt_name, data)
            except Exception as e:
                print(f"Warning: Could not load template {phase}/{prompt_name}: {e}")
                # Fallback to direct prompt
                if isinstance(data, str):
                    content = data
                elif isinstance(prompt_name, str):
                    content = prompt_name
                else:
                    content = str(data)
        elif prompt_name and data is None:
            # Direct prompt passed as prompt_name
            content = prompt_name
        elif data and not phase and not prompt_name:
            # Direct prompt passed as data
            content = str(data)
        else:
            # Try to use whatever is available
            if isinstance(prompt_name, str):
                content = prompt_name
            elif isinstance(data, str):
                content = data
            else:
                content = "Please provide a response."
        
        try:
            completion = self.client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": content}],
                temperature=temperature,
                max_completion_tokens=max_tokens,
                top_p=top_p,
                stream=False,
            )
            
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error in LLM generation: {e}")
            return None