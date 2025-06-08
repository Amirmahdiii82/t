import os
import base64
import mimetypes
from google import genai
from google.genai import types
from config.vlm_config import VLMConfig
from utils.prompt_utils import render_prompt
from utils.file_utils import ensure_directory
from PIL import Image
import io

class VLMInterface:
    def __init__(self, config=None):
        self.config = config or VLMConfig()
        self.client = genai.Client(
            api_key=os.environ.get("GEMINI_API_KEY", self.config.api_key),
        )
    
    def generate_text(self, phase, prompt_name, data=None, **kwargs):
        """Generate text using the VLM with a prompt template."""
        temperature = kwargs.get('temperature', self.config.temperature)
        top_p = kwargs.get('top_p', self.config.top_p)
        top_k = kwargs.get('top_k', self.config.top_k)
        
        # Handle different prompt input methods
        if phase and prompt_name and data:
            # Load from template file
            try:
                prompt = render_prompt(phase, prompt_name, data)
            except Exception as e:
                print(f"Error loading prompt template: {e}")
                # Fallback to using prompt_name as direct prompt
                prompt = str(prompt_name)
        elif data and not phase and not prompt_name:
            # Direct prompt passed as data
            prompt = str(data)
        elif prompt_name and not phase:
            # Direct prompt passed as prompt_name
            prompt = str(prompt_name)
        else:
            # Fallback
            prompt = str(prompt_name) if prompt_name else "Generate a response"
        
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=prompt),
                ],
            ),
        ]
        
        generate_content_config = types.GenerateContentConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_output_tokens=self.config.max_output_tokens,
            response_modalities=["text"],
        )
        
        try:
            response = self.client.models.generate_content(
                model=self.config.text_model,
                contents=contents,
                config=generate_content_config,
            )
            
            full_text = ""
            if hasattr(response, 'candidates') and response.candidates:
                for candidate in response.candidates:
                    if hasattr(candidate, 'content') and candidate.content:
                        for part in candidate.content.parts:
                            if hasattr(part, 'text'):
                                full_text += part.text or ""
            
            return full_text
        
        except Exception as e:
            print(f"Error generating text: {e}")
            return f"Error generating text: {str(e)}"
    
    def save_binary_file(self, file_name, data):
        """Save binary data to a file."""
        ensure_directory(os.path.dirname(file_name))
        with open(file_name, "wb") as f:
            f.write(data)
    
    def generate_image(self, phase, prompt_name, data, output_path, **kwargs):
        """Generate an image using the VLM with a prompt template."""
        temperature = kwargs.get('temperature', self.config.temperature)
        top_p = kwargs.get('top_p', self.config.top_p)
        top_k = kwargs.get('top_k', self.config.top_k)
        
        # Handle different prompt input methods
        if phase and prompt_name and data:
            try:
                prompt = render_prompt(phase, prompt_name, data)
            except Exception as e:
                print(f"Error loading prompt template: {e}")
                prompt = str(prompt_name)
        elif data and not phase and not prompt_name:
            prompt = str(data)
        elif prompt_name and not phase:
            prompt = str(prompt_name)
        else:
            prompt = str(prompt_name) if prompt_name else "Generate an image"
        
        ensure_directory(os.path.dirname(output_path))
        
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=prompt),
                ],
            ),
        ]
        
        generate_content_config = types.GenerateContentConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_output_tokens=self.config.max_output_tokens,
            response_modalities=["image", "text"],
            response_mime_type="text/plain",
        )
        
        try:
            response_text = ""
            image_saved = False
            image_path = None
            thumbnail_path = None
            
            for chunk in self.client.models.generate_content_stream(
                model=self.config.model,
                contents=contents,
                config=generate_content_config,
            ):
                if not chunk.candidates or not chunk.candidates[0].content or not chunk.candidates[0].content.parts:
                    continue
                
                for part in chunk.candidates[0].content.parts:
                    if hasattr(part, 'inline_data') and part.inline_data:
                        inline_data = part.inline_data
                        file_extension = mimetypes.guess_extension(inline_data.mime_type) or ".png"
                        
                        full_path = f"{output_path}{file_extension}"
                        self.save_binary_file(full_path, inline_data.data)
                        
                        try:
                            img = Image.open(io.BytesIO(inline_data.data))
                            thumb_path = f"{output_path}_thumb{file_extension}"
                            img.thumbnail((200, 200))
                            img.save(thumb_path)
                            thumbnail_path = thumb_path
                        except Exception as e:
                            print(f"Error creating thumbnail: {e}")
                        
                        image_saved = True
                        image_path = full_path
                        
                        print(f"File of mime type {inline_data.mime_type} saved to: {full_path}")
                    
                    elif hasattr(part, 'text'):
                        text = part.text or ""
                        response_text += text
                        print(text, end="")
                
                if hasattr(chunk, 'text'):
                    text = chunk.text or ""
                    response_text += text
                    print(text, end="")
            
            return {
                "success": image_saved,
                "image_path": image_path,
                "thumbnail_path": thumbnail_path,
                "response_text": response_text
            }
        
        except Exception as e:
            print(f"Error generating image: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def direct_image_generation(self, prompt, output_path):
        """Generate an image directly without using a template."""
        return self.generate_image(None, None, prompt, output_path)