import pystache

def load_prompt(phase, prompt_name):
    """Load a prompt template from the prompts directory."""
    prompt_path = f"{phase}/prompts/{prompt_name}.mustache"
    with open(prompt_path, 'r') as f:
        return f.read()

def render_prompt(phase, prompt_name, data):
    """Render a prompt template with the given data."""
    template = load_prompt(phase, prompt_name)
    return pystache.render(template, data)