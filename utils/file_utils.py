import os
import json
from datetime import datetime

def ensure_directory(directory):
    """Ensure a directory exists, creating it if necessary."""
    os.makedirs(directory, exist_ok=True)

class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder that converts datetime objects to ISO format strings."""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

def save_json(data, file_path):
    """Save data to a JSON file."""
    ensure_directory(os.path.dirname(file_path))
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2, cls=DateTimeEncoder)

def load_json(file_path):
    """Load data from a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)
    
def get_timestamp():
    """Get a formatted timestamp string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")