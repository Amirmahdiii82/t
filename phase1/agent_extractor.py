import os
import time
import json
import re
from utils.file_utils import ensure_directory, load_json, save_json, get_timestamp
from interfaces.llm_interface import LLMInterface
from interfaces.vlm_interface import VLMInterface
from phase1.conscious_builder import ConsciousBuilder
from phase1.unconscious_builder import UnconsciousBuilder

class AgentExtractor:
    """
    Extracts psychoanalytic AI agents from dream datasets.
    
    Transforms raw dream narratives into structured conscious and unconscious
    memory systems following Lacanian psychoanalytic principles.
    """
    
    def __init__(self):
        self.llm = LLMInterface()
        self.vlm = VLMInterface()
        self.conscious_builder = ConsciousBuilder(self.llm)
        self.unconscious_builder = UnconsciousBuilder(self.vlm)
    
    def extract_agent_from_dreams(self, dream_file_path):
        """
        Extract complete agent from dream dataset.
        
        Args:
            dream_file_path: Path to JSON file containing dream data
            
        Returns:
            Dictionary containing complete agent data
        """
        print(f"Starting agent extraction from: {dream_file_path}")
        
        # Load dream data
        if not os.path.exists(dream_file_path):
            raise FileNotFoundError(f"Dream file not found: {dream_file_path}")
        
        dream_data = load_json(dream_file_path)
        print(f"Loaded dream data: {len(str(dream_data))} characters")
        
        # Determine agent name
        agent_name = self._determine_agent_name(dream_file_path, dream_data)
        print(f"Agent name: {agent_name}")
        
        # Create agent directory structure
        agent_dir = f"base_agents/{agent_name}"
        ensure_directory(agent_dir)
        
        subdirs = ["dreams", "signifier_images", "vector_db"]
        for subdir in subdirs:
            ensure_directory(f"{agent_dir}/{subdir}")
        
        # Initialize agent data
        agent = {
            "id": dream_data.get("id", f"{agent_name}_{int(time.time())}"),
            "name": agent_name,
            "created_at": dream_data.get("created_at", get_timestamp()),
            "last_updated": get_timestamp(),
            "extraction_status": "in_progress"
        }
        
        # Build conscious memory
        print("Building conscious memory...")
        try:
            conscious = self.conscious_builder.build_conscious_memory(dream_data, agent_name, agent_dir)
            agent["conscious"] = conscious
            agent["conscious_extraction_status"] = "completed"
            print("✅ Conscious memory extraction completed")
        except Exception as e:
            print(f"❌ Error in conscious memory extraction: {e}")
            agent["conscious_extraction_status"] = "failed"
            agent["conscious_extraction_error"] = str(e)
        
        # Build unconscious memory
        print("Building unconscious memory...")
        try:
            unconscious = self.unconscious_builder.build_unconscious_memory(dream_data, agent_name, agent_dir)
            agent["unconscious"] = unconscious
            agent["unconscious_extraction_status"] = "completed"
            print("✅ Unconscious memory extraction completed")
        except Exception as e:
            print(f"❌ Error in unconscious memory extraction: {e}")
            agent["unconscious_extraction_status"] = "failed"
            agent["unconscious_extraction_error"] = str(e)
        
        # Finalize agent data
        agent["extraction_status"] = "completed"
        agent["last_updated"] = get_timestamp()
        
        # Save complete agent data
        agent_file_path = f"{agent_dir}/base_{agent_name}.json"
        save_json(agent, agent_file_path)
        print(f"Agent data saved to {agent_file_path}")
        
        # Generate extraction summary
        self._generate_extraction_summary(agent, agent_dir)
        
        print(f"Agent extraction completed for: {agent_name}")
        return agent
    
    def _determine_agent_name(self, dream_file_path, dream_data):
        """Determine agent name from file path and dream data."""
        agent_name = None
        
        # Check if name is in dream data
        if isinstance(dream_data, dict) and "name" in dream_data:
            agent_name = dream_data["name"]
        
        # Extract from filename if not found
        if not agent_name:
            file_basename = os.path.basename(dream_file_path)
            agent_name = os.path.splitext(file_basename)[0].capitalize()
        
        # Search in dream content
        if not agent_name or agent_name.lower() in ['data', 'dreams', 'dataset']:
            try:
                dreams_text = json.dumps(dream_data)
                name_patterns = [
                    r'"name"\s*:\s*"([^"]+)"',
                    r'"dreamer"\s*:\s*"([^"]+)"',
                    r'"subject"\s*:\s*"([^"]+)"'
                ]
                
                for pattern in name_patterns:
                    match = re.search(pattern, dreams_text, re.IGNORECASE)
                    if match:
                        extracted_name = match.group(1).strip()
                        if extracted_name and len(extracted_name) > 1 and len(extracted_name) < 50:
                            agent_name = extracted_name.capitalize()
                            break
            except Exception:
                pass
        
        # Fallback
        if not agent_name:
            file_basename = os.path.basename(dream_file_path)
            agent_name = os.path.splitext(file_basename)[0].capitalize()
            if not agent_name:
                agent_name = "UnknownAgent"
        
        # Clean the agent name
        agent_name = re.sub(r'[^\w\s-]', '', agent_name).strip()
        if not agent_name:
            agent_name = "UnknownAgent"
        
        return agent_name
    
    def _generate_extraction_summary(self, agent, agent_dir):
        """Generate summary of extraction process."""
        summary = {
            "agent_name": agent.get("name", "Unknown"),
            "extraction_date": agent.get("last_updated", "Unknown"),
            "conscious_status": agent.get("conscious_extraction_status", "unknown"),
            "unconscious_status": agent.get("unconscious_extraction_status", "unknown"),
            "overall_status": agent.get("extraction_status", "unknown")
        }
        
        if "conscious" in agent:
            conscious = agent["conscious"]
            summary["conscious_stats"] = {
                "memories_extracted": len(conscious.get("memories", [])),
                "relationships_extracted": len(conscious.get("relationships", [])),
                "persona_extracted": bool(conscious.get("persona"))
            }
        
        if "unconscious" in agent:
            unconscious = agent["unconscious"]
            summary["unconscious_stats"] = {
                "signifiers_extracted": len(unconscious.get("signifiers", [])),
                "signifying_chains": len(unconscious.get("signifying_chains", [])),
                "visualizations_created": len(unconscious.get("visualizations", {}))
            }
        
        # Save summary
        summary_path = f"{agent_dir}/extraction_summary.json"
        save_json(summary, summary_path)
        
        print(f"Extraction Summary:")
        print(f"  Agent: {summary['agent_name']}")
        print(f"  Conscious Status: {summary['conscious_status']}")
        print(f"  Unconscious Status: {summary['unconscious_status']}")
        
        if "conscious_stats" in summary:
            stats = summary["conscious_stats"]
            print(f"  Memories: {stats['memories_extracted']}, Relationships: {stats['relationships_extracted']}")
        
        if "unconscious_stats" in summary:
            stats = summary["unconscious_stats"]
            print(f"  Signifiers: {stats['signifiers_extracted']}, Chains: {stats['signifying_chains']}")
    
    def extract_agent_data(self, agent_name):
        """Extract agent by name (compatibility method)."""
        dataset_path = f"dataset/{agent_name.lower()}.json"
        
        if not os.path.exists(dataset_path):
            print(f"Dataset file not found: {dataset_path}")
            return None
        
        try:
            return self.extract_agent_from_dreams(dataset_path)
        except Exception as e:
            print(f"Error extracting agent {agent_name}: {e}")
            return None