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
    def __init__(self):
        """Initialize the agent extractor with all necessary components."""
        print("Initializing Agent Extractor...")
        
        try:
            self.llm = LLMInterface()
            print("✓ LLM Interface initialized")
        except Exception as e:
            print(f"❌ Error initializing LLM Interface: {e}")
            raise
        
        try:
            self.vlm = VLMInterface()
            print("✓ VLM Interface initialized")
        except Exception as e:
            print(f"❌ Error initializing VLM Interface: {e}")
            raise
        
        try:
            self.conscious_builder = ConsciousBuilder(self.llm)
            print("✓ Conscious Builder initialized")
        except Exception as e:
            print(f"❌ Error initializing Conscious Builder: {e}")
            raise
        
        try:
            self.unconscious_builder = UnconsciousBuilder(self.vlm)
            print("✓ Unconscious Builder initialized")
        except Exception as e:
            print(f"❌ Error initializing Unconscious Builder: {e}")
            raise
        
        print("🎉 Agent Extractor fully initialized!")
    
    def extract_agent_from_dreams(self, dream_file_path):
        """Extract an agent from a dream dataset with comprehensive error handling."""
        print(f"\n{'='*60}")
        print(f"STARTING AGENT EXTRACTION FROM: {dream_file_path}")
        print(f"{'='*60}")
        
        try:
            # Step 1: Load and validate dream data
            print("\nStep 1: Loading dream data...")
            if not os.path.exists(dream_file_path):
                raise FileNotFoundError(f"Dream file not found: {dream_file_path}")
            
            dream_data = load_json(dream_file_path)
            print(f"✓ Dream data loaded: {len(str(dream_data))} characters")
            
            # Step 2: Determine agent name
            print("\nStep 2: Determining agent name...")
            agent_name = self._determine_agent_name(dream_file_path, dream_data)
            print(f"✓ Agent name determined: {agent_name}")
            
            # Step 3: Create agent directory structure
            print("\nStep 3: Creating agent directory structure...")
            agent_dir = f"base_agents/{agent_name}"
            ensure_directory(agent_dir)
            
            # Create subdirectories
            subdirs = ["dreams", "signifier_images", "vector_db"]
            for subdir in subdirs:
                ensure_directory(f"{agent_dir}/{subdir}")
            
            print(f"✓ Agent directory structure created: {agent_dir}")
            
            # Step 4: Initialize agent data
            print("\nStep 4: Initializing agent data...")
            agent = {
                "id": dream_data.get("id", f"{agent_name}_{int(time.time())}"),
                "name": agent_name,
                "created_at": dream_data.get("created_at", get_timestamp()),
                "last_updated": get_timestamp(),
                "extraction_status": "in_progress"
            }
            print(f"✓ Agent data initialized")
            
            # Step 5: Build conscious memory
            print(f"\nStep 5: Building conscious memory for {agent_name}...")
            try:
                conscious = self.conscious_builder.build_conscious_memory(dream_data, agent_name, agent_dir)
                agent["conscious"] = conscious
                agent["conscious_extraction_status"] = "completed"
                print(f"✓ Conscious memory extraction completed")
            except Exception as e:
                print(f"❌ Error in conscious memory extraction: {e}")
                agent["conscious_extraction_status"] = "failed"
                agent["conscious_extraction_error"] = str(e)
                # Continue with unconscious extraction even if conscious fails
            
            # Step 6: Build unconscious memory
            print(f"\nStep 6: Building unconscious memory for {agent_name}...")
            try:
                unconscious = self.unconscious_builder.build_unconscious_memory(dream_data, agent_name, agent_dir)
                agent["unconscious"] = unconscious
                agent["unconscious_extraction_status"] = "completed"
                print(f"✓ Unconscious memory extraction completed")
            except Exception as e:
                print(f"❌ Error in unconscious memory extraction: {e}")
                agent["unconscious_extraction_status"] = "failed"
                agent["unconscious_extraction_error"] = str(e)
            
            # Step 7: Finalize and save agent data
            print(f"\nStep 7: Finalizing agent data...")
            agent["extraction_status"] = "completed"
            agent["last_updated"] = get_timestamp()
            
            # Save the complete agent data
            agent_file_path = f"{agent_dir}/base_{agent_name}.json"
            save_json(agent, agent_file_path)
            print(f"✓ Agent data saved to {agent_file_path}")
            
            # Step 8: Generate extraction summary
            self._generate_extraction_summary(agent, agent_dir)
            
            print(f"\n{'='*60}")
            print(f"AGENT EXTRACTION COMPLETED FOR: {agent_name}")
            print(f"{'='*60}")
            
            return agent
            
        except Exception as e:
            print(f"\n❌ CRITICAL ERROR during agent extraction: {e}")
            import traceback
            traceback.print_exc()
            
            # Try to save partial results
            try:
                if 'agent' in locals() and 'agent_dir' in locals():
                    agent["extraction_status"] = "failed"
                    agent["extraction_error"] = str(e)
                    agent["last_updated"] = get_timestamp()
                    
                    error_file_path = f"{agent_dir}/extraction_error.json"
                    save_json(agent, error_file_path)
                    print(f"Partial results saved to {error_file_path}")
            except:
                pass
            
            raise

    def _determine_agent_name(self, dream_file_path, dream_data):
        """Determine the agent name from file path and dream data."""
        agent_name = None
        
        # Method 1: Check if name is explicitly in dream data
        if isinstance(dream_data, dict) and "name" in dream_data:
            agent_name = dream_data["name"]
            print(f"Name found in dream data: {agent_name}")
        
        # Method 2: Extract from filename
        if not agent_name:
            file_basename = os.path.basename(dream_file_path)
            agent_name = os.path.splitext(file_basename)[0].capitalize()
            print(f"Name extracted from filename: {agent_name}")
        
        # Method 3: Search in dream content
        if not agent_name or agent_name.lower() in ['data', 'dreams', 'dataset']:
            try:
                dreams_text = json.dumps(dream_data)
                
                # Look for name patterns in the JSON
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
                            print(f"Name found in content: {agent_name}")
                            break
            except Exception as e:
                print(f"Error searching for name in content: {e}")
        
        # Fallback: Use filename or default
        if not agent_name:
            file_basename = os.path.basename(dream_file_path)
            agent_name = os.path.splitext(file_basename)[0].capitalize()
            if not agent_name:
                agent_name = "UnknownAgent"
        
        # Clean the agent name (now with proper re import)
        agent_name = re.sub(r'[^\w\s-]', '', agent_name).strip()
        if not agent_name:
            agent_name = "UnknownAgent"
        
        return agent_name

    def _generate_extraction_summary(self, agent, agent_dir):
        """Generate a summary of the extraction process."""
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
        
        # Print summary
        print(f"\n--- EXTRACTION SUMMARY ---")
        print(f"Agent: {summary['agent_name']}")
        print(f"Date: {summary['extraction_date']}")
        print(f"Conscious Status: {summary['conscious_status']}")
        print(f"Unconscious Status: {summary['unconscious_status']}")
        print(f"Overall Status: {summary['overall_status']}")
        
        if "conscious_stats" in summary:
            stats = summary["conscious_stats"]
            print(f"Conscious Stats: {stats['memories_extracted']} memories, {stats['relationships_extracted']} relationships")
        
        if "unconscious_stats" in summary:
            stats = summary["unconscious_stats"]
            print(f"Unconscious Stats: {stats['signifiers_extracted']} signifiers, {stats['visualizations_created']} visualizations")
        
        print(f"Summary saved to: {summary_path}")

    def extract_agent_data(self, agent_name):
        """Extract agent data by name (for compatibility with other scripts)."""
        dataset_path = f"dataset/{agent_name.lower()}.json"
        
        if not os.path.exists(dataset_path):
            print(f"❌ Dataset file not found: {dataset_path}")
            return None
        
        try:
            return self.extract_agent_from_dreams(dataset_path)
        except Exception as e:
            print(f"❌ Error extracting agent {agent_name}: {e}")
            return None