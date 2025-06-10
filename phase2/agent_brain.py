import os
from typing import Dict, Any
from phase2.memory_manager import MemoryManager
from phase2.conscious_processor import ConsciousProcessor
from phase2.unconscious_processor import UnconsciousProcessor
from phase2.dream_generator import DreamGenerator
from utils.psychic_integration import PsychicIntegration

class AgentBrain:
    """Agent brain with full psychoanalytic integration."""
    
    def __init__(self, agent_name: str, base_path: str = "base_agents"):
        self.agent_name = agent_name
        self.base_path = base_path
        self.agent_path = os.path.join(base_path, agent_name)
        
        print(f"Initializing Agent Brain for {agent_name}...")
        
        try:
            self.memory_manager = MemoryManager(agent_name, base_path)
            self.conscious_processor = ConsciousProcessor(agent_name, self.memory_manager)
            self.dream_generator = DreamGenerator(agent_name, self.memory_manager, base_path)
            self.unconscious_processor = UnconsciousProcessor(
                agent_name, self.memory_manager, self.dream_generator
            )
            self.psychic_integration = PsychicIntegration(
                self.conscious_processor,
                self.unconscious_processor,
                self.memory_manager
            )
            self.state = {
                "mode": "wake", "is_active": True,
                "session_data": {"start_time": None, "interactions": 0, "dreams_generated": 0}
            }
            print(f"✅ Agent Brain initialized successfully for {agent_name}")
            
        except Exception as e:
            print(f"❌ Error initializing Agent Brain: {e}")
            raise
    
    def process_message(self, user_input: str, context: str = "dialogue") -> Dict[str, Any]:
        """Process user message through full psychoanalytic apparatus."""
        if self.state["mode"] != "wake":
            return {
                "error": "Agent is sleeping", "agent": self.agent_name,
                "response": "I am currently sleeping. Please wake me up first.", "mode": "sleep"
            }
        
        try:
            # --- FIX: Explicitly add user input to short-term memory first. ---
            # This guarantees every interaction is available to the dream generator.
            self.memory_manager.add_to_short_term_memory(user_input, context=context)
            # --- END FIX ---

            # Now, process through psychic integration as before
            result = self.psychic_integration.process_interaction(user_input, context)
            
            self.state["session_data"]["interactions"] += 1
            
            return {
                "agent": self.agent_name, "response": result["response"],
                "mode": self.state["mode"], "psychic_state": result["psychic_state"],
                "interpretation_hints": result.get("interpretation_hints", []),
                "session_stats": {
                    "interactions": self.state["session_data"]["interactions"],
                    "energy_distribution": result["psychic_state"]["energy_distribution"]
                }
            }
            
        except Exception as e:
            print(f"Error processing message: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "error": str(e), "agent": self.agent_name,
                "response": "I'm having difficulty processing that right now.", "mode": self.state["mode"]
            }
    
    def switch_mode(self, new_mode: str) -> Dict[str, Any]:
        """Switch between wake and sleep modes."""
        if new_mode not in ["wake", "sleep"]:
            return {"error": f"Invalid mode: {new_mode}", "current_mode": self.state["mode"]}
        
        old_mode = self.state["mode"]
        self.state["mode"] = new_mode
        
        response = {
            "agent": self.agent_name, "previous_mode": old_mode,
            "current_mode": new_mode, "message": f"{self.agent_name} is now in {new_mode} mode"
        }
        
        if new_mode == "sleep":
            response["message"] += ". Ready to generate dreams."
        
        return response
    
    def generate_dream(self) -> Dict[str, Any]:
        """Generate a dream (must be in sleep mode)."""
        if self.state["mode"] != "sleep":
            return {
                "error": "Must be in sleep mode to dream", "agent": self.agent_name, "current_mode": self.state["mode"]
            }
        
        try:
            dream = self.dream_generator.generate_dream("sleep")
            
            if dream:
                self.state["session_data"]["dreams_generated"] += 1
                return {
                    "agent": self.agent_name, "dream": dream,
                    "mode": "sleep", "dream_count": self.state["session_data"]["dreams_generated"]
                }
            else:
                return {"error": "Failed to generate dream", "agent": self.agent_name, "mode": "sleep"}
                
        except Exception as e:
            print(f"Error generating dream: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e), "agent": self.agent_name, "mode": "sleep"}
    
    def get_state(self) -> Dict[str, Any]:
        """Get comprehensive agent state."""
        try:
            memory_stats = self.memory_manager.get_memory_stats()
            psychic_analysis = self.psychic_integration.get_session_analysis()
            
            return {
                "agent": self.agent_name, "mode": self.state["mode"], "is_active": self.state["is_active"],
                "memory": memory_stats, "psychic_analysis": psychic_analysis, "session": self.state["session_data"]
            }
            
        except Exception as e:
            print(f"Error getting state: {e}")
            return {"agent": self.agent_name, "mode": self.state["mode"], "error": str(e)}
    
    def save_state(self) -> None:
        """Save current agent state."""
        try:
            self.memory_manager.save_state()
            print(f"Agent state saved for {self.agent_name}")
        except Exception as e:
            print(f"Error saving state: {e}")
    
    def shutdown(self) -> None:
        """Gracefully shutdown the agent."""
        self.save_state()
        self.state["is_active"] = False
        print(f"Agent {self.agent_name} shutdown complete")
