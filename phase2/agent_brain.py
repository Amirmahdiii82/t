import os
from typing import Dict, Any
from phase2.memory_manager import MemoryManager
from phase2.conscious_processor import ConsciousProcessor
from phase2.unconscious_processor import UnconsciousProcessor
from phase2.dream_generator import DreamGenerator

class AgentBrain:
    """
    Central orchestrator for psychoanalytically-informed AI agent.
    
    Integrates conscious and unconscious processing through memory-based
    signifier activation and neurochemical emotion simulation.
    """
    
    def __init__(self, agent_name: str, base_path: str = "base_agents"):
        self.agent_name = agent_name
        self.base_path = base_path
        self.agent_path = os.path.join(base_path, agent_name)
        
        # Initialize core components
        self.memory_manager = MemoryManager(agent_name, base_path)
        self.conscious_processor = ConsciousProcessor(agent_name, self.memory_manager)
        self.unconscious_processor = UnconsciousProcessor(agent_name, self.memory_manager)
        self.dream_generator = DreamGenerator(agent_name, self.memory_manager, base_path)
        
        # Set bidirectional references
        self.conscious_processor.set_unconscious_processor(self.unconscious_processor)
        
        self.state = {
            "mode": "wake",
            "is_active": True,
            "session_data": {
                "start_time": None,
                "interactions": 0,
                "dreams_generated": 0
            }
        }
    
    def process_message(self, user_input: str, context: str = "dialogue") -> Dict[str, Any]:
        """Process user message through integrated conscious-unconscious dynamics."""
        if self.state["mode"] != "wake":
            return {
                "error": "Agent is sleeping",
                "agent": self.agent_name,
                "response": "I am currently sleeping. Please wake me up first.",
                "mode": "sleep"
            }
        
        try:
            # Minimal debug output for production
            print(f"=== Processing Message for {self.agent_name} ===")
            print(f"User: '{user_input[:50]}...'")
            
            # Add input to memory and update emotional state
            self.memory_manager.add_to_short_term_memory(user_input, context=context)
            
            # Process through unconscious first (signifier activation)
            print("--- Unconscious Processing ---")
            unconscious_influence = self.unconscious_processor.process_input(user_input, context)
            
            # Process through conscious with unconscious influence
            print("--- Conscious Processing ---")
            response = self.conscious_processor.process_input(user_input, unconscious_influence)
            
            # Add response to memory
            self.memory_manager.add_to_short_term_memory(response, context="agent_response")
            
            self.state["session_data"]["interactions"] += 1
            
            # Clean summary for output
            active_signifiers = unconscious_influence.get('active_signifiers', [])
            chain_activations = unconscious_influence.get('chain_activations', {}).get('activated_chains', [])
            
            print("--- Response Generated ---")
            print(f"Agent: '{response[:100]}...'")
            print(f"Unconscious influence: {len(active_signifiers)} signifiers, {len(chain_activations)} chains")
            
            # Calculate unconscious activity summary
            object_a_activity = sum(1 for s in active_signifiers 
                                  if 'object_a' in s.get('activation_type', ''))
            symptom_activity = unconscious_influence.get('jouissance_effects', {}).get('symptom_activation', False)
            
            print(f"Object a proximity: {object_a_activity/max(len(active_signifiers), 1):.2f}")
            print(f"Symptom activation: {'Yes' if symptom_activity else 'No'}")
            
            return {
                "agent": self.agent_name,
                "response": response,
                "mode": self.state["mode"],
                "unconscious_state": {
                    "active_signifiers": [s['signifier'] for s in active_signifiers],
                    "discourse_position": unconscious_influence.get('discourse_position', {}),
                    "emotional_state": self.memory_manager.get_emotional_state().get('emotion_category', 'neutral'),
                    "object_a_proximity": object_a_activity / max(len(active_signifiers), 1),
                    "symptom_activation": symptom_activity
                },
                "session_stats": {
                    "interactions": self.state["session_data"]["interactions"]
                }
            }
            
        except Exception as e:
            print(f"❌ Error processing message: {e}")
            return {
                "error": str(e),
                "agent": self.agent_name,
                "response": "I'm experiencing some difficulty processing that.",
                "mode": self.state["mode"]
            }
    
    def switch_mode(self, new_mode: str) -> Dict[str, Any]:
        """Switch between wake and sleep modes."""
        if new_mode not in ["wake", "sleep"]:
            return {"error": f"Invalid mode: {new_mode}", "current_mode": self.state["mode"]}
        
        old_mode = self.state["mode"]
        self.state["mode"] = new_mode
        
        response = {
            "agent": self.agent_name,
            "previous_mode": old_mode,
            "current_mode": new_mode,
            "message": f"{self.agent_name} is now in {new_mode} mode"
        }
        
        if new_mode == "sleep":
            response["message"] += ". Ready to generate dreams."
        
        return response
    
    def generate_dream(self) -> Dict[str, Any]:
        """Generate psychoanalytically authentic dream from activated signifiers."""
        if self.state["mode"] != "sleep":
            return {
                "error": "Must be in sleep mode to dream",
                "agent": self.agent_name,
                "current_mode": self.state["mode"]
            }
        
        try:
            dream = self.dream_generator.generate_dream("sleep")
            
            if dream:
                self.state["session_data"]["dreams_generated"] += 1
                return {
                    "agent": self.agent_name,
                    "dream": dream,
                    "mode": "sleep",
                    "dream_count": self.state["session_data"]["dreams_generated"]
                }
            else:
                return {"error": "Failed to generate dream", "agent": self.agent_name, "mode": "sleep"}
                
        except Exception as e:
            print(f"❌ Error generating dream: {e}")
            return {"error": str(e), "agent": self.agent_name, "mode": "sleep"}
    
    def get_state(self) -> Dict[str, Any]:
        """Get comprehensive agent state for analysis."""
        memory_stats = self.memory_manager.get_memory_stats()
        
        return {
            "agent": self.agent_name,
            "mode": self.state["mode"],
            "is_active": self.state["is_active"],
            "memory_statistics": memory_stats,
            "session_data": self.state["session_data"]
        }
    
    def save_state(self) -> None:
        """Persist agent state to disk."""
        try:
            self.memory_manager.save_state()
            print(f"   Neurochemical and unconscious influence state saved")
            print(f"✅ {self.agent_name} state saved successfully")
        except Exception as e:
            print(f"❌ Error saving {self.agent_name} state: {e}")
    
    def shutdown(self) -> None:
        """Gracefully shutdown agent with state persistence."""
        self.save_state()
        self.state["is_active"] = False