import os
from typing import Dict, Any
from phase2.memory_manager import MemoryManager
from phase2.conscious_processor import ConsciousProcessor
from phase2.unconscious_processor import UnconsciousProcessor
from phase2.dream_generator import DreamGenerator
from phase2.interaction_handler import InteractionHandler

class AgentBrain:
    def __init__(self, agent_name: str, base_path: str = "base_agents"):
        """Initialize the agent brain with all cognitive components."""
        self.agent_name = agent_name
        self.base_path = base_path
        self.agent_path = os.path.join(base_path, agent_name)
        
        print(f"Agent path: {self.agent_path}")
        
        # Initialize core components
        try:
            # Memory Manager - central hub for all memory operations
            self.memory_manager = MemoryManager(agent_name, base_path)
            
            # Conscious Processor - handles rational thought and dialogue
            self.conscious_processor = ConsciousProcessor(agent_name, self.memory_manager, base_path)
            
            # Unconscious Processor - handles symbolic and emotional processing
            self.unconscious_processor = UnconsciousProcessor(agent_name, self.memory_manager, base_path)
            
            # Dream Generator - creates dreams during sleep mode
            self.dream_generator = DreamGenerator(agent_name, self.memory_manager, base_path)
            
            # Interaction Handler - manages user interactions
            self.interaction_handler = InteractionHandler(agent_name, self.memory_manager, base_path)
            
            # Agent state
            self.current_mode = "wake"  # "wake" or "sleep"
            self.is_active = True
            
            print(f"✅ Agent Brain initialized successfully for {agent_name}")
            
        except Exception as e:
            print(f"❌ Error initializing Agent Brain: {e}")
            raise
    
    def wake_mode(self) -> None:
        """Enter wake mode - ready for interactions."""
        self.current_mode = "wake"
        print(f"{self.agent_name} is now in wake mode")
    
    def sleep_mode(self) -> None:
        """Enter sleep mode - generate dreams."""
        self.current_mode = "sleep"
        print(f"{self.agent_name} is now in sleep mode")
        
        # Generate a dream
        try:
            dream = self.dream_generator.generate_dream()
            if dream:
                print(f"Dream generated: {dream['id']}")
                return dream
            else:
                print("Failed to generate dream")
                return None
        except Exception as e:
            print(f"Error in sleep mode: {e}")
            return None
    
    def process_input(self, user_input: str, context: str = "dialogue") -> str:
        """Process user input and generate response."""
        if self.current_mode != "wake":
            return "I am currently sleeping. Please wake me up first."
        
        try:
            # Add input to short-term memory
            self.memory_manager.add_to_short_term_memory(user_input, context)
            
            # Process input through conscious processor
            conscious_response = self.conscious_processor.process_input(user_input, context)
            
            # Process through unconscious processor for emotional coloring
            unconscious_influence = self.unconscious_processor.process_input(user_input, context)
            
            # Generate final response through interaction handler
            final_response = self.interaction_handler.generate_response(
                user_input, conscious_response, unconscious_influence, context
            )
            
            # Add response to short-term memory
            self.memory_manager.add_to_short_term_memory(final_response, "response")
            
            return final_response
            
        except Exception as e:
            print(f"Error processing input: {e}")
            return f"I'm having trouble processing that right now. Error: {str(e)}"
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        try:
            memory_stats = self.memory_manager.get_memory_stats()
            emotional_state = self.memory_manager.get_emotional_state()
            
            return {
                "agent_name": self.agent_name,
                "current_mode": self.current_mode,
                "is_active": self.is_active,
                "memory_stats": memory_stats,
                "emotional_state": emotional_state,
                "recent_dreams": len(self.dream_generator.get_recent_dreams(5))
            }
        except Exception as e:
            print(f"Error getting agent status: {e}")
            return {
                "agent_name": self.agent_name,
                "current_mode": self.current_mode,
                "is_active": self.is_active,
                "error": str(e)
            }
    
    def save_state(self) -> None:
        """Save current agent state."""
        try:
            self.memory_manager.save_state()
            print(f"Agent state saved for {self.agent_name}")
        except Exception as e:
            print(f"Error saving agent state: {e}")
    
    def shutdown(self) -> None:
        """Shutdown the agent gracefully."""
        try:
            self.save_state()
            self.is_active = False
            print(f"Agent {self.agent_name} shutdown gracefully")
        except Exception as e:
            print(f"Error during agent shutdown: {e}")