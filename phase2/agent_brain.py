import os
from typing import Dict, Any
from phase2.memory_manager import MemoryManager
from phase2.conscious_processor import ConsciousProcessor
from phase2.unconscious_processor import UnconsciousProcessor
from phase2.dream_generator import DreamGenerator

class AgentBrain:
    """
    Central orchestrator for psychoanalytically-informed AI agent.
    
    Fully integrates conscious and unconscious processing using ALL extracted
    unconscious data: signifying chains, object_a dynamics, symptom patterns,
    repressed content, and neurochemical responses.
    """
    
    def __init__(self, agent_name: str, base_path: str = "base_agents"):
        self.agent_name = agent_name
        self.base_path = base_path
        self.agent_path = os.path.join(base_path, agent_name)
        
        print(f"\n=== Initializing {agent_name} with Full Psychoanalytic Integration ===")
        
        # Initialize core components
        print("1. Initializing Memory Manager...")
        self.memory_manager = MemoryManager(agent_name, base_path)
        
        print("2. Initializing Unconscious Processor...")
        self.unconscious_processor = UnconsciousProcessor(agent_name, self.memory_manager)
        
        print("3. Initializing Conscious Processor...")
        self.conscious_processor = ConsciousProcessor(agent_name, self.memory_manager)
        
        print("4. Initializing Dream Generator...")
        self.dream_generator = DreamGenerator(agent_name, self.memory_manager, base_path)
        
        # Set bidirectional references
        self.conscious_processor.set_unconscious_processor(self.unconscious_processor)
        
        self.state = {
            "mode": "wake",
            "is_active": True,
            "session_data": {
                "start_time": None,
                "interactions": 0,
                "dreams_generated": 0,
                "unconscious_activations": 0,
                "signifiers_activated": set(),
                "chains_activated": set()
            }
        }
        
        print(f"✅ {agent_name} fully initialized with psychoanalytic architecture")
        self._print_agent_summary()
    
    def _print_agent_summary(self):
        """Print summary of agent's psychoanalytic structure."""
        try:
            # Get unconscious data summary
            unconscious_data = self.memory_manager.unconscious_memory
            conscious_data = self.memory_manager.conscious_memory
            
            print(f"\n📊 Agent Summary:")
            print(f"   Conscious: {len(conscious_data.get('memories', []))} memories, {len(conscious_data.get('relationships', []))} relationships")
            print(f"   Unconscious: {len(unconscious_data.get('signifiers', []))} signifiers, {len(unconscious_data.get('signifying_chains', []))} chains")
            
            # Print key signifiers
            signifiers = unconscious_data.get('signifiers', [])
            if signifiers:
                key_signifiers = [sig.get('name', 'Unknown') for sig in signifiers[:3] if isinstance(sig, dict)]
                print(f"   Key Signifiers: {', '.join(key_signifiers)}")
            
            # Print repressed content
            repressed = [sig.get('name', 'Unknown') for sig in signifiers if isinstance(sig, dict) and sig.get('repressed', False)]
            if repressed:
                print(f"   Repressed Content: {', '.join(repressed)}")
            
            # Print object_a
            object_a = unconscious_data.get('object_a', {})
            if object_a.get('description'):
                print(f"   Object a: {object_a['description'][:50]}...")
            
            # Print symptom
            symptom = unconscious_data.get('symptom', {})
            if symptom.get('description'):
                print(f"   Symptom: {symptom['description'][:50]}...")
            
        except Exception as e:
            print(f"   Error getting summary: {e}")
    
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
            print(f"\n=== Processing Message for {self.agent_name} ===")
            print(f"User: '{user_input[:100]}...'")
            
            # Add input to memory and update emotional state
            self.memory_manager.add_to_short_term_memory(user_input, context=context)
            
            # 1. Process through unconscious FIRST (this triggers neurochemical responses)
            print("\n--- Unconscious Processing ---")
            unconscious_influence = self.unconscious_processor.process_input(user_input, context)
            
            # Track unconscious activation stats
            self.state["session_data"]["unconscious_activations"] += 1
            
            # Track activated signifiers and chains
            active_signifiers = unconscious_influence.get('active_signifiers', [])
            active_chains = unconscious_influence.get('activated_chains', [])
            
            for sig in active_signifiers:
                self.state["session_data"]["signifiers_activated"].add(sig.get('signifier', ''))
            
            for chain in active_chains:
                self.state["session_data"]["chains_activated"].add(chain.get('name', ''))
            
            # 2. Apply unconscious triggers to neurochemical system
            self._apply_unconscious_to_neurochemistry(unconscious_influence)
            
            # 3. Process through conscious with unconscious influence
            print("\n--- Conscious Processing ---")
            response = self.conscious_processor.process_input(user_input, unconscious_influence)
            
            # Add response to memory
            self.memory_manager.add_to_short_term_memory(response, context="agent_response")
            
            self.state["session_data"]["interactions"] += 1
            
            # Create comprehensive response data
            response_data = {
                "agent": self.agent_name,
                "response": response,
                "mode": self.state["mode"],
                "unconscious_state": {
                    "active_signifiers": [s.get('signifier', '') for s in active_signifiers],
                    "activated_chains": [c.get('name', '') for c in active_chains],
                    "repressed_emerging": [r.get('signifier', '') for r in unconscious_influence.get('repressed_returns', [])],
                    "object_a_proximity": unconscious_influence.get('object_a_effects', {}).get('proximity_level', 0.0),
                    "symptom_activation": unconscious_influence.get('symptom_effects', {}).get('activation_level', 0.0),
                    "discourse_position": unconscious_influence.get('discourse_position', {}),
                    "neurochemical_triggers": len(unconscious_influence.get('active_signifiers', []))
                },
                "emotional_state": {
                    "emotion_category": self.memory_manager.get_emotional_state().get('emotion_category', 'neutral'),
                    "neurochemical_influenced": self.memory_manager.get_emotional_state().get('unconscious_influence', False),
                    "pad_values": {
                        "pleasure": self.memory_manager.get_emotional_state().get('pleasure', 0.0),
                        "arousal": self.memory_manager.get_emotional_state().get('arousal', 0.0),
                        "dominance": self.memory_manager.get_emotional_state().get('dominance', 0.0)
                    }
                },
                "session_stats": {
                    "interactions": self.state["session_data"]["interactions"],
                    "unconscious_activations": self.state["session_data"]["unconscious_activations"],
                    "unique_signifiers_activated": len(self.state["session_data"]["signifiers_activated"]),
                    "unique_chains_activated": len(self.state["session_data"]["chains_activated"])
                }
            }
            
            print(f"\n--- Response Generated ---")
            print(f"Agent: '{response[:100]}...'")
            print(f"Unconscious influence: {len(active_signifiers)} signifiers, {len(active_chains)} chains")
            print(f"Object a proximity: {unconscious_influence.get('object_a_effects', {}).get('proximity_level', 0.0):.2f}")
            print(f"Symptom activation: {unconscious_influence.get('symptom_effects', {}).get('activation_level', 0.0):.2f}")
            
            return response_data
            
        except Exception as e:
            print(f"Error in message processing: {e}")
            import traceback
            traceback.print_exc()
            return {
                "error": str(e),
                "agent": self.agent_name,
                "response": "I'm experiencing some difficulty processing that.",
                "mode": self.state["mode"]
            }
    
    def _apply_unconscious_to_neurochemistry(self, unconscious_influence: Dict[str, Any]):
        """Apply unconscious dynamics to neurochemical system."""
        neuroproxy = self.memory_manager.neuroproxy_engine
        
        # 1. Object a effects
        object_a_effects = unconscious_influence.get('object_a_effects', {})
        proximity = object_a_effects.get('proximity_level', 0.0)
        desire_direction = object_a_effects.get('desire_direction', 'neutral')
        
        if proximity > 0.3:
            if desire_direction == 'seeking_substitute':
                neuroproxy.trigger_object_a_seeking(intensity=proximity)
            elif desire_direction == 'circling_void':
                neuroproxy.trigger_object_a_circling(intensity=proximity)
        
        # 2. Symptom activation
        symptom_effects = unconscious_influence.get('symptom_effects', {})
        symptom_level = symptom_effects.get('activation_level', 0.0)
        
        if symptom_level > 0.3:
            # Determine symptom type from symptom effects
            if any('vulnerability' in pattern.lower() for pattern in symptom_effects.get('active_symptom_signifiers', [])):
                neuroproxy.trigger_symptom_activation("vulnerability", intensity=symptom_level)
            else:
                neuroproxy.trigger_symptom_activation("repetition", intensity=symptom_level)
        
        # 3. Repressed content returning
        repressed_returns = unconscious_influence.get('repressed_returns', [])
        if repressed_returns:
            max_intensity = max([r.get('return_strength', 0.0) for r in repressed_returns])
            if max_intensity > 0.3:
                primary_repressed = max(repressed_returns, key=lambda x: x.get('return_strength', 0.0))
                neuroproxy.trigger_repressed_return(primary_repressed.get('signifier', ''), intensity=max_intensity)
        
        # 4. Signifying chain activation
        activated_chains = unconscious_influence.get('activated_chains', [])
        for chain in activated_chains:
            if chain.get('activation_strength', 0.0) > 0.5:
                neuroproxy.trigger_signifying_chain_activation(
                    chain.get('name', ''),
                    chain.get('active_signifiers', []),
                    intensity=chain.get('activation_strength', 0.5)
                )
        
        # 5. Discourse position shift
        discourse_position = unconscious_influence.get('discourse_position', {})
        if discourse_position:
            primary_discourse = max(discourse_position.items(), key=lambda x: x[1])[0]
            discourse_strength = discourse_position[primary_discourse]
            if discourse_strength > 0.4:  # Only trigger if strong enough
                neuroproxy.trigger_discourse_position_shift(primary_discourse, intensity=discourse_strength)
    
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
            response["message"] += ". Ready to generate psychoanalytically authentic dreams."
        elif new_mode == "wake":
            response["message"] += ". Conscious and unconscious systems active."
        
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
            print(f"\n=== Generating Dream for {self.agent_name} ===")
            
            dream = self.dream_generator.generate_dream("sleep")
            
            if dream and not dream.get('error'):
                self.state["session_data"]["dreams_generated"] += 1
                
                return {
                    "agent": self.agent_name,
                    "dream": dream,
                    "mode": "sleep",
                    "dream_analysis": {
                        "activated_signifiers": dream.get('activated_signifiers', []),
                        "active_chains": dream.get('active_chains', []),
                        "emerging_repressed": dream.get('emerging_repressed', []),
                        "object_a_proximity": dream.get('object_a_proximity', 0.0),
                        "symptom_activation": dream.get('symptom_activation', False),
                        "generation_method": dream.get('generation_method', 'unknown')
                    },
                    "dream_count": self.state["session_data"]["dreams_generated"]
                }
            else:
                return {"error": "Failed to generate dream", "agent": self.agent_name, "mode": "sleep"}
                
        except Exception as e:
            print(f"Error generating dream: {e}")
            return {"error": str(e), "agent": self.agent_name, "mode": "sleep"}
    
    def get_state(self) -> Dict[str, Any]:
        """Get comprehensive agent state for analysis."""
        memory_stats = self.memory_manager.get_memory_stats()
        unconscious_stats = self.memory_manager.neuroproxy_engine.get_unconscious_influence_stats()
        
        return {
            "agent": self.agent_name,
            "mode": self.state["mode"],
            "is_active": self.state["is_active"],
            "memory_statistics": memory_stats,
            "unconscious_statistics": {
                "signifiers_available": len(self.unconscious_processor.signifiers),
                "chains_available": len(self.unconscious_processor.chains),
                "repressed_signifiers": len(self.unconscious_processor.repressed_signifiers),
                "object_a_manifestations": len(self.unconscious_processor.object_a_manifestations),
                "symptom_signifiers": len(self.unconscious_processor.symptom_signifiers)
            },
            "neurochemical_influence": unconscious_stats,
            "session_data": {
                **self.state["session_data"],
                "signifiers_activated": list(self.state["session_data"]["signifiers_activated"]),
                "chains_activated": list(self.state["session_data"]["chains_activated"])
            }
        }
    
    def get_unconscious_analysis(self) -> Dict[str, Any]:
        """Get detailed unconscious analysis."""
        unconscious_data = self.memory_manager.unconscious_memory
        
        analysis = {
            "agent_name": self.agent_name,
            "signifier_analysis": {
                "total_signifiers": len(unconscious_data.get('signifiers', [])),
                "repressed_signifiers": [
                    sig.get('name', 'Unknown') for sig in unconscious_data.get('signifiers', [])
                    if isinstance(sig, dict) and sig.get('repressed', False)
                ],
                "master_signifiers": [
                    sig.get('name', 'Unknown') for sig in unconscious_data.get('signifiers', [])
                    if isinstance(sig, dict) and any('identity' in sig.get('significance', '').lower() or 
                                                   'recurring' in sig.get('significance', '').lower() for _ in [1])
                ]
            },
            "chain_analysis": {
                "total_chains": len(unconscious_data.get('signifying_chains', [])),
                "chain_names": [chain.get('name', 'Unknown') for chain in unconscious_data.get('signifying_chains', [])
                               if isinstance(chain, dict)]
            },
            "object_a_analysis": {
                "description": unconscious_data.get('object_a', {}).get('description', 'Not identified'),
                "manifestations": unconscious_data.get('object_a', {}).get('manifestations', []),
                "void_manifestations": unconscious_data.get('object_a', {}).get('void_manifestations', [])
            },
            "symptom_analysis": {
                "description": unconscious_data.get('symptom', {}).get('description', 'Not identified'),
                "signifiers_involved": unconscious_data.get('symptom', {}).get('signifiers_involved', []),
                "jouissance_pattern": unconscious_data.get('symptom', {}).get('jouissance_pattern', '')
            },
            "discourse_positions": unconscious_data.get('structural_positions', {}),
            "current_emotional_state": self.memory_manager.get_emotional_state(),
            "unconscious_influence_stats": self.memory_manager.neuroproxy_engine.get_unconscious_influence_stats()
        }
        
        return analysis
    
    def save_state(self) -> None:
        """Persist agent state to disk."""
        try:
            self.memory_manager.save_state()
            print(f"✅ {self.agent_name} state saved successfully")
        except Exception as e:
            print(f"❌ Error saving {self.agent_name} state: {e}")
    
    def shutdown(self) -> None:
        """Gracefully shutdown agent with state persistence."""
        print(f"Shutting down {self.agent_name}...")
        self.save_state()
        self.state["is_active"] = False
        print(f"✅ {self.agent_name} shutdown complete")