import os
import json
from datetime import datetime
from typing import Dict, List, Any
from utils.rag_system import RAGSystem
from utils.neuroproxy_engine import NeuroProxyEngine

class MemoryManager:
    def __init__(self, agent_name: str, base_path: str = "base_agents"):
        """Initialize memory manager for an agent."""
        self.agent_name = agent_name
        self.base_path = base_path
        
        if isinstance(base_path, str):
            self.agent_path = os.path.join(base_path, agent_name)
        else:
            self.agent_path = os.path.join("base_agents", agent_name)
                
        self.rag_system = RAGSystem(agent_name, self.base_path)
        self.neuroproxy_engine = NeuroProxyEngine(agent_name, self.base_path)
        
        # Load conscious and unconscious memory
        self.conscious_memory = self._load_conscious_memory()
        self.unconscious_memory = self._load_unconscious_memory()
        
        # --- FIX: Load short-term memory from the file where it was saved ---
        # Instead of self.short_term_memory = [], we now load the data that 
        # the save_state() function stored in conscious_memory.json.
        self.short_term_memory = self.conscious_memory.get("short_term_memory", [])
        # --- END FIX ---

        self.max_short_term_size = 10
        
        print(f"Memory Manager initialized for {agent_name}")
        # Add a more informative startup message
        if self.short_term_memory:
            print(f"✅ Loaded {len(self.short_term_memory)} entries from persistent short-term memory.")
        print(f"Loaded conscious memory: {len(self.conscious_memory.get('memories', []))} memories, {len(self.conscious_memory.get('relationships', []))} relationships")
        print(f"Loaded unconscious memory: {len(self.unconscious_memory.get('signifiers', []))} signifiers")
    
    def _load_conscious_memory(self) -> Dict[str, Any]:
        """Load conscious memory from file."""
        conscious_path = os.path.join(self.agent_path, "conscious_memory.json")
        
        try:
            if os.path.exists(conscious_path):
                with open(conscious_path, 'r') as f:
                    conscious_data = json.load(f)
                print(f"Loaded conscious memory from {conscious_path}")
                return conscious_data
            else:
                print(f"No conscious_memory.json found at {conscious_path}")
                return {"memories": [], "relationships": [], "persona": {}}
        except Exception as e:
            print(f"Error loading conscious memory: {e}")
            return {"memories": [], "relationships": [], "persona": {}}
    
    def _load_unconscious_memory(self) -> Dict[str, Any]:
        """Load unconscious memory from file."""
        unconscious_path = os.path.join(self.agent_path, "unconscious_memory.json")
        
        try:
            if os.path.exists(unconscious_path):
                with open(unconscious_path, 'r') as f:
                    unconscious_data = json.load(f)
                print(f"Loaded unconscious memory from {unconscious_path}")
                return unconscious_data
            else:
                print(f"No unconscious_memory.json found at {unconscious_path}")
                return {
                    "signifiers": [], "signifying_chains": [], "object_a": {}, "symptom": {},
                    "structural_positions": [], "fantasy_formula": "", "jouissance_economy": {}, "dream_work_patterns": {}
                }
        except Exception as e:
            print(f"Error loading unconscious memory: {e}")
            return {
                "signifiers": [], "signifying_chains": [], "object_a": {}, "symptom": {},
                "structural_positions": [], "fantasy_formula": "", "jouissance_economy": {}, "dream_work_patterns": {}
            }
    
    def add_to_short_term_memory(self, content: str, context: str = "interaction") -> None:
        """Add content to short-term memory."""
        memory_entry = {
            "content": content,
            "context": context,
            "timestamp": datetime.now().isoformat(),
            "emotional_state": self.neuroproxy_engine.get_current_state()
        }
        
        self.short_term_memory.append(memory_entry)
        
        if len(self.short_term_memory) > self.max_short_term_size:
            self.short_term_memory = self.short_term_memory[-self.max_short_term_size:]
        
        self.neuroproxy_engine.update_emotional_state(content, context)
    
    def retrieve_memories(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant memories using RAG system."""
        return self.rag_system.search_memories(query, n_results)
    
    def retrieve_relationships(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """Retrieve relevant relationships using RAG system."""
        return self.rag_system.search_relationships(query, n_results)
    
    def get_short_term_memory(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent short-term memory entries."""
        return self.short_term_memory[-limit:]
    
    def get_emotional_state(self) -> Dict[str, Any]:
        """Get current emotional state."""
        return self.neuroproxy_engine.get_current_state()
    
    def get_response_style_guidance(self) -> Dict[str, str]:
        """Get guidance for response style based on emotional state."""
        return self.neuroproxy_engine.get_response_style_guidance()
    
    def get_persona(self) -> Dict[str, Any]:
        """Get agent persona."""
        return self.conscious_memory.get("persona", {})
    
    def get_unconscious_signifiers(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get unconscious signifiers."""
        signifiers = self.unconscious_memory.get("signifiers", [])
        return signifiers[:limit]
    
    def get_signifying_chains(self) -> List[Dict[str, Any]]:
        """Get signifying chains."""
        return self.unconscious_memory.get("signifying_chains", [])
    
    def get_object_a(self) -> Dict[str, Any]:
        """Get object a configuration."""
        return self.unconscious_memory.get("object_a", {})
    
    def get_symptom(self) -> Dict[str, Any]:
        """Get symptom configuration."""
        return self.unconscious_memory.get("symptom", {})
    
    def get_structural_positions(self) -> List[Dict[str, Any]]:
        """Get structural positions."""
        return self.unconscious_memory.get("structural_positions", [])
    
    def consolidate_short_term_memory(self) -> None:
        """Consolidate short-term memory into long-term memory (for future implementation)."""
        pass
    
    def save_state(self) -> None:
        """Save current memory state."""
        try:
            self.conscious_memory["short_term_memory"] = self.short_term_memory
            self.conscious_memory["last_updated"] = datetime.now().isoformat()
            
            conscious_path = os.path.join(self.agent_path, "conscious_memory.json")
            with open(conscious_path, 'w') as f:
                json.dump(self.conscious_memory, f, indent=2)
            
            print(f"Memory state saved to {conscious_path}")
            
            self.neuroproxy_engine._save_persistent_state()
            
        except Exception as e:
            print(f"Error saving memory state: {e}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        rag_stats = self.rag_system.get_collection_stats()
        
        return {
            "conscious_memories": len(self.conscious_memory.get("memories", [])),
            "conscious_relationships": len(self.conscious_memory.get("relationships", [])),
            "unconscious_signifiers": len(self.unconscious_memory.get("signifiers", [])),
            "signifying_chains": len(self.unconscious_memory.get("signifying_chains", [])),
            "short_term_entries": len(self.short_term_memory),
            "rag_memories": rag_stats["memories_count"],
            "rag_relationships": rag_stats["relationships_count"],
            "emotional_state": self.neuroproxy_engine.get_current_state()["emotion_category"]
        }