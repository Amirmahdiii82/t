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
        
        # FIXED: Two separate short-term memory systems
        # 1. Psychological short-term memory (persistent, affects emotional state)
        self.psychological_short_term = self.conscious_memory.get("short_term_memory", [])
        
        # 2. Conversational short-term memory (session-only, for context)
        self.conversational_history = []
        
        self.max_short_term_size = 10
        self.max_conversation_history = 8  # Recent exchanges only
        
        print(f"Memory Manager initialized for {agent_name}")
        if self.psychological_short_term:
            print(f"✅ Loaded {len(self.psychological_short_term)} psychological memories.")
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
    
    def add_to_psychological_memory(self, content: str, context: str = "interaction") -> None:
        """Add content to psychological short-term memory (affects emotional state)."""
        memory_entry = {
            "content": content,
            "context": context,
            "timestamp": datetime.now().isoformat(),
            "emotional_state": self.neuroproxy_engine.get_current_state()
        }
        
        self.psychological_short_term.append(memory_entry)
        
        if len(self.psychological_short_term) > self.max_short_term_size:
            self.psychological_short_term = self.psychological_short_term[-self.max_short_term_size:]
        
        # Update emotional state based on psychological memory
        self.neuroproxy_engine.update_emotional_state(content, context)
    
    def add_conversation_exchange(self, user_message: str, agent_response: str) -> None:
        """Add a conversation exchange to conversational history."""
        exchange = {
            "user": user_message,
            "agent": agent_response,
            "timestamp": datetime.now().isoformat()
        }
        
        self.conversational_history.append(exchange)
        
        # Keep only recent exchanges for context
        if len(self.conversational_history) > self.max_conversation_history:
            self.conversational_history = self.conversational_history[-self.max_conversation_history:]
        
        # Also add to psychological memory for emotional impact
        self.add_to_psychological_memory(user_message, context="user_interaction")
        self.add_to_psychological_memory(agent_response, context="agent_response")
    
    def get_conversation_context(self, limit: int = 4) -> List[Dict[str, Any]]:
        """Get recent conversation history for context."""
        return self.conversational_history[-limit:] if self.conversational_history else []
    
    def get_conversation_summary(self) -> str:
        """Generate a summary of the conversation so far."""
        if not self.conversational_history:
            return "No conversation history yet."
        
        # Create a simple summary of recent topics
        recent_exchanges = self.conversational_history[-3:]
        topics = []
        
        for exchange in recent_exchanges:
            user_msg = exchange["user"][:50] + "..." if len(exchange["user"]) > 50 else exchange["user"]
            topics.append(f"User asked about: {user_msg}")
        
        return "Recent conversation: " + " | ".join(topics)
    
    def retrieve_memories(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant memories using RAG system."""
        return self.rag_system.search_memories(query, n_results)
    
    def retrieve_relationships(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """Retrieve relevant relationships using RAG system."""
        return self.rag_system.search_relationships(query, n_results)
    
    def get_short_term_memory(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent psychological short-term memory entries."""
        return self.psychological_short_term[-limit:]
    
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
            # Save psychological short-term memory (persistent)
            self.conscious_memory["short_term_memory"] = self.psychological_short_term
            self.conscious_memory["last_updated"] = datetime.now().isoformat()
            
            conscious_path = os.path.join(self.agent_path, "conscious_memory.json")
            with open(conscious_path, 'w') as f:
                json.dump(self.conscious_memory, f, indent=2)
            
            print(f"Memory state saved to {conscious_path}")
            
            # Save conversational history separately (session-specific)
            if self.conversational_history:
                session_path = os.path.join(self.agent_path, "last_session.json")
                session_data = {
                    "conversation_history": self.conversational_history,
                    "session_end": datetime.now().isoformat()
                }
                with open(session_path, 'w') as f:
                    json.dump(session_data, f, indent=2)
                print(f"Session history saved to {session_path}")
            
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
            "psychological_short_term": len(self.psychological_short_term),
            "conversational_exchanges": len(self.conversational_history),
            "rag_memories": rag_stats["memories_count"],
            "rag_relationships": rag_stats["relationships_count"],
            "emotional_state": self.neuroproxy_engine.get_current_state()["emotion_category"]
        }