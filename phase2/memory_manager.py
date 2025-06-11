import os
import json
from datetime import datetime
from typing import Dict, List, Any
from utils.rag_system import RAGSystem
from utils.neuroproxy_engine import NeuroProxyEngine

class MemoryManager:
    def __init__(self, agent_name: str, base_path: str = "base_agents"):
        """Initialize memory manager for an agent with full unconscious integration."""
        self.agent_name = agent_name
        self.base_path = base_path
        
        if isinstance(base_path, str):
            self.agent_path = os.path.join(base_path, agent_name)
        else:
            self.agent_path = os.path.join("base_agents", agent_name)
        
        print(f"Initializing Memory Manager for {agent_name}...")
        
        # Initialize RAG system for conscious memory
        self.rag_system = RAGSystem(agent_name, self.base_path)
        
        # Initialize enhanced neuroproxy engine with unconscious integration
        self.neuroproxy_engine = NeuroProxyEngine(agent_name, self.base_path)
        
        # Load conscious and unconscious memory
        self.conscious_memory = self._load_conscious_memory()
        self.unconscious_memory = self._load_unconscious_memory()
        
        # Load short-term memory from persistent storage
        self.short_term_memory = self.conscious_memory.get("short_term_memory", [])
        self.max_short_term_size = 10
        
        print(f"✅ Memory Manager initialized for {agent_name}")
        print(f"   Conscious: {len(self.conscious_memory.get('memories', []))} memories, {len(self.conscious_memory.get('relationships', []))} relationships")
        print(f"   Unconscious: {len(self.unconscious_memory.get('signifiers', []))} signifiers, {len(self.unconscious_memory.get('signifying_chains', []))} chains")
        
        if self.short_term_memory:
            print(f"   Short-term: {len(self.short_term_memory)} entries loaded from persistent storage")
        
        # Validate unconscious structure
        self._validate_unconscious_structure()
    
    def _load_conscious_memory(self) -> Dict[str, Any]:
        """Load conscious memory from file."""
        conscious_path = os.path.join(self.agent_path, "conscious_memory.json")
        
        try:
            if os.path.exists(conscious_path):
                with open(conscious_path, 'r') as f:
                    conscious_data = json.load(f)
                print(f"   Loaded conscious memory from {conscious_path}")
                return conscious_data
            else:
                print(f"   No conscious_memory.json found at {conscious_path}")
                return {"memories": [], "relationships": [], "persona": {}}
        except Exception as e:
            print(f"   Error loading conscious memory: {e}")
            return {"memories": [], "relationships": [], "persona": {}}
    
    def _load_unconscious_memory(self) -> Dict[str, Any]:
        """Load unconscious memory from file."""
        unconscious_path = os.path.join(self.agent_path, "unconscious_memory.json")
        
        try:
            if os.path.exists(unconscious_path):
                with open(unconscious_path, 'r') as f:
                    unconscious_data = json.load(f)
                print(f"   Loaded unconscious memory from {unconscious_path}")
                return unconscious_data
            else:
                print(f"   No unconscious_memory.json found at {unconscious_path}")
                return self._create_empty_unconscious_structure()
        except Exception as e:
            print(f"   Error loading unconscious memory: {e}")
            return self._create_empty_unconscious_structure()
    
    def _create_empty_unconscious_structure(self) -> Dict[str, Any]:
        """Create empty unconscious structure."""
        return {
            "signifiers": [],
            "signifying_chains": [],
            "object_a": {
                "description": "No object a identified",
                "manifestations": [],
                "void_manifestations": []
            },
            "symptom": {
                "description": "No symptom identified",
                "signifiers_involved": [],
                "jouissance_pattern": "",
                "repetition_structure": ""
            },
            "structural_positions": {
                "hysteric": 0.4,
                "master": 0.3,
                "university": 0.2,
                "analyst": 0.1
            },
            "signifier_graph": {"nodes": [], "edges": []},
            "metadata": {
                "agent_name": self.agent_name,
                "extraction_date": datetime.now().isoformat(),
                "theoretical_framework": "Lacanian Psychoanalysis"
            }
        }
    
    def _validate_unconscious_structure(self):
        """Validate and log unconscious memory structure."""
        try:
            # Check signifiers
            signifiers = self.unconscious_memory.get('signifiers', [])
            valid_signifiers = [s for s in signifiers if isinstance(s, dict) and s.get('name')]
            if len(valid_signifiers) != len(signifiers):
                print(f"   ⚠️ Warning: {len(signifiers) - len(valid_signifiers)} invalid signifiers found")
            
            # Check repressed content
            repressed = [s for s in valid_signifiers if s.get('repressed', False)]
            print(f"   Repressed signifiers: {len(repressed)}")
            
            # Check object_a
            object_a = self.unconscious_memory.get('object_a', {})
            if object_a.get('description') and object_a.get('description') != "No object a identified":
                print(f"   Object a: '{object_a['description'][:50]}...'")
            
            # Check symptom
            symptom = self.unconscious_memory.get('symptom', {})
            if symptom.get('description') and symptom.get('description') != "No symptom identified":
                print(f"   Symptom: '{symptom['description'][:50]}...'")
            
            # Check chains
            chains = self.unconscious_memory.get('signifying_chains', [])
            valid_chains = [c for c in chains if isinstance(c, dict) and c.get('name')]
            print(f"   Signifying chains: {len(valid_chains)}")
            
        except Exception as e:
            print(f"   Error validating unconscious structure: {e}")
    
    def add_to_short_term_memory(self, content: str, context: str = "interaction") -> None:
        """Add content to short-term memory with neurochemical and unconscious integration."""
        memory_entry = {
            "content": content,
            "context": context,
            "timestamp": datetime.now().isoformat(),
            "emotional_state": self.neuroproxy_engine.get_current_state()
        }
        
        self.short_term_memory.append(memory_entry)
        
        if len(self.short_term_memory) > self.max_short_term_size:
            self.short_term_memory = self.short_term_memory[-self.max_short_term_size:]
        
        # Update emotional state based on content
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
        """Get current emotional state with unconscious influence."""
        return self.neuroproxy_engine.get_current_state()
    
    def get_response_style_guidance(self) -> Dict[str, str]:
        """Get guidance for response style based on emotional state and unconscious dynamics."""
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
    
    def get_structural_positions(self) -> Dict[str, float]:
        """Get discourse structural positions."""
        return self.unconscious_memory.get("structural_positions", {})
    
    def get_repressed_signifiers(self) -> List[Dict[str, Any]]:
        """Get signifiers marked as repressed."""
        signifiers = self.unconscious_memory.get("signifiers", [])
        return [sig for sig in signifiers if isinstance(sig, dict) and sig.get('repressed', False)]
    
    def get_signifier_by_name(self, signifier_name: str) -> Dict[str, Any]:
        """Get specific signifier by name."""
        signifiers = self.unconscious_memory.get("signifiers", [])
        for sig in signifiers:
            if isinstance(sig, dict) and sig.get('name') == signifier_name:
                return sig
        return {}
    
    def check_signifier_activation(self, text: str) -> List[Dict[str, Any]]:
        """Check which signifiers are activated by given text."""
        activated = []
        text_lower = text.lower()
        
        signifiers = self.unconscious_memory.get("signifiers", [])
        for sig in signifiers:
            if not isinstance(sig, dict):
                continue
                
            sig_name = sig.get('name', '')
            if not sig_name:
                continue
            
            activation_strength = 0.0
            activation_sources = []
            
            # Direct name match
            if sig_name.lower() in text_lower:
                activation_strength += 1.0
                activation_sources.append("direct_match")
            
            # Association matches
            associations = sig.get('associations', [])
            for assoc in associations:
                if isinstance(assoc, str) and assoc.lower() in text_lower:
                    activation_strength += 0.5
                    activation_sources.append(f"association_{assoc}")
            
            if activation_strength > 0:
                activated.append({
                    'signifier': sig_name,
                    'activation_strength': activation_strength,
                    'activation_sources': activation_sources,
                    'signifier_data': sig
                })
        
        return sorted(activated, key=lambda x: x['activation_strength'], reverse=True)
    
    def check_chain_activation(self, activated_signifiers: List[str]) -> List[Dict[str, Any]]:
        """Check which signifying chains are activated by given signifiers."""
        active_chains = []
        
        chains = self.unconscious_memory.get("signifying_chains", [])
        for chain in chains:
            if not isinstance(chain, dict):
                continue
                
            chain_signifiers = chain.get('signifiers', [])
            active_in_chain = [sig for sig in activated_signifiers if sig in chain_signifiers]
            
            if active_in_chain:
                activation_strength = len(active_in_chain) / len(chain_signifiers) if chain_signifiers else 0
                active_chains.append({
                    'name': chain.get('name', 'unnamed_chain'),
                    'signifiers': chain_signifiers,
                    'active_signifiers': active_in_chain,
                    'activation_strength': activation_strength,
                    'chain_data': chain
                })
        
        return sorted(active_chains, key=lambda x: x['activation_strength'], reverse=True)
    
    def get_object_a_proximity(self, text: str) -> float:
        """Calculate proximity to object_a based on text content."""
        object_a = self.unconscious_memory.get('object_a', {})
        manifestations = object_a.get('manifestations', [])
        void_manifestations = object_a.get('void_manifestations', [])
        
        proximity = 0.0
        text_lower = text.lower()
        
        # Check manifestations
        for manifestation in manifestations:
            if isinstance(manifestation, str) and any(word in text_lower for word in manifestation.lower().split()):
                proximity += 0.3
        
        # Check void manifestations (stronger proximity)
        for void_manifest in void_manifestations:
            if isinstance(void_manifest, str) and any(word in text_lower for word in void_manifest.lower().split()):
                proximity += 0.5
        
        return min(proximity, 1.0)
    
    def get_symptom_activation_level(self, text: str, activated_signifiers: List[str]) -> float:
        """Calculate symptom activation level based on text and signifiers."""
        symptom = self.unconscious_memory.get('symptom', {})
        symptom_signifiers = symptom.get('signifiers_involved', [])
        
        activation = 0.0
        
        # Check if symptom signifiers are activated
        for symptom_sig in symptom_signifiers:
            if symptom_sig in activated_signifiers:
                activation += 0.5
        
        # Check for symptom-related content in text
        jouissance_pattern = symptom.get('jouissance_pattern', '').lower()
        repetition_structure = symptom.get('repetition_structure', '').lower()
        
        text_lower = text.lower()
        
        if any(word in text_lower for word in jouissance_pattern.split() if len(word) > 3):
            activation += 0.3
        
        if any(word in text_lower for word in repetition_structure.split() if len(word) > 3):
            activation += 0.3
        
        return min(activation, 1.0)
    
    def consolidate_short_term_memory(self) -> None:
        """Consolidate short-term memory into long-term memory (for future implementation)."""
        # This could analyze patterns in short-term memory and create new long-term memories
        # For now, we just maintain the short-term memory size
        pass
    
    def save_state(self) -> None:
        """Save current memory state including unconscious dynamics."""
        try:
            # Update conscious memory with current short-term memory
            self.conscious_memory["short_term_memory"] = self.short_term_memory
            self.conscious_memory["last_updated"] = datetime.now().isoformat()
            
            # Save conscious memory
            conscious_path = os.path.join(self.agent_path, "conscious_memory.json")
            with open(conscious_path, 'w') as f:
                json.dump(self.conscious_memory, f, indent=2)
            
            print(f"   Conscious memory state saved to {conscious_path}")
            
            # Save neuroproxy engine state (includes unconscious influence data)
            self.neuroproxy_engine._save_persistent_state()
            
            print(f"   Neurochemical and unconscious influence state saved")
            
        except Exception as e:
            print(f"   Error saving memory state: {e}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory system statistics."""
        rag_stats = self.rag_system.get_collection_stats()
        neurochemical_stats = self.neuroproxy_engine.get_unconscious_influence_stats()
        
        return {
            "conscious_memories": len(self.conscious_memory.get("memories", [])),
            "conscious_relationships": len(self.conscious_memory.get("relationships", [])),
            "unconscious_signifiers": len(self.unconscious_memory.get("signifiers", [])),
            "repressed_signifiers": len([s for s in self.unconscious_memory.get("signifiers", []) 
                                       if isinstance(s, dict) and s.get('repressed', False)]),
            "signifying_chains": len(self.unconscious_memory.get("signifying_chains", [])),
            "short_term_entries": len(self.short_term_memory),
            "rag_memories": rag_stats["memories_count"],
            "rag_relationships": rag_stats["relationships_count"],
            "emotional_state": self.neuroproxy_engine.get_current_state()["emotion_category"],
            "unconscious_influence": neurochemical_stats,
            "object_a_identified": bool(self.unconscious_memory.get("object_a", {}).get("description") != "No object a identified"),
            "symptom_identified": bool(self.unconscious_memory.get("symptom", {}).get("description") != "No symptom identified")
        }
    
    def get_unconscious_summary(self) -> Dict[str, Any]:
        """Get summary of unconscious structure for analysis."""
        signifiers = self.unconscious_memory.get("signifiers", [])
        chains = self.unconscious_memory.get("signifying_chains", [])
        
        return {
            "total_signifiers": len(signifiers),
            "repressed_signifiers": [s.get('name', 'Unknown') for s in signifiers 
                                   if isinstance(s, dict) and s.get('repressed', False)],
            "key_signifiers": [s.get('name', 'Unknown') for s in signifiers[:5] 
                             if isinstance(s, dict)],
            "signifying_chains": [c.get('name', 'Unknown') for c in chains 
                                if isinstance(c, dict)],
            "object_a_description": self.unconscious_memory.get("object_a", {}).get("description", "Not identified"),
            "symptom_description": self.unconscious_memory.get("symptom", {}).get("description", "Not identified"),
            "structural_positions": self.unconscious_memory.get("structural_positions", {}),
            "unconscious_influence_active": self.neuroproxy_engine.get_unconscious_influence_stats()["total_triggers"] > 0
        }