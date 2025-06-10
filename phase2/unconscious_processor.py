import json
from typing import Dict, List, Any
from datetime import datetime
from utils.lacanian_graph import LacanianSignifierGraph

class UnconsciousProcessor:
    """
    Processes input through unconscious signifier networks and Lacanian dynamics.
    
    Activates signifiers based on memory content and traces their influence
    through signifying chains to model unconscious associations.
    """
    
    def __init__(self, agent_name: str, memory_manager):
        self.agent_name = agent_name
        self.memory_manager = memory_manager
        
        # Load unconscious structures
        self._load_unconscious_structures()
        
        # Track current unconscious state
        self.active_signifiers = []
        self.current_discourse = "hysteric"  # Default position
        
    def _load_unconscious_structures(self):
        """Load agent's unconscious structures from memory."""
        try:
            unconscious_data = self.memory_manager.unconscious_memory
            
            # Load signifier graph
            if 'signifier_graph' in unconscious_data:
                self.signifier_graph = self._reconstruct_graph(unconscious_data['signifier_graph'])
            else:
                self.signifier_graph = LacanianSignifierGraph()
            
            # Load other structures
            self.signifiers = unconscious_data.get('signifiers', [])
            self.chains = unconscious_data.get('signifying_chains', [])
            self.object_a = unconscious_data.get('object_a', {})
            self.symptom = unconscious_data.get('symptom', {})
            
        except Exception as e:
            self._initialize_empty_structures()
    
    def _reconstruct_graph(self, graph_data: Dict) -> LacanianSignifierGraph:
        """Reconstruct signifier graph from serialized data."""
        graph = LacanianSignifierGraph()
        
        # Rebuild nodes
        for node in graph_data.get('nodes', []):
            node_id = node.pop('id')
            graph.graph.add_node(node_id, **node)
        
        # Rebuild edges
        for edge in graph_data.get('edges', []):
            graph.graph.add_edge(edge['source'], edge['target'], **edge)
        
        # Restore structures
        graph.master_signifiers = graph_data.get('master_signifiers', {})
        graph.signifying_chains = graph_data.get('signifying_chains', {})
        graph.object_a_positions = graph_data.get('object_a_positions', [])
        
        return graph
    
    def process_input(self, user_input: str, context: str = "dialogue") -> Dict[str, Any]:
        """
        Process input through unconscious mechanisms.
        
        Args:
            user_input: User's message
            context: Interaction context
            
        Returns:
            Dictionary containing unconscious analysis results
        """
        try:
            # 1. Identify signifiers activated by memories
            active_signifiers = self._identify_signifiers_from_memories(user_input)
            
            # 2. Process through signifying chains
            chain_effects = self._process_through_chains(active_signifiers)
            
            # 3. Analyze discourse position
            discourse_position = self.signifier_graph.analyze_discourse_position(
                [s['signifier'] for s in active_signifiers]
            )
            
            # 4. Calculate jouissance dynamics
            jouissance_effects = self._calculate_jouissance_effects(active_signifiers)
            
            # Create unconscious influence object
            unconscious_influence = {
                "active_signifiers": active_signifiers,
                "chain_activations": chain_effects,
                "discourse_position": discourse_position,
                "jouissance_effects": jouissance_effects,
                "timestamp": datetime.now().isoformat()
            }
            
            # Update internal state
            self._update_unconscious_state(unconscious_influence)
            
            return unconscious_influence
            
        except Exception as e:
            return self._minimal_unconscious_response()
    
    def _identify_signifiers_from_memories(self, user_input: str) -> List[Dict[str, Any]]:
        """
        Identify active signifiers based on memory retrieval results.
        
        This is the core innovation: signifiers activate based on actual
        memory content rather than simple text matching.
        """
        active_signifiers = []
        
        # Get memories triggered by user input
        relevant_memories = self.memory_manager.retrieve_memories(user_input, 10)
        relevant_relationships = self.memory_manager.retrieve_relationships(user_input, 5)
        
        # Extract signifiers from memory content
        for memory in relevant_memories:
            memory_signifiers = self._extract_signifiers_from_memory(memory)
            active_signifiers.extend(memory_signifiers)
        
        # Extract signifiers from relationship content
        for relationship in relevant_relationships:
            rel_signifiers = self._extract_signifiers_from_relationship(relationship)
            active_signifiers.extend(rel_signifiers)
        
        # Remove duplicates and calculate activation strength
        unique_signifiers = {}
        for sig in active_signifiers:
            name = sig['signifier']
            if name in unique_signifiers:
                # Increase activation if signifier appears multiple times
                unique_signifiers[name]['activation_strength'] += 0.2
            else:
                unique_signifiers[name] = sig
        
        # Apply resonance through signifier graph
        final_signifiers = list(unique_signifiers.values())
        if final_signifiers:
            resonance_signifiers = self._apply_signifier_resonance(final_signifiers)
            final_signifiers.extend(resonance_signifiers)
        
        return final_signifiers[:10]  # Limit to most relevant
    
    def _extract_signifiers_from_memory(self, memory: Dict) -> List[Dict[str, Any]]:
        """Extract signifiers that might be present in memory content."""
        signifiers = []
        
        # Get memory text content
        memory_text = ""
        if isinstance(memory, dict):
            memory_text += memory.get('title', '') + " "
            memory_text += memory.get('description', '') + " "
            memory_text += " ".join(memory.get('associated_people', []))
        
        memory_text = memory_text.lower()
        
        # Check against known signifiers
        for signifier_data in self.signifiers:
            if isinstance(signifier_data, dict):
                signifier_name = signifier_data.get('name', '').lower()
                
                # Direct match
                if signifier_name in memory_text:
                    signifiers.append({
                        'signifier': signifier_data.get('name'),
                        'activation_type': 'memory_content',
                        'activation_strength': 0.8,
                        'source_memory': memory.get('title', 'untitled'),
                        'significance': signifier_data.get('significance', '')
                    })
                
                # Association match
                associations = signifier_data.get('associations', [])
                for assoc in associations:
                    if isinstance(assoc, str) and assoc.lower() in memory_text:
                        signifiers.append({
                            'signifier': signifier_data.get('name'),
                            'activation_type': 'memory_association',
                            'activation_strength': 0.6,
                            'source_memory': memory.get('title', 'untitled'),
                            'triggered_by': assoc,
                            'significance': signifier_data.get('significance', '')
                        })
                        break
        
        return signifiers
    
    def _extract_signifiers_from_relationship(self, relationship: Dict) -> List[Dict[str, Any]]:
        """Extract signifiers from relationship content."""
        signifiers = []
        
        if not isinstance(relationship, dict):
            return signifiers
        
        # Get relationship content
        rel_text = (
            relationship.get('name', '') + " " +
            relationship.get('relationship_type', '') + " " +
            relationship.get('emotional_significance', '')
        ).lower()
        
        # Check against known signifiers
        for signifier_data in self.signifiers:
            if isinstance(signifier_data, dict):
                signifier_name = signifier_data.get('name', '').lower()
                
                if signifier_name in rel_text:
                    signifiers.append({
                        'signifier': signifier_data.get('name'),
                        'activation_type': 'relationship_content',
                        'activation_strength': 0.7,
                        'source_relationship': relationship.get('name', 'unknown'),
                        'significance': signifier_data.get('significance', '')
                    })
        
        return signifiers
    
    def _apply_signifier_resonance(self, primary_signifiers: List[Dict]) -> List[Dict[str, Any]]:
        """Apply resonance through signifier network to activate related signifiers."""
        resonance_signifiers = []
        
        for sig_data in primary_signifiers[:3]:  # Use top 3 for resonance
            signifier = sig_data['signifier']
            
            if signifier in self.signifier_graph.graph:
                # Get resonance map
                resonance = self.signifier_graph.get_signifier_resonance(signifier, depth=2)
                
                # Add resonated signifiers with lower activation
                for resonated_sig, strength in resonance.items():
                    if resonated_sig != signifier and strength > 0.3:
                        resonance_signifiers.append({
                            'signifier': resonated_sig,
                            'activation_type': 'resonance',
                            'activation_strength': strength * 0.5,
                            'resonated_from': signifier,
                            'significance': f'Resonance from {signifier}'
                        })
        
        return resonance_signifiers[:5]  # Limit resonance additions
    
    def _process_through_chains(self, active_signifiers: List[Dict]) -> Dict[str, Any]:
        """Process active signifiers through signifying chains."""
        chain_effects = {
            'activated_chains': [],
            'retroactive_determinations': []
        }
        
        # Find which chains are activated
        for chain_name, chain_data in self.signifier_graph.signifying_chains.items():
            chain_signifiers = chain_data['signifiers']
            active_in_chain = [s['signifier'] for s in active_signifiers 
                              if s['signifier'] in chain_signifiers]
            
            if active_in_chain:
                activation_strength = len(active_in_chain) / len(chain_signifiers)
                chain_effects['activated_chains'].append({
                    'chain': chain_name,
                    'activation': activation_strength,
                    'active_signifiers': active_in_chain
                })
                
                # Check for retroactive effects (Nachträglichkeit)
                if chain_data.get('retroactive', False) and chain_signifiers:
                    last_sig = chain_signifiers[-1]
                    if last_sig in active_in_chain:
                        chain_effects['retroactive_determinations'].append({
                            'determining_signifier': last_sig,
                            'affected_signifiers': [s for s in chain_signifiers if s != last_sig],
                            'retroactive_meaning': f"{last_sig} determines meaning of {chain_name}"
                        })
        
        return chain_effects
    
    def _calculate_jouissance_effects(self, active_signifiers: List[Dict]) -> Dict[str, Any]:
        """Calculate jouissance dynamics based on signifier activation."""
        jouissance = {
            'level': 0.0,
            'patterns': [],
            'symptom_activation': False
        }
        
        # Check if symptom signifiers are active
        if self.symptom:
            symptom_signifiers = self.symptom.get('signifiers_involved', [])
            active_names = [s['signifier'] for s in active_signifiers]
            
            symptom_activation = any(sig in active_names for sig in symptom_signifiers)
            if symptom_activation:
                jouissance['symptom_activation'] = True
                jouissance['level'] += 0.4
                jouissance['patterns'].append({
                    'type': 'symptomatic_jouissance',
                    'description': 'Symptom signifiers activated'
                })
        
        # Check for proximity to object a
        for sig_data in active_signifiers:
            signifier = sig_data['signifier']
            if signifier in self.signifier_graph.graph:
                node_data = self.signifier_graph.graph.nodes[signifier]
                if node_data.get('object_a_proximity', False):
                    jouissance['level'] += 0.3
                    jouissance['patterns'].append({
                        'type': 'object_a_circulation',
                        'signifier': signifier,
                        'description': f"{signifier} circles the void of object a"
                    })
        
        # Cap jouissance level
        jouissance['level'] = min(1.0, jouissance['level'])
        
        return jouissance
    
    def _update_unconscious_state(self, influence: Dict[str, Any]) -> None:
        """Update unconscious processor's internal state."""
        # Update active signifiers
        self.active_signifiers = influence.get('active_signifiers', [])
        
        # Update discourse position
        discourse_scores = influence.get('discourse_position', {})
        if discourse_scores:
            self.current_discourse = max(discourse_scores.items(), key=lambda x: x[1])[0]
    
    def _initialize_empty_structures(self):
        """Initialize empty structures if loading fails."""
        self.signifier_graph = LacanianSignifierGraph()
        self.signifiers = []
        self.chains = []
        self.object_a = {}
        self.symptom = {}
    
    def _minimal_unconscious_response(self) -> Dict[str, Any]:
        """Provide minimal response when processing fails."""
        return {
            "active_signifiers": [],
            "chain_activations": {"activated_chains": []},
            "discourse_position": {"hysteric": 0.5, "master": 0.2, "university": 0.2, "analyst": 0.1},
            "jouissance_effects": {"level": 0.0, "patterns": []},
            "timestamp": datetime.now().isoformat()
        }