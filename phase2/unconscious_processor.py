import json
from typing import Dict, List, Any
from datetime import datetime
from utils.lacanian_graph import LacanianSignifierGraph
from interfaces.vlm_interface import VLMInterface

class UnconsciousProcessor:
    """
    Processes input through unconscious signifier networks using ALL extracted data.
    
    Actually uses signifying chains, object_a dynamics, symptom patterns, 
    retroactive effects, and repressed content to create authentic unconscious responses.
    """
    
    def __init__(self, agent_name: str, memory_manager):
        self.agent_name = agent_name
        self.memory_manager = memory_manager
        
        # Use VLM for complex unconscious analysis
        self.vlm = VLMInterface()
        
        # Load unconscious structures
        self._load_unconscious_structures()
        
        # Track current unconscious state
        self.active_signifiers = []
        self.active_chains = []
        self.current_discourse = "hysteric"
        self.object_a_proximity = 0.0
        self.symptom_activation_level = 0.0
        
        print(f"Unconscious processor initialized for {agent_name}")
        print(f"Loaded: {len(self.signifiers)} signifiers, {len(self.chains)} chains")
        
    def _load_unconscious_structures(self):
        """Load agent's unconscious structures from memory."""
        try:
            unconscious_data = self.memory_manager.unconscious_memory
            
            # Load signifier graph
            if 'signifier_graph' in unconscious_data:
                self.signifier_graph = self._reconstruct_graph(unconscious_data['signifier_graph'])
            else:
                self.signifier_graph = LacanianSignifierGraph()
            
            # Load core structures
            self.signifiers = unconscious_data.get('signifiers', [])
            self.chains = unconscious_data.get('signifying_chains', [])
            self.object_a = unconscious_data.get('object_a', {})
            self.symptom = unconscious_data.get('symptom', {})
            self.structural_positions = unconscious_data.get('structural_positions', {})
            
            # Extract key data for efficient processing
            self._prepare_unconscious_data()
            
        except Exception as e:
            print(f"Error loading unconscious structures: {e}")
            self._initialize_empty_structures()
    
    def _prepare_unconscious_data(self):
        """Prepare unconscious data for efficient processing."""
        # Map signifier names to full data
        self.signifier_map = {sig['name']: sig for sig in self.signifiers if isinstance(sig, dict)}
        
        # Extract repressed signifiers
        self.repressed_signifiers = [
            sig['name'] for sig in self.signifiers 
            if isinstance(sig, dict) and sig.get('repressed', False)
        ]
        
        # Extract object_a manifestations
        self.object_a_manifestations = self.object_a.get('manifestations', [])
        self.void_manifestations = self.object_a.get('void_manifestations', [])
        
        # Extract symptom signifiers
        self.symptom_signifiers = self.symptom.get('signifiers_involved', [])
        
        # Prepare chain activation maps
        self.chain_map = {}
        for chain in self.chains:
            if isinstance(chain, dict):
                name = chain.get('name', '')
                signifiers = chain.get('signifiers', [])
                self.chain_map[name] = {
                    'signifiers': signifiers,
                    'type': chain.get('type', 'mixed'),
                    'explanation': chain.get('explanation', ''),
                    'relation_to_fantasy': chain.get('relation_to_fantasy', '')
                }
    
    def process_input(self, user_input: str, context: str = "dialogue") -> Dict[str, Any]:
        """
        Process input through complete unconscious mechanisms using all extracted data.
        """
        try:
            print(f"\n=== Unconscious Processing for {self.agent_name} ===")
            print(f"Input: '{user_input[:50]}...'")
            
            # 1. Identify directly activated signifiers from memories
            directly_activated = self._identify_signifiers_from_memories(user_input)
            
            # 2. Apply signifying chain dynamics
            chain_activated = self._activate_signifying_chains(directly_activated)
            
            # 3. Apply retroactive determination (Nachträglichkeit)
            retroactively_determined = self._apply_retroactive_effects(chain_activated)
            
            # 4. Check for return of repressed content
            repressed_returns = self._check_return_of_repressed(retroactively_determined, user_input)
            
            # 5. Calculate object_a proximity and dynamics
            object_a_effects = self._calculate_object_a_dynamics(retroactively_determined, user_input)
            
            # 6. Check symptom activation patterns
            symptom_effects = self._calculate_symptom_activation(retroactively_determined, user_input)
            
            # 7. Determine discourse position based on current dynamics
            discourse_position = self._determine_discourse_position(retroactively_determined, object_a_effects, symptom_effects)
            
            # 8. Trigger neurochemical responses based on unconscious dynamics
            self._trigger_neurochemical_responses(retroactively_determined, object_a_effects, symptom_effects)
            
            # Create comprehensive unconscious influence
            unconscious_influence = {
                "active_signifiers": retroactively_determined,
                "activated_chains": chain_activated,
                "repressed_returns": repressed_returns,
                "object_a_effects": object_a_effects,
                "symptom_effects": symptom_effects,
                "discourse_position": discourse_position,
                "unconscious_dynamics": {
                    "object_a_proximity": self.object_a_proximity,
                    "symptom_activation": self.symptom_activation_level,
                    "primary_discourse": self.current_discourse,
                    "repressed_content_emerging": len(repressed_returns) > 0
                },
                "timestamp": datetime.now().isoformat(),
                "processing_method": "complete_unconscious_integration"
            }
            
            # Update internal state
            self._update_unconscious_state(unconscious_influence)
            
            print(f"Unconscious activation: {len(retroactively_determined)} signifiers")
            print(f"Active chains: {[chain['name'] for chain in chain_activated]}")
            print(f"Object a proximity: {self.object_a_proximity:.2f}")
            print(f"Symptom activation: {self.symptom_activation_level:.2f}")
            print(f"Discourse position: {self.current_discourse}")
            
            return unconscious_influence
            
        except Exception as e:
            print(f"Error in unconscious processing: {e}")
            return self._minimal_unconscious_response()
    
    def _identify_signifiers_from_memories(self, user_input: str) -> List[Dict[str, Any]]:
        """Identify signifiers from memory content and direct input analysis."""
        active_signifiers = []
        
        # Get memories triggered by user input
        relevant_memories = self.memory_manager.retrieve_memories(user_input, 8)
        relevant_relationships = self.memory_manager.retrieve_relationships(user_input, 5)
        recent_interactions = self.memory_manager.get_short_term_memory(5)
        
        # Combine all memory content
        memory_texts = []
        for memory in relevant_memories:
            if isinstance(memory, dict):
                text = f"{memory.get('title', '')} {memory.get('description', '')}"
                memory_texts.append(text.lower())
        
        for relationship in relevant_relationships:
            if isinstance(relationship, dict):
                text = f"{relationship.get('name', '')} {relationship.get('emotional_significance', '')}"
                memory_texts.append(text.lower())
        
        for interaction in recent_interactions:
            if isinstance(interaction, dict):
                memory_texts.append(interaction.get('content', '').lower())
        
        # Add user input
        memory_texts.append(user_input.lower())
        
        all_text = " ".join(memory_texts)
        
        # Check each signifier for activation
        for signifier_name, signifier_data in self.signifier_map.items():
            activation_strength = 0.0
            activation_sources = []
            
            # Direct name match
            if signifier_name.lower() in all_text:
                activation_strength += 1.0
                activation_sources.append("direct_match")
            
            # Association matches
            associations = signifier_data.get('associations', [])
            for assoc in associations:
                if isinstance(assoc, str) and assoc.lower() in all_text:
                    activation_strength += 0.6
                    activation_sources.append(f"association_{assoc}")
            
            # Significance-based activation
            significance = signifier_data.get('significance', '').lower()
            significance_words = significance.split()
            for word in significance_words:
                if len(word) > 4 and word in all_text:
                    activation_strength += 0.3
                    activation_sources.append(f"significance_{word}")
            
            if activation_strength > 0:
                active_signifiers.append({
                    'signifier': signifier_name,
                    'activation_strength': min(activation_strength, 2.0),
                    'activation_sources': activation_sources,
                    'signifier_data': signifier_data,
                    'repressed': signifier_data.get('repressed', False)
                })
        
        # Sort by activation strength
        active_signifiers.sort(key=lambda x: x['activation_strength'], reverse=True)
        
        return active_signifiers[:8]  # Limit to top 8
    
    def _activate_signifying_chains(self, active_signifiers: List[Dict]) -> List[Dict]:
        """Activate signifying chains based on active signifiers."""
        activated_chains = []
        active_names = [sig['signifier'] for sig in active_signifiers]
        
        for chain_name, chain_data in self.chain_map.items():
            chain_signifiers = chain_data['signifiers']
            
            # Check how many signifiers in this chain are active
            active_in_chain = [sig for sig in active_names if sig in chain_signifiers]
            
            if active_in_chain:
                activation_strength = len(active_in_chain) / len(chain_signifiers)
                
                activated_chains.append({
                    'name': chain_name,
                    'signifiers': chain_signifiers,
                    'active_signifiers': active_in_chain,
                    'activation_strength': activation_strength,
                    'chain_type': chain_data['type'],
                    'explanation': chain_data['explanation'],
                    'relation_to_fantasy': chain_data['relation_to_fantasy']
                })
                
                print(f"Chain activated: {chain_name} ({activation_strength:.2f})")
        
        return activated_chains
    
    def _apply_retroactive_effects(self, active_signifiers: List[Dict]) -> List[Dict]:
        """Apply Lacanian retroactive determination effects."""
        enhanced_signifiers = active_signifiers.copy()
        active_names = [sig['signifier'] for sig in active_signifiers]
        
        # Check retroactive effects from signifier graph
        if hasattr(self.signifier_graph, 'retroactive_effects'):
            for effect_name, effect_data in self.signifier_graph.retroactive_effects.items():
                determining_sig = effect_data.get('determining_signifier')
                affected_sigs = effect_data.get('retroactive_targets', [])
                
                # If determining signifier is active, boost affected signifiers
                if determining_sig in active_names:
                    for affected_sig in affected_sigs:
                        # Find and boost affected signifier if it exists
                        for sig_data in enhanced_signifiers:
                            if sig_data['signifier'] == affected_sig:
                                original_strength = sig_data['activation_strength']
                                sig_data['activation_strength'] = min(original_strength + 0.5, 2.0)
                                sig_data['retroactive_boost'] = determining_sig
                                print(f"Retroactive boost: {affected_sig} boosted by {determining_sig}")
                                break
                        else:
                            # Add affected signifier if not already active
                            if affected_sig in self.signifier_map:
                                enhanced_signifiers.append({
                                    'signifier': affected_sig,
                                    'activation_strength': 0.7,
                                    'activation_sources': ['retroactive_determination'],
                                    'signifier_data': self.signifier_map[affected_sig],
                                    'retroactive_boost': determining_sig,
                                    'repressed': self.signifier_map[affected_sig].get('repressed', False)
                                })
                                print(f"Retroactive activation: {affected_sig} activated by {determining_sig}")
        
        # Re-sort by activation strength
        enhanced_signifiers.sort(key=lambda x: x['activation_strength'], reverse=True)
        return enhanced_signifiers
    
    def _check_return_of_repressed(self, active_signifiers: List[Dict], user_input: str) -> List[Dict]:
        """Check for return of repressed content."""
        repressed_returns = []
        
        for sig_data in active_signifiers:
            if sig_data.get('repressed', False):
                # Repressed content is returning
                repressed_returns.append({
                    'signifier': sig_data['signifier'],
                    'return_strength': sig_data['activation_strength'],
                    'return_context': 'activated_through_interaction',
                    'significance': sig_data['signifier_data'].get('significance', ''),
                    'associations': sig_data['signifier_data'].get('associations', [])
                })
                
                print(f"Return of repressed: {sig_data['signifier']}")
        
        return repressed_returns
    
    def _calculate_object_a_dynamics(self, active_signifiers: List[Dict], user_input: str) -> Dict[str, Any]:
        """Calculate object_a proximity and effects."""
        object_a_effects = {
            'proximity_level': 0.0,
            'active_manifestations': [],
            'desire_direction': 'neutral',
            'anxiety_triggers': []
        }
        
        active_names = [sig['signifier'] for sig in active_signifiers]
        
        # Check if object_a manifestations are active
        for manifestation in self.object_a_manifestations:
            for active_name in active_names:
                if active_name.lower() in manifestation.lower():
                    object_a_effects['proximity_level'] += 0.4
                    object_a_effects['active_manifestations'].append(manifestation)
                    print(f"Object a manifestation active: {manifestation}")
        
        # Check void manifestations
        for void_manifest in self.void_manifestations:
            for active_name in active_names:
                if active_name.lower() in void_manifest.lower():
                    object_a_effects['proximity_level'] += 0.6
                    object_a_effects['anxiety_triggers'].append(void_manifest)
                    print(f"Void manifestation detected: {void_manifest}")
        
        # Determine desire direction based on object_a dynamics
        if object_a_effects['proximity_level'] > 0.5:
            if any('missing' in manifest.lower() for manifest in object_a_effects['active_manifestations']):
                object_a_effects['desire_direction'] = 'seeking_substitute'
            else:
                object_a_effects['desire_direction'] = 'circling_void'
        
        self.object_a_proximity = min(object_a_effects['proximity_level'], 1.0)
        
        return object_a_effects
    
    def _calculate_symptom_activation(self, active_signifiers: List[Dict], user_input: str) -> Dict[str, Any]:
        """Calculate symptom activation based on extracted symptom structure."""
        symptom_effects = {
            'activation_level': 0.0,
            'active_symptom_signifiers': [],
            'jouissance_pattern': None,
            'repetition_detected': False
        }
        
        active_names = [sig['signifier'] for sig in active_signifiers]
        
        # Check if symptom signifiers are active
        for symptom_sig in self.symptom_signifiers:
            if symptom_sig in active_names:
                symptom_effects['activation_level'] += 0.6
                symptom_effects['active_symptom_signifiers'].append(symptom_sig)
                print(f"Symptom signifier active: {symptom_sig}")
        
        # If symptom is activated, apply patterns
        if symptom_effects['activation_level'] > 0:
            symptom_effects['jouissance_pattern'] = self.symptom.get('jouissance_pattern', '')
            symptom_effects['repetition_detected'] = True
            
            # Check repetition structure
            repetition = self.symptom.get('repetition_structure', '')
            if 'repeat' in repetition.lower():
                symptom_effects['repetition_detected'] = True
        
        self.symptom_activation_level = min(symptom_effects['activation_level'], 1.0)
        
        return symptom_effects
    
    def _determine_discourse_position(self, active_signifiers: List[Dict], object_a_effects: Dict, symptom_effects: Dict) -> Dict[str, float]:
        """Determine current discourse position based on unconscious dynamics."""
        # Start with extracted structural positions
        discourse_scores = self.structural_positions.copy()
        
        # Modify based on current unconscious state
        if object_a_effects['proximity_level'] > 0.5:
            # High object_a proximity = more hysteric position
            discourse_scores['hysteric'] = discourse_scores.get('hysteric', 0.3) + 0.3
            discourse_scores['analyst'] = discourse_scores.get('analyst', 0.2) + 0.2
        
        if symptom_effects['activation_level'] > 0.5:
            # Symptom activation = more hysteric position  
            discourse_scores['hysteric'] = discourse_scores.get('hysteric', 0.3) + 0.2
        
        if len([sig for sig in active_signifiers if sig.get('repressed', False)]) > 0:
            # Repressed content returning = analyst position
            discourse_scores['analyst'] = discourse_scores.get('analyst', 0.2) + 0.3
        
        # Normalize
        total = sum(discourse_scores.values())
        if total > 0:
            discourse_scores = {k: v/total for k, v in discourse_scores.items()}
        
        # Determine primary discourse
        self.current_discourse = max(discourse_scores.items(), key=lambda x: x[1])[0]
        
        return discourse_scores
    
    def _trigger_neurochemical_responses(self, active_signifiers: List[Dict], object_a_effects: Dict, symptom_effects: Dict):
        """Trigger specific neurochemical responses based on unconscious dynamics."""
        neuroproxy = self.memory_manager.neuroproxy_engine
        
        # Object a proximity effects
        if object_a_effects['proximity_level'] > 0.3:
            if 'seeking_substitute' in object_a_effects.get('desire_direction', ''):
                # Seeking behavior - increase oxytocin need, decrease serotonin
                neuroproxy.neurochemical_state['oxytocin'] = max(0.0, neuroproxy.neurochemical_state['oxytocin'] - 0.2)
                neuroproxy.neurochemical_state['serotonin'] = max(0.0, neuroproxy.neurochemical_state['serotonin'] - 0.2)
                neuroproxy.neurochemical_state['cortisol'] = min(1.0, neuroproxy.neurochemical_state['cortisol'] + 0.3)
                print("Triggered object_a seeking neurochemical pattern")
        
        # Symptom activation effects
        if symptom_effects['activation_level'] > 0.3:
            # Symptom = vulnerability pattern - specific neurochemical signature
            neuroproxy.neurochemical_state['dopamine'] = max(0.0, neuroproxy.neurochemical_state['dopamine'] - 0.2)
            neuroproxy.neurochemical_state['cortisol'] = min(1.0, neuroproxy.neurochemical_state['cortisol'] + 0.3)
            neuroproxy.neurochemical_state['norepinephrine'] = min(1.0, neuroproxy.neurochemical_state['norepinephrine'] + 0.2)
            print("Triggered symptom activation neurochemical pattern")
        
        # Repressed content returning
        repressed_active = [sig for sig in active_signifiers if sig.get('repressed', False)]
        if repressed_active:
            # Return of repressed = anxiety + disruption
            neuroproxy.neurochemical_state['cortisol'] = min(1.0, neuroproxy.neurochemical_state['cortisol'] + 0.4)
            neuroproxy.neurochemical_state['norepinephrine'] = min(1.0, neuroproxy.neurochemical_state['norepinephrine'] + 0.3)
            neuroproxy.neurochemical_state['gaba'] = max(0.0, neuroproxy.neurochemical_state['gaba'] - 0.3)
            print(f"Triggered return of repressed pattern for {len(repressed_active)} signifiers")
        
        # Signifier-specific patterns (customize based on agent's unconscious)
        for sig_data in active_signifiers:
            sig_name = sig_data['signifier']
            
            # Example patterns - these should be customized per agent
            if 'loss' in sig_name.lower() or 'missing' in sig_name.lower():
                neuroproxy.neurochemical_state['serotonin'] = max(0.0, neuroproxy.neurochemical_state['serotonin'] - 0.1)
                neuroproxy.neurochemical_state['oxytocin'] = max(0.0, neuroproxy.neurochemical_state['oxytocin'] - 0.1)
            
            if 'fear' in sig_data['signifier_data'].get('associations', []):
                neuroproxy.neurochemical_state['cortisol'] = min(1.0, neuroproxy.neurochemical_state['cortisol'] + 0.2)
    
    def _update_unconscious_state(self, influence: Dict[str, Any]) -> None:
        """Update unconscious processor's internal state."""
        self.active_signifiers = influence.get('active_signifiers', [])
        self.active_chains = influence.get('activated_chains', [])
        
    def _reconstruct_graph(self, graph_data: Dict) -> LacanianSignifierGraph:
        """Reconstruct signifier graph from serialized data."""
        graph = LacanianSignifierGraph()
        
        # Rebuild nodes
        for node in graph_data.get('nodes', []):
            node_copy = node.copy()
            node_id = node_copy.pop('id')
            graph.graph.add_node(node_id, **node_copy)
        
        # Rebuild edges
        for edge in graph_data.get('edges', []):
            edge_copy = edge.copy()
            source = edge_copy.pop('source')
            target = edge_copy.pop('target')
            graph.graph.add_edge(source, target, **edge_copy)
        
        # Restore structures
        graph.master_signifiers = graph_data.get('master_signifiers', {})
        graph.signifying_chains = graph_data.get('signifying_chains', {})
        graph.object_a_positions = graph_data.get('object_a_positions', [])
        graph.retroactive_effects = graph_data.get('retroactive_effects', {})
        
        return graph
    
    def _initialize_empty_structures(self):
        """Initialize empty structures if loading fails."""
        self.signifier_graph = LacanianSignifierGraph()
        self.signifiers = []
        self.chains = []
        self.object_a = {}
        self.symptom = {}
        self.structural_positions = {'hysteric': 0.4, 'master': 0.3, 'university': 0.2, 'analyst': 0.1}
        self._prepare_unconscious_data()
    
    def _minimal_unconscious_response(self) -> Dict[str, Any]:
        """Provide minimal response when processing fails."""
        return {
            "active_signifiers": [],
            "activated_chains": [],
            "object_a_effects": {"proximity_level": 0.0},
            "symptom_effects": {"activation_level": 0.0},
            "discourse_position": self.structural_positions,
            "timestamp": datetime.now().isoformat()
        }