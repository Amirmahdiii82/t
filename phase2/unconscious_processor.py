import json
import random
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from interfaces.llm_interface import LLMInterface
from utils.lacanian_graph import LacanianSignifierGraph

class UnconsciousProcessor:
    """Process interactions through genuine Lacanian unconscious dynamics."""
    
    def __init__(self, agent_name: str, memory_manager, dream_generator):
        self.agent_name = agent_name
        self.memory_manager = memory_manager
        self.dream_generator = dream_generator
        self.llm = LLMInterface()
        
        # Load unconscious structures
        self._load_unconscious_structures()
        
        # Initialize psychoanalytic components
        self.active_signifiers = []
        self.repressed_content = {}
        self.transference_patterns = {}
        self.resistance_points = []
        self.current_discourse = "hysteric"  # Default position
        
        print(f"Unconscious Processor initialized for {agent_name}")
    
    def _load_unconscious_structures(self):
        """Load the agent's unconscious structures."""
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
            self.jouissance_economy = unconscious_data.get('jouissance_economy', {})
            self.fantasy_formula = unconscious_data.get('fundamental_fantasy', '')
            
        except Exception as e:
            print(f"Error loading unconscious structures: {e}")
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
        
        # Restore other structures
        graph.master_signifiers = graph_data.get('master_signifiers', {})
        graph.signifying_chains = graph_data.get('signifying_chains', {})
        graph.object_a_positions = graph_data.get('object_a_positions', [])
        graph.quilting_points = graph_data.get('quilting_points', [])
        graph.retroactive_effects = graph_data.get('retroactive_effects', {})
        
        return graph
    
    def process_input(self, user_input: str, context: str = "dialogue") -> Dict[str, Any]:
        """Process input through unconscious mechanisms."""
        try:
            # 1. Identify active signifiers in the input
            active_signifiers = self._identify_active_signifiers(user_input)
            
            # 2. Check for transference patterns
            transference = self._analyze_transference(user_input, context)
            
            # 3. Detect resistance
            resistance = self._detect_resistance(user_input, active_signifiers)
            
            # 4. Process through signifying chains
            chain_effects = self._process_through_chains(active_signifiers)
            
            # 5. Check for return of repressed
            repressed_returns = self._check_return_of_repressed(active_signifiers)
            
            # 6. Analyze discourse position
            discourse_position = self.signifier_graph.analyze_discourse_position(active_signifiers)
            
            # 7. Calculate jouissance dynamics
            jouissance_effects = self._calculate_jouissance_effects(active_signifiers, user_input)
            
            # 8. Generate unconscious influence
            unconscious_influence = {
                "active_signifiers": active_signifiers,
                "chain_activations": chain_effects,
                "transference": transference,
                "resistance": resistance,
                "repressed_returns": repressed_returns,
                "discourse_position": discourse_position,
                "jouissance_effects": jouissance_effects,
                "fantasy_activation": self._check_fantasy_activation(active_signifiers),
                "slips_and_parapraxes": self._generate_parapraxes(user_input, active_signifiers)
            }
            
            # Update internal state
            self._update_unconscious_state(unconscious_influence)
            
            return unconscious_influence
            
        except Exception as e:
            print(f"Error in unconscious processing: {e}")
            return self._minimal_unconscious_response()
    
    def _identify_active_signifiers(self, text: str) -> List[Dict[str, Any]]:
        """Identify which unconscious signifiers are activated by the input."""
        active = []
        text_lower = text.lower()
        
        # Check each signifier in the graph
        for node in self.signifier_graph.graph.nodes():
            node_data = self.signifier_graph.graph.nodes[node]
            
            # Direct matching
            if node.lower() in text_lower:
                active.append({
                    'signifier': node,
                    'activation_type': 'direct',
                    'strength': 1.0,
                    'node_type': node_data.get('node_type', 'S2')
                })
                continue
            
            # Check associations
            associations = node_data.get('associations', [])
            for assoc in associations:
                if isinstance(assoc, str) and assoc.lower() in text_lower:
                    active.append({
                        'signifier': node,
                        'activation_type': 'associative',
                        'strength': 0.7,
                        'triggered_by': assoc,
                        'node_type': node_data.get('node_type', 'S2')
                    })
                    break
            
            # Check for phonetic similarities (signifier slippage)
            if self._check_phonetic_similarity(node, text):
                active.append({
                    'signifier': node,
                    'activation_type': 'phonetic_slippage',
                    'strength': 0.5,
                    'node_type': node_data.get('node_type', 'S2')
                })
        
        # Calculate resonance effects
        for activation in active:
            signifier = activation['signifier']
            resonance = self.signifier_graph.get_signifier_resonance(signifier, depth=2)
            activation['resonance'] = resonance
        
        return active
    
    def _analyze_transference(self, text: str, context: str) -> Dict[str, Any]:
        """Analyze transference patterns in the interaction."""
        transference = {
            'type': 'neutral',
            'intensity': 0.0,
            'projected_figures': [],
            'unconscious_address': None
        }
        
        # Analyze who the subject is really addressing
        prompt = f"""
        Analyze the transference in this interaction from a Lacanian perspective.
        
        User input: "{text}"
        Context: {context}
        Active symptom: {json.dumps(self.symptom, indent=2)}
        
        Consider:
        1. Who is the subject really addressing? (The Other, a parental figure, themselves?)
        2. What unconscious demand is being made?
        3. Is there positive or negative transference?
        4. What position is the subject putting the analyst/agent in?
        
        Return a brief JSON analysis.
        """
        
        analysis = self.llm.generate(None, None, prompt)
        try:
            transference_data = json.loads(analysis)
            transference.update(transference_data)
        except:
            # Extract key patterns manually
            if any(word in text.lower() for word in ['you always', 'you never', 'why don\'t you']):
                transference['type'] = 'negative'
                transference['intensity'] = 0.7
                transference['projected_figures'].append('critical parent')
            elif any(word in text.lower() for word in ['help me', 'tell me', 'i need']):
                transference['type'] = 'positive'
                transference['intensity'] = 0.6
                transference['projected_figures'].append('idealized helper')
        
        return transference
    
    def _detect_resistance(self, text: str, active_signifiers: List[Dict]) -> Dict[str, Any]:
        """Detect resistance to unconscious material."""
        resistance = {
            'present': False,
            'type': None,
            'intensity': 0.0,
            'defensive_operations': []
        }
        
        # Check if any active signifiers are repressed
        for sig_data in active_signifiers:
            signifier = sig_data['signifier']
            if signifier in self.signifier_graph.graph:
                node_data = self.signifier_graph.graph.nodes[signifier]
                if node_data.get('repressed', False):
                    resistance['present'] = True
                    resistance['intensity'] = max(resistance['intensity'], 0.8)
                    resistance['defensive_operations'].append({
                        'type': 'repression',
                        'signifier': signifier
                    })
        
        # Check for common resistance patterns
        resistance_phrases = [
            "i don't want to talk about",
            "that's not important",
            "let's change the subject",
            "i don't remember",
            "that's irrelevant"
        ]
        
        text_lower = text.lower()
        for phrase in resistance_phrases:
            if phrase in text_lower:
                resistance['present'] = True
                resistance['type'] = 'avoidance'
                resistance['intensity'] = max(resistance['intensity'], 0.6)
                resistance['defensive_operations'].append({
                    'type': 'verbal_avoidance',
                    'indicator': phrase
                })
        
        return resistance
    
    def _process_through_chains(self, active_signifiers: List[Dict]) -> Dict[str, Any]:
        """Process active signifiers through signifying chains."""
        chain_effects = {
            'activated_chains': [],
            'meaning_effects': [],
            'retroactive_determinations': [],
            'slippage_points': []
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
                
                # Check for retroactive effects
                if chain_data.get('retroactive', False) and chain_signifiers:
                    last_sig = chain_signifiers[-1]
                    if last_sig in active_in_chain:
                        chain_effects['retroactive_determinations'].append({
                            'determining_signifier': last_sig,
                            'retroactive_meaning': f"{last_sig} determines meaning of {chain_name}",
                            'affected_signifiers': [s for s in chain_signifiers if s != last_sig]
                        })
        
        # Detect slippage points
        slippage = self.signifier_graph.detect_slippage_points()
        for point in slippage:
            # Check if slippage point is near active signifiers
            if any(s['signifier'] in point.get('position', '') for s in active_signifiers):
                chain_effects['slippage_points'].append(point)
        
        return chain_effects
    
    def _check_return_of_repressed(self, active_signifiers: List[Dict]) -> List[Dict[str, Any]]:
        """Check if repressed signifiers are returning in distorted form."""
        returns = []
        
        for sig_data in active_signifiers:
            signifier = sig_data['signifier']
            if signifier in self.signifier_graph.graph:
                node_data = self.signifier_graph.graph.nodes[signifier]
                
                # Check if this is a return formation
                if node_data.get('return_of_repressed', False):
                    original = node_data.get('original_signifier', 'unknown')
                    returns.append({
                        'return_formation': signifier,
                        'repressed_signifier': original,
                        'return_type': sig_data['activation_type'],
                        'interpretation': f"{original} returns as {signifier}"
                    })
                
                # Check for metaphoric substitutions that might be returns
                substitutions = node_data.get('metaphoric_substitutions', [])
                for sub in substitutions:
                    if sub in self.signifier_graph.graph:
                        sub_data = self.signifier_graph.graph.nodes[sub]
                        if sub_data.get('repressed', False):
                            returns.append({
                                'return_formation': signifier,
                                'repressed_signifier': sub,
                                'return_type': 'metaphoric_substitution',
                                'interpretation': f"Repressed {sub} returns through {signifier}"
                            })
        
        return returns
    
    def _calculate_jouissance_effects(self, active_signifiers: List[Dict], text: str) -> Dict[str, Any]:
        """Calculate jouissance dynamics in the current interaction."""
        jouissance = {
            'level': 0.0,
            'type': 'neutral',
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
                    'description': 'Subject derives jouissance from symptom'
                })
        
        # Check for repetition compulsion
        jouissance_points = self.signifier_graph.identify_jouissance_points()
        for point in jouissance_points:
            if point['signifier'] in [s['signifier'] for s in active_signifiers]:
                jouissance['level'] += 0.2
                jouissance['patterns'].append(point)
        
        # Analyze proximity to object a
        for sig_data in active_signifiers:
            signifier = sig_data['signifier']
            if signifier in self.signifier_graph.graph:
                if self.signifier_graph.graph.nodes[signifier].get('object_a_proximity', False):
                    jouissance['level'] += 0.3
                    jouissance['type'] = 'object_a_circulation'
                    jouissance['patterns'].append({
                        'type': 'desire_around_void',
                        'signifier': signifier,
                        'description': f"{signifier} circles the void of object a"
                    })
        
        # Cap jouissance level
        jouissance['level'] = min(1.0, jouissance['level'])
        
        return jouissance
    
    def _check_fantasy_activation(self, active_signifiers: List[Dict]) -> Dict[str, Any]:
        """Check if the fundamental fantasy is activated."""
        fantasy_activation = {
            'activated': False,
            'strength': 0.0,
            'elements': []
        }
        
        # Get fantasy structure
        fantasy = self.signifier_graph.find_fantasy_structure()
        
        # Check if divided subject positions are active
        active_names = [s['signifier'] for s in active_signifiers]
        for subject in fantasy.get('divided_subjects', []):
            if subject['signifier'] in active_names:
                fantasy_activation['activated'] = True
                fantasy_activation['strength'] += 0.3
                fantasy_activation['elements'].append({
                    'type': 'divided_subject',
                    'signifier': subject['signifier']
                })
        
        # Check if object a positions are referenced
        for obj_a in fantasy.get('object_a_manifestations', []):
            surrounding = obj_a.get('surrounding_signifiers', [])
            if any(sig in active_names for sig in surrounding):
                fantasy_activation['activated'] = True
                fantasy_activation['strength'] += 0.4
                fantasy_activation['elements'].append({
                    'type': 'object_a_proximity',
                    'position': obj_a['position']
                })
        
        return fantasy_activation
    
    def _generate_parapraxes(self, text: str, active_signifiers: List[Dict]) -> List[Dict[str, Any]]:
        """Generate potential slips and parapraxes based on unconscious activity."""
        parapraxes = []
        
        # High unconscious activation can cause slips
        if len(active_signifiers) > 3:
            # Potential for substitution
            for sig_data in active_signifiers[:2]:  # Limit to avoid too many slips
                signifier = sig_data['signifier']
                if random.random() < 0.3:  # 30% chance
                    # Find a related signifier for substitution
                    if signifier in self.signifier_graph.graph:
                        neighbors = list(self.signifier_graph.graph.neighbors(signifier))
                        if neighbors:
                            substitute = random.choice(neighbors)
                            parapraxes.append({
                                'type': 'substitution',
                                'intended': signifier,
                                'slip': substitute,
                                'interpretation': f"Unconscious substitution revealing connection"
                            })
        
        # Repressed content trying to emerge
        repressed_returns = self._check_return_of_repressed(active_signifiers)
        if repressed_returns:
            parapraxes.append({
                'type': 'return_of_repressed',
                'content': repressed_returns[0],
                'interpretation': 'Repressed content breaking through'
            })
        
        return parapraxes
    
    def _check_phonetic_similarity(self, signifier: str, text: str) -> bool:
        """Check for phonetic similarities that might trigger signifiers."""
        # Simple phonetic matching - could be enhanced with proper phonetic algorithms
        words = text.lower().split()
        sig_lower = signifier.lower()
        
        for word in words:
            # Check for rhymes
            if len(word) > 2 and len(sig_lower) > 2:
                if word[-2:] == sig_lower[-2:]:
                    return True
            # Check for alliteration
            if word[0] == sig_lower[0] and len(word) > 3:
                return True
        
        return False
    
    def _update_unconscious_state(self, influence: Dict[str, Any]) -> None:
        """Update the unconscious processor's state based on current processing."""
        # Update active signifiers list
        self.active_signifiers = influence.get('active_signifiers', [])
        
        # Update discourse position
        discourse_scores = influence.get('discourse_position', {})
        if discourse_scores:
            self.current_discourse = max(discourse_scores.items(), key=lambda x: x[1])[0]
        
        # Track resistance patterns
        resistance = influence.get('resistance', {})
        if resistance.get('present', False):
            self.resistance_points.append({
                'timestamp': datetime.now(),
                'resistance': resistance
            })
            # Keep only recent resistance
            self.resistance_points = self.resistance_points[-10:]
        
        # Update transference patterns
        transference = influence.get('transference', {})
        if transference.get('intensity', 0) > 0.5:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.transference_patterns[session_id] = transference
    
    def generate_unconscious_response_influence(self, conscious_response: str) -> str:
        """Apply unconscious influence to the conscious response."""
        if not self.active_signifiers:
            return conscious_response
        
        try:
            # Get current unconscious state
            current_influence = {
                'active_signifiers': self.active_signifiers,
                'discourse': self.current_discourse,
                'resistance': self.resistance_points[-1] if self.resistance_points else None,
                'fantasy_activation': self._check_fantasy_activation(self.active_signifiers)
            }
            
            prompt = f"""
            Apply unconscious influence to this conscious response from a Lacanian perspective.
            
            Conscious response: "{conscious_response}"
            
            Unconscious state:
            - Active signifiers: {json.dumps([s['signifier'] for s in self.active_signifiers[:5]])}
            - Current discourse position: {self.current_discourse}
            - Fantasy activated: {current_influence['fantasy_activation']['activated']}
            
            Modify the response to include:
            1. Subtle references to active signifiers (not obvious)
            2. Speech patterns reflecting the current discourse position
            3. Potential slips or ambiguities
            4. Defensive formations if resistance is present
            
            The modification should be subtle - the unconscious speaks between the lines.
            Return only the modified response.
            """
            
            influenced_response = self.llm.generate(None, None, prompt)
            
            # Add parapraxes if highly activated
            if len(self.active_signifiers) > 4 and random.random() < 0.3:
                parapraxes = self._generate_parapraxes(conscious_response, self.active_signifiers)
                if parapraxes:
                    # Insert a slip
                    slip = parapraxes[0]
                    if slip['type'] == 'substitution':
                        influenced_response = influenced_response.replace(
                            slip['intended'], 
                            f"{slip['slip']}... I mean, {slip['intended']}", 
                            1
                        )
            
            return influenced_response if influenced_response else conscious_response
            
        except Exception as e:
            print(f"Error generating unconscious influence: {e}")
            return conscious_response
    
    def _initialize_empty_structures(self):
        """Initialize empty structures if loading fails."""
        self.signifier_graph = LacanianSignifierGraph()
        self.signifiers = []
        self.chains = []
        self.object_a = {}
        self.symptom = {}
        self.jouissance_economy = {}
        self.fantasy_formula = ""
    
    def _minimal_unconscious_response(self) -> Dict[str, Any]:
        """Provide minimal response when processing fails."""
        return {
            "active_signifiers": [],
            "chain_activations": {"activated_chains": []},
            "transference": {"type": "neutral", "intensity": 0.0},
            "resistance": {"present": False},
            "repressed_returns": [],
            "discourse_position": {"hysteric": 0.25, "master": 0.25, "university": 0.25, "analyst": 0.25},
            "jouissance_effects": {"level": 0.0, "patterns": []},
            "fantasy_activation": {"activated": False},
            "slips_and_parapraxes": []
        }