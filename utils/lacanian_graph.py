import networkx as nx
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

class LacanianSignifierGraph:
    """
    Graph implementing Lacanian signifier dynamics with S1/S2 distinction.
    
    Represents the unconscious as structured like a language, with master signifiers (S1)
    anchoring chains of knowledge signifiers (S2) through retroactive determination.
    """
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.master_signifiers = {}  # S1 - Points de capiton
        self.signifying_chains = {}  # Named chains with logic
        self.object_a_positions = []  # Where object a manifests as void
        self.quilting_points = []  # Points de capiton that fix meaning
        self.retroactive_effects = {}  # Nachträglichkeit mappings
        
    def add_master_signifier(self, s1: str, anchoring_function: str = "", primal_repression: bool = False):
        """Add master signifier (S1) that anchors meaning chains."""
        self.graph.add_node(
            s1,
            node_type='S1',
            anchoring_function=anchoring_function,
            primal_repression=primal_repression,
            inscription_time=datetime.now(),
            retroactive_meanings=[]
        )
        self.master_signifiers[s1] = {
            'anchors': [],
            'void_relation': None,
            'jouissance_mode': None
        }
        return self
    
    def add_knowledge_signifier(self, s2: str, associations: List[str], 
                               metaphoric_substitutions: Optional[List[str]] = None):
        """Add knowledge signifier (S2) that forms chains of meaning."""
        self.graph.add_node(
            s2,
            node_type='S2',
            associations=associations,
            metaphoric_substitutions=metaphoric_substitutions or [],
            activation=0.0,
            repressed=False,
            timestamp=datetime.now()
        )
        
        # Create metonymic links to associations
        for assoc in associations:
            if assoc in self.graph:
                self.add_metonymic_link(s2, assoc, association_type='contiguity')
        
        return self
    
    def add_metonymic_link(self, s1: str, s2: str, association_type: str = 'displacement'):
        """Add metonymic (horizontal) link in signifying chain."""
        self.graph.add_edge(
            s1, s2,
            edge_type='metonymy',
            association_type=association_type,
            weight=0.7,
            timestamp=datetime.now()
        )
        return self
    
    def add_metaphoric_link(self, signifier: str, substitute: str, repressed_content: Optional[str] = None):
        """Add metaphoric (vertical) substitution link."""
        self.graph.add_edge(
            signifier, substitute,
            edge_type='metaphor',
            repressed_content=repressed_content,
            substitution_type='paradigmatic',
            weight=0.9,
            timestamp=datetime.now()
        )
        
        # Mark original as potentially repressed
        if repressed_content and signifier in self.graph:
            self.graph.nodes[signifier]['repressed'] = True
            self.graph.nodes[signifier]['repressed_content'] = repressed_content
        
        return self
    
    def create_signifying_chain(self, chain_name: str, signifiers: List[str], 
                               chain_type: str = 'mixed', retroactive_meaning: bool = True):
        """Create named signifying chain with Lacanian logic."""
        chain_data = {
            'name': chain_name,
            'signifiers': signifiers,
            'type': chain_type,
            'retroactive': retroactive_meaning,
            'quilting_points': [],
            'slippage_points': []
        }
        
        # Build the chain
        for i in range(len(signifiers) - 1):
            if chain_type == 'metonymic' or chain_type == 'mixed':
                self.add_metonymic_link(signifiers[i], signifiers[i+1])
            
        # Implement retroactive determination (Nachträglichkeit)
        if retroactive_meaning and len(signifiers) > 1:
            last_signifier = signifiers[-1]
            self.retroactive_effects[chain_name] = {
                'determining_signifier': last_signifier,
                'retroactive_targets': signifiers[:-1],
                'meaning_effect': f"{last_signifier} retroactively determines meaning"
            }
            
            # Add retroactive edges
            for target in signifiers[:-1]:
                if target in self.graph:
                    self.graph.add_edge(
                        last_signifier, target,
                        edge_type='retroactive',
                        effect='meaning_determination',
                        timestamp=datetime.now()
                    )
        
        self.signifying_chains[chain_name] = chain_data
        return self
    
    def add_quilting_point(self, signifier: str, chains_to_quilt: List[str]):
        """Add point de capiton that fixes meaning across chains."""
        if signifier not in self.graph:
            self.add_master_signifier(signifier, anchoring_function="quilting point")
        
        quilting_data = {
            'signifier': signifier,
            'quilted_chains': chains_to_quilt,
            'fixing_function': f"{signifier} arrests sliding of signification",
            'timestamp': datetime.now()
        }
        
        # Create quilting edges
        for chain_name in chains_to_quilt:
            if chain_name in self.signifying_chains:
                chain = self.signifying_chains[chain_name]
                for s in chain['signifiers']:
                    if s in self.graph and s != signifier:
                        self.graph.add_edge(
                            signifier, s,
                            edge_type='quilting',
                            chain=chain_name,
                            effect='meaning_fixation'
                        )
                chain['quilting_points'].append(signifier)
        
        self.quilting_points.append(quilting_data)
        return self
    
    def mark_object_a_void(self, position: str, surrounding_signifiers: List[str]):
        """Mark where object a appears as void in signifying structure."""
        void_data = {
            'position': position,
            'surrounding_signifiers': surrounding_signifiers,
            'void_type': 'object_cause_of_desire',
            'effects': []
        }
        
        # Object a creates gravitational effect on surrounding signifiers
        for signifier in surrounding_signifiers:
            if signifier in self.graph:
                self.graph.nodes[signifier]['object_a_proximity'] = True
                self.graph.nodes[signifier]['desire_vector'] = position
                void_data['effects'].append(f"{signifier} circles around void")
        
        self.object_a_positions.append(void_data)
        return self
    
    def apply_repression(self, signifier: str, return_formation: Optional[str] = None):
        """Apply repression to signifier with optional return formation."""
        if signifier in self.graph:
            self.graph.nodes[signifier]['repressed'] = True
            self.graph.nodes[signifier]['repression_timestamp'] = datetime.now()
            
            if return_formation:
                # Repressed returns in distorted form
                self.add_metaphoric_link(
                    signifier, 
                    return_formation,
                    repressed_content=f"repressed form of {signifier}"
                )
                
                if return_formation in self.graph:
                    self.graph.nodes[return_formation]['return_of_repressed'] = True
                    self.graph.nodes[return_formation]['original_signifier'] = signifier
        
        return self
    
    def trace_desire_path(self, start_signifier: str, max_depth: int = 5) -> List[Tuple[str, str]]:
        """Trace path of desire through signifying chain."""
        if start_signifier not in self.graph:
            return []
        
        desire_path = []
        current = start_signifier
        visited = set()
        depth = 0
        
        while depth < max_depth and current not in visited:
            visited.add(current)
            
            # Find next signifier based on metonymic displacement
            next_signifiers = [
                (n, d) for n, d in self.graph[current].items()
                if d.get('edge_type') == 'metonymy'
            ]
            
            if not next_signifiers:
                # Check for metaphoric substitution
                next_signifiers = [
                    (n, d) for n, d in self.graph[current].items()
                    if d.get('edge_type') == 'metaphor'
                ]
            
            if next_signifiers:
                # Desire follows path of greatest weight
                next_node = max(next_signifiers, key=lambda x: x[1].get('weight', 0))
                desire_path.append((current, next_node[0]))
                current = next_node[0]
                depth += 1
            else:
                break
        
        return desire_path
    
    def identify_jouissance_points(self) -> List[Dict[str, Any]]:
        """Identify points of jouissance (painful enjoyment) in structure."""
        jouissance_points = []
        
        # Look for repetition compulsion patterns
        for node in self.graph.nodes():
            # Check for cycles indicating repetition
            try:
                cycles = list(nx.simple_cycles(self.graph))
                for cycle in cycles:
                    if node in cycle and len(cycle) > 2:
                        jouissance_points.append({
                            'type': 'repetition_compulsion',
                            'signifier': node,
                            'cycle': cycle,
                            'interpretation': f"Compulsive return to {node}"
                        })
            except:
                pass
            
            # Check for proximity to object a
            if self.graph.nodes[node].get('object_a_proximity', False):
                jouissance_points.append({
                    'type': 'proximity_to_void',
                    'signifier': node,
                    'interpretation': f"{node} circles around void of object a"
                })
            
            # Check for repressed signifiers (jouissance of symptom)
            if self.graph.nodes[node].get('repressed', False):
                jouissance_points.append({
                    'type': 'symptomatic_jouissance',
                    'signifier': node,
                    'interpretation': f"{node} provides jouissance through repression"
                })
        
        return jouissance_points
    
    def analyze_discourse_position(self, active_signifiers: List[str]) -> Dict[str, float]:
        """Analyze which of four discourses is active based on signifier positions."""
        discourse_scores = {
            'master': 0.0,
            'university': 0.0,
            'hysteric': 0.0,
            'analyst': 0.0
        }
        
        # Analyze structural positions
        for signifier in active_signifiers:
            if signifier not in self.graph:
                continue
                
            node_data = self.graph.nodes[signifier]
            
            # Master's discourse: S1 → S2
            if node_data.get('node_type') == 'S1':
                s2_targets = [n for n in self.graph.successors(signifier) 
                             if self.graph.nodes.get(n, {}).get('node_type') == 'S2']
                if s2_targets:
                    discourse_scores['master'] += 0.3
            
            # University discourse: S2 → a
            elif node_data.get('node_type') == 'S2':
                if node_data.get('object_a_proximity', False):
                    discourse_scores['university'] += 0.3
            
            # Hysteric's discourse: $ → S1
            if node_data.get('repressed', False) or node_data.get('return_of_repressed', False):
                s1_targets = [n for n in self.graph.successors(signifier)
                             if self.graph.nodes.get(n, {}).get('node_type') == 'S1']
                if s1_targets:
                    discourse_scores['hysteric'] += 0.3
            
            # Analyst's discourse: a → $
            if node_data.get('object_a_proximity', False):
                repressed_targets = [n for n in self.graph.successors(signifier)
                                   if self.graph.nodes.get(n, {}).get('repressed', False)]
                if repressed_targets:
                    discourse_scores['analyst'] += 0.3
        
        # Normalize scores
        total = sum(discourse_scores.values())
        if total > 0:
            for discourse in discourse_scores:
                discourse_scores[discourse] /= total
        else:
            # Default distribution if no clear pattern
            discourse_scores = {'hysteric': 0.4, 'master': 0.3, 'university': 0.2, 'analyst': 0.1}
        
        return discourse_scores
    
    def get_signifier_resonance(self, signifier: str, depth: int = 3) -> Dict[str, float]:
        """Calculate how signifier resonates through network."""
        if signifier not in self.graph:
            return {}
        
        resonance = {signifier: 1.0}
        current_level = {signifier: 1.0}
        
        for d in range(depth):
            next_level = {}
            decay = 0.7 ** (d + 1)
            
            for node, strength in current_level.items():
                # Forward resonance (metonymy)
                for successor in self.graph.successors(node):
                    edge_data = self.graph[node][successor]
                    edge_weight = edge_data.get('weight', 0.5)
                    
                    # Different decay for different edge types
                    if edge_data.get('edge_type') == 'metaphor':
                        transfer = strength * edge_weight * 0.9
                    elif edge_data.get('edge_type') == 'metonymy':
                        transfer = strength * edge_weight * 0.7
                    elif edge_data.get('edge_type') == 'retroactive':
                        transfer = strength * edge_weight * 1.1
                    else:
                        transfer = strength * edge_weight * decay
                    
                    if successor not in resonance:
                        resonance[successor] = 0
                    resonance[successor] += transfer
                    
                    if successor not in next_level:
                        next_level[successor] = 0
                    next_level[successor] += transfer
                
                # Backward resonance (retroactive determination)
                for predecessor in self.graph.predecessors(node):
                    edge_data = self.graph[predecessor][node]
                    if edge_data.get('edge_type') == 'retroactive':
                        transfer = strength * 0.8
                        if predecessor not in resonance:
                            resonance[predecessor] = 0
                        resonance[predecessor] += transfer
            
            current_level = next_level
        
        return resonance
    
    def find_fantasy_structure(self) -> Dict[str, Any]:
        """Identify fundamental fantasy structure ($ ◊ a)."""
        fantasy = {
            'divided_subjects': [],
            'object_a_manifestations': [],
            'relations': [],
            'defensive_formations': []
        }
        
        # Find divided subject positions (repressed/symptomatic nodes)
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]
            if node_data.get('repressed', False) or node_data.get('return_of_repressed', False):
                fantasy['divided_subjects'].append({
                    'signifier': node,
                    'division_type': 'repression' if node_data.get('repressed') else 'return'
                })
        
        # Find object a manifestations
        for void in self.object_a_positions:
            fantasy['object_a_manifestations'].append(void)
        
        # Analyze relations between $ and a
        for subject in fantasy['divided_subjects']:
            for obj_a in fantasy['object_a_manifestations']:
                if subject['signifier'] in obj_a['surrounding_signifiers']:
                    fantasy['relations'].append({
                        'subject': subject['signifier'],
                        'object_a': obj_a['position'],
                        'relation_type': 'circling',
                        'interpretation': f"{subject['signifier']} desires around void of {obj_a['position']}"
                    })
        
        # Identify defensive formations
        for node in self.graph.nodes():
            if self.graph.nodes[node].get('node_type') == 'S1':
                fantasy['defensive_formations'].append({
                    'signifier': node,
                    'defense_type': 'anchoring',
                    'function': self.graph.nodes[node].get('anchoring_function', 'unknown')
                })
        
        return fantasy
    
    def detect_slippage_points(self) -> List[Dict[str, Any]]:
        """Detect where meaning slips in signifying chain."""
        slippage_points = []
        
        for chain_name, chain_data in self.signifying_chains.items():
            signifiers = chain_data['signifiers']
            
            for i in range(len(signifiers) - 1):
                current = signifiers[i]
                next_sig = signifiers[i + 1]
                
                if current in self.graph and next_sig in self.graph:
                    # Check for weak metonymic links
                    if self.graph.has_edge(current, next_sig):
                        edge_data = self.graph[current][next_sig]
                        if edge_data.get('weight', 1.0) < 0.5:
                            slippage_points.append({
                                'chain': chain_name,
                                'position': f"{current} -> {next_sig}",
                                'slippage_type': 'weak_link',
                                'interpretation': 'Meaning may slip at this connection'
                            })
                    
                    # Check for multiple possible paths (overdetermination)
                    successors = list(self.graph.successors(current))
                    if len(successors) > 2:
                        slippage_points.append({
                            'chain': chain_name,
                            'position': current,
                            'slippage_type': 'overdetermination',
                            'possible_paths': successors,
                            'interpretation': 'Multiple meaning paths create slippage'
                        })
        
        return slippage_points
    
    def serialize(self) -> Dict[str, Any]:
        """Serialize graph for storage."""
        nodes = []
        for node_id, node_data in self.graph.nodes(data=True):
            node_dict = {'id': node_id}
            for key, value in node_data.items():
                if isinstance(value, datetime):
                    node_dict[key] = value.isoformat()
                else:
                    node_dict[key] = value
            nodes.append(node_dict)
        
        edges = []
        for source, target, edge_data in self.graph.edges(data=True):
            edge_dict = {'source': source, 'target': target}
            for key, value in edge_data.items():
                if isinstance(value, datetime):
                    edge_dict[key] = value.isoformat()
                else:
                    edge_dict[key] = value
            edges.append(edge_dict)
        
        return {
            'nodes': nodes,
            'edges': edges,
            'master_signifiers': self.master_signifiers,
            'signifying_chains': self.signifying_chains,
            'object_a_positions': self.object_a_positions,
            'quilting_points': self.quilting_points,
            'retroactive_effects': self.retroactive_effects,
            'metadata': {
                'node_count': self.graph.number_of_nodes(),
                'edge_count': self.graph.number_of_edges(),
                'chain_count': len(self.signifying_chains),
                'quilting_point_count': len(self.quilting_points)
            }
        }