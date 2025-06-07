import networkx as nx
import spacy
from datetime import datetime

class SignifierGraph:
    """A directed graph representing unconscious signifiers and their relationships."""
    
    def __init__(self):
        """Initialize the signifier graph."""
        self.graph = nx.DiGraph()
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            import subprocess
            import sys
            subprocess.call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
    
    def add_node(self, node_id, **attr):
        """Add a node with required attributes."""
        # Set default values for required attributes
        node_attr = {
            'type': attr.get('type', 'symbolic'),  # symbolic, imaginary, or real
            'activation': attr.get('activation', 0.0),  # 0.0 to 1.0
            'repressed': attr.get('repressed', False),  # Boolean or float 0.0-1.0
            'timestamp': attr.get('timestamp', datetime.now())  # Current time by default
        }
        
        # Validate attributes
        if node_attr['type'] not in ['symbolic', 'imaginary', 'real']:
            node_attr['type'] = 'symbolic'
        
        node_attr['activation'] = max(0.0, min(1.0, node_attr['activation']))
        
        if isinstance(node_attr['repressed'], (int, float)):
            node_attr['repressed'] = max(0.0, min(1.0, node_attr['repressed']))
        
        # Add any additional attributes
        for key, value in attr.items():
            if key not in node_attr:
                node_attr[key] = value
        
        # Add node to graph
        self.graph.add_node(node_id, **node_attr)
        return self
    
    def add_edge(self, source, target, **attr):
        """Add an edge with required attributes."""
        # Set default values for required attributes
        edge_attr = {
            'weight': attr.get('weight', 0.5),  # 0.0 to 1.0
            'type': attr.get('type', 'neutral'),  # condensation, displacement, or neutral
            'context': attr.get('context', [])  # List of context strings
        }
        
        # Validate attributes
        if edge_attr['type'] not in ['condensation', 'displacement', 'neutral']:
            edge_attr['type'] = 'neutral'
        
        edge_attr['weight'] = max(0.0, min(1.0, edge_attr['weight']))
        
        if not isinstance(edge_attr['context'], list):
            edge_attr['context'] = [str(edge_attr['context'])]
        
        # Add any additional attributes
        for key, value in attr.items():
            if key not in edge_attr:
                edge_attr[key] = value
        
        # Add edge to graph
        self.graph.add_edge(source, target, **edge_attr)
        return self
    
    def get_node_attributes(self, node_id):
        """Get all attributes of a node as a dictionary."""
        if node_id in self.graph:
            return dict(self.graph.nodes[node_id])
        return {}
    
    def get_edge_data(self, source, target):
        """Get edge data between two nodes."""
        if self.graph.has_edge(source, target):
            return dict(self.graph.edges[source, target])
        return {}
    
    def nodes(self):
        """Return all nodes in the graph."""
        return self.graph.nodes()
    
    def edges(self):
        """Return all edges in the graph."""
        return self.graph.edges()
    
    def condense_nodes(self, threshold=0.75):
        """Identify and merge or connect similar nodes."""
        nodes = list(self.graph.nodes())
        merged = set()
        
        for i, node1 in enumerate(nodes):
            if node1 in merged:
                continue
                
            for j in range(i+1, len(nodes)):
                node2 = nodes[j]
                if node2 in merged:
                    continue
                
                # Skip if either node is repressed
                if self.graph.nodes[node1].get('repressed', False) or self.graph.nodes[node2].get('repressed', False):
                    continue
                
                # Calculate similarity using simple string matching
                similarity = self._calculate_similarity(node1, node2)
                
                if similarity > threshold:
                    # If same type, merge nodes
                    if self.graph.nodes[node1].get('type') == self.graph.nodes[node2].get('type'):
                        new_id = f"{node1}_{node2}"
                        # Average activation
                        new_activation = (self.graph.nodes[node1].get('activation', 0.5) + 
                                         self.graph.nodes[node2].get('activation', 0.5)) / 2
                        
                        # Create new node
                        self.add_node(
                            new_id,
                            type=self.graph.nodes[node1].get('type', 'symbolic'),
                            activation=new_activation,
                            repressed=False,
                            timestamp=datetime.now(),
                            merged_from=[node1, node2]
                        )
                        
                        # Transfer all edges
                        for pred in list(self.graph.predecessors(node1)):
                            if pred != node2:  # Avoid self-loops
                                edge_data = self.get_edge_data(pred, node1)
                                self.add_edge(pred, new_id, **edge_data)
                        
                        for succ in list(self.graph.successors(node1)):
                            if succ != node2:  # Avoid self-loops
                                edge_data = self.get_edge_data(node1, succ)
                                self.add_edge(new_id, succ, **edge_data)
                        
                        for pred in list(self.graph.predecessors(node2)):
                            if pred != node1:  # Avoid self-loops
                                edge_data = self.get_edge_data(pred, node2)
                                self.add_edge(pred, new_id, **edge_data)
                        
                        for succ in list(self.graph.successors(node2)):
                            if succ != node1:  # Avoid self-loops
                                edge_data = self.get_edge_data(node2, succ)
                                self.add_edge(new_id, succ, **edge_data)
                        
                        # Mark nodes as merged
                        merged.add(node1)
                        merged.add(node2)
                    else:
                        # If different types, strengthen connection
                        if self.graph.has_edge(node1, node2):
                            # Increase weight by 0.2, up to max 1.0
                            current_weight = self.graph.edges[node1, node2].get('weight', 0.5)
                            new_weight = min(1.0, current_weight + 0.2)
                            self.graph.edges[node1, node2]['weight'] = new_weight
                            self.graph.edges[node1, node2]['type'] = 'condensation'
                        else:
                            # Create new edge
                            self.add_edge(
                                node1, node2,
                                weight=0.7,
                                type='condensation',
                                context=['semantic_similarity']
                            )
        
        # Remove merged nodes
        for node in merged:
            if node in self.graph:
                self.graph.remove_node(node)
        
        return self
    
    def _calculate_similarity(self, node1, node2):
        """Calculate semantic similarity between two nodes using simple string matching."""
        # Simple string similarity
        s1 = node1.lower()
        s2 = node2.lower()
        
        # If one is contained in the other, high similarity
        if s1 in s2 or s2 in s1:
            return 0.8
        
        # Check for common substrings
        common = 0
        for i in range(min(len(s1), len(s2))):
            if s1[i] == s2[i]:
                common += 1
        
        # Return similarity score
        return common / max(len(s1), len(s2))
    
    def displace_association(self, source_id, target_id):
        """Create a displacement association between two nodes."""
        # Skip if direct edge already exists
        if self.graph.has_edge(source_id, target_id):
            return False
        
        # Find paths between source and target
        try:
            paths = list(nx.all_simple_paths(self.graph, source_id, target_id, cutoff=3))
            
            # If no paths or only direct path, nothing to displace
            if not paths or (len(paths) == 1 and len(paths[0]) == 2):
                return False
            
            # Get the shortest indirect path
            indirect_paths = [p for p in paths if len(p) > 2]
            if not indirect_paths:
                return False
                
            path = min(indirect_paths, key=len)
            
            # Create displacement edge
            weight = 0.3
            
            # Reduce weight if either node is repressed
            if (self.graph.nodes[source_id].get('repressed', False) or 
                self.graph.nodes[target_id].get('repressed', False)):
                weight *= 0.5
            
            self.add_edge(
                source_id, target_id,
                weight=weight,
                type='displacement',
                context=['displaced_path'],
                original_path=path
            )
            
            # Reduce weights of intermediate edges
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                current_weight = self.graph.edges[u, v].get('weight', 0.5)
                new_weight = max(0.0, current_weight - 0.1)
                self.graph.edges[u, v]['weight'] = new_weight
            
            return True
        except:
            return False
    
    def spread_activation(self, start_id, max_depth=3, repression_factor=0.5):
        """Spread activation from a starting node."""
        if start_id not in self.graph:
            return []
        
        # Initialize activation values
        activation_values = {node: 0.0 for node in self.graph.nodes()}
        activation_values[start_id] = 1.0  # Full activation for start node
        
        # BFS to spread activation
        visited = {start_id}
        queue = [(start_id, 0)]  # (node, depth)
        
        while queue:
            node, depth = queue.pop(0)
            
            if depth >= max_depth:
                continue
            
            # Get current activation
            current_activation = activation_values[node]
            
            # Apply depth decay
            depth_decay = 0.8 ** depth
            
            # Spread to neighbors
            for neighbor in self.graph.successors(node):
                if neighbor in visited:
                    continue
                    
                # Get edge weight
                edge_weight = self.graph.edges[node, neighbor].get('weight', 0.5)
                
                # Calculate repression factor
                node_repressed = self.graph.nodes[neighbor].get('repressed', False)
                if isinstance(node_repressed, bool):
                    rep_factor = repression_factor if node_repressed else 1.0
                else:  # Float repression level
                    rep_factor = 1.0 - node_repressed
                
                # Calculate new activation
                new_activation = current_activation * edge_weight * rep_factor * depth_decay
                
                # Update if higher
                if new_activation > activation_values[neighbor]:
                    activation_values[neighbor] = new_activation
                
                # Add to queue
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))
        
        # Sort by activation and return top 5
        sorted_nodes = sorted(
            [(node, act) for node, act in activation_values.items() if act > 0],
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_nodes[:5]
    
    def mark_repressed(self, node_id, level=1.0):
        """Mark a node as repressed."""
        if node_id in self.graph:
            if isinstance(level, bool):
                self.graph.nodes[node_id]['repressed'] = level
            else:
                self.graph.nodes[node_id]['repressed'] = max(0.0, min(1.0, level))
            return True
        return False
    
    def check_return_of_repressed(self, start_id):
        """Check if any repressed nodes appear in top activated nodes."""
        if not start_id or start_id not in self.graph:
            return []
            
        activated_nodes = self.spread_activation(start_id)
        
        repressed_returns = []
        for node_id, activation in activated_nodes:
            if self.graph.nodes[node_id].get('repressed', False):
                # Find indirect paths from start to this repressed node
                try:
                    paths = list(nx.all_simple_paths(self.graph, start_id, node_id, cutoff=3))
                    indirect_paths = [p for p in paths if len(p) > 2]
                    
                    if indirect_paths:
                        repressed_returns.append({
                            'node': node_id,
                            'activation': activation,
                            'path': min(indirect_paths, key=len)
                        })
                except:
                    pass
        
        return repressed_returns
    
    def update_memory(self, decay_rate=0.9, prune_threshold=0.1):
        """Update memory by decaying activations and pruning weak nodes."""
        # Decay all activations
        for node in self.graph.nodes():
            current_activation = self.graph.nodes[node].get('activation', 0.0)
            new_activation = current_activation * decay_rate
            self.graph.nodes[node]['activation'] = new_activation
            self.graph.nodes[node]['timestamp'] = datetime.now()
        
        # Identify nodes to prune
        to_prune = []
        for node in list(self.graph.nodes()):
            # Skip repressed nodes
            if self.graph.nodes[node].get('repressed', False):
                continue
                
            # Check activation and connections
            if (self.graph.nodes[node].get('activation', 0.0) < prune_threshold and
                self.graph.in_degree(node) == 0 and
                self.graph.out_degree(node) < 2):
                to_prune.append(node)
        
        # Prune nodes
        for node in to_prune:
            self.graph.remove_node(node)
        
        return to_prune

def encode_experience(text, graph):
    """Encode an experience in the signifier graph."""
    # Process text with spaCy
    doc = graph.nlp(text)
    
    # Extract signifiers (nouns, verbs, adjectives)
    signifiers = []
    for token in doc:
        if token.pos_ in ['NOUN', 'VERB', 'ADJ'] and not token.is_stop:
            signifiers.append(token.text.lower())
    
    # Add nodes to graph
    for signifier in signifiers:
        if signifier in graph.graph:
            # Update existing node
            graph.graph.nodes[signifier]['activation'] = 1.0
            graph.graph.nodes[signifier]['timestamp'] = datetime.now()
        else:
            # Add new node
            graph.add_node(
                signifier,
                type='symbolic',
                activation=1.0,
                repressed=False,
                timestamp=datetime.now()
            )
    
    # Create edges between consecutive signifiers
    for i in range(len(signifiers) - 1):
        source = signifiers[i]
        target = signifiers[i + 1]
        
        if graph.graph.has_edge(source, target):
            # Strengthen existing edge
            current_weight = graph.graph.edges[source, target].get('weight', 0.5)
            new_weight = min(1.0, current_weight + 0.1)
            graph.graph.edges[source, target]['weight'] = new_weight
            if 'context' in graph.graph.edges[source, target]:
                graph.graph.edges[source, target]['context'].append(text)
        else:
            # Create new edge
            graph.add_edge(
                source, target,
                weight=0.5,
                type='neutral',
                context=[text]
            )
    
    return signifiers

def generate_dream(graph, start_id):
    """Generate a dream narrative based on activated signifiers."""
    # Get top activated nodes
    top_nodes = graph.spread_activation(start_id, max_depth=3)
    
    if not top_nodes:
        return "I had a dream but couldn't remember it."
    
    # Extract node IDs
    node_ids = [node for node, _ in top_nodes]
    
    # Check for condensation (nodes with underscore)
    condensed_nodes = [node for node in node_ids if '_' in node]
    
    # Check for displacement
    displacement_edges = []
    for i in range(len(node_ids)):
        for j in range(i + 1, len(node_ids)):
            if graph.graph.has_edge(node_ids[i], node_ids[j]):
                edge_data = graph.get_edge_data(node_ids[i], node_ids[j])
                if edge_data.get('type') == 'displacement':
                    displacement_edges.append((node_ids[i], node_ids[j]))
    
    # Generate narrative
    narrative = "I was in a place that reminded me of "
    narrative += node_ids[0]
    
    if len(node_ids) > 1:
        narrative += f". It felt {node_ids[1]}"
    
    if len(node_ids) > 2:
        if condensed_nodes:
            narrative += f", like a mixture of {condensed_nodes[0].replace('_', ' and ')}"
        else:
            narrative += f", like {node_ids[2]}"
    
    if len(node_ids) > 3:
        if displacement_edges:
            src, tgt = displacement_edges[0]
            narrative += f". Then strangely, the {src} transformed into {tgt}"
        else:
            narrative += f". But then, it shifted to {node_ids[3]}"
    
    if len(node_ids) > 4:
        narrative += f", leaving me with a sense of {node_ids[4]}"
    
    return narrative