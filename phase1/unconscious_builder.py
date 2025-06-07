import json
import re
import time
from datetime import datetime
import spacy
from utils.signifier_graph import SignifierGraph, encode_experience
from utils.file_utils import ensure_directory, save_json

class UnconsciousBuilder:
    def __init__(self, vlm_interface):
        self.vlm_interface = vlm_interface
        self.signifier_graph = SignifierGraph()
        self.nlp = spacy.load("en_core_web_sm")

    def _serialize_graph_data(self):
        """Convert graph data to a serializable format."""
        nodes = []
        for node_id in self.signifier_graph.graph.nodes():
            # Get node attributes and ensure datetime is serialized
            attrs = self.signifier_graph.get_node_attributes(node_id)
            if 'timestamp' in attrs and isinstance(attrs['timestamp'], datetime):
                attrs['timestamp'] = attrs['timestamp'].isoformat()
            nodes.append({'id': node_id, **attrs})
        
        edges = []
        for source, target in self.signifier_graph.graph.edges():
            edge_data = self.signifier_graph.get_edge_data(source, target)
            for key, value in list(edge_data.items()):
                if isinstance(value, datetime):
                    edge_data[key] = value.isoformat()
            edges.append({'source': source, 'target': target, **edge_data})
        
        return {'nodes': nodes, 'edges': edges}
    
    def extract_json_from_response(self, text):
        """Extract JSON from a response that might contain markdown or other text."""
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        json_match = re.search(r'(\{[\s\S]*\})', text)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        signifiers = []
        chains = []
        
        signifier_matches = re.findall(r'(?:signifier|symbol):\s*"?([^",]+)"?', text, re.IGNORECASE)
        for match in signifier_matches:
            signifiers.append({"name": match.strip(), "associations": [], "significance": "Extracted from analysis"})
        
        list_items = re.findall(r'- ([^:]+)(?::|$)', text)
        for item in list_items:
            if len(item.strip()) > 0 and len(item.strip()) < 30: 
                signifiers.append({"name": item.strip(), "associations": [], "significance": "Extracted from analysis"})
        
        return {
            "signifiers": signifiers[:15], 
            "signifying_chains": chains,
            "analysis": text
        }
    
    def extract_unconscious_signifiers(self, dream_data):
        """Extract unconscious signifiers from dream data."""
        print("Extracting unconscious signifiers...")
        
        try:
            # Try to extract using VLM
            result = self.vlm_interface.generate_text("phase1", "extract_unconscious_signifiers", {"dreams": dream_data})
            
            # Parse the result
            unconscious_data = self.extract_json_from_response(result)
            
            # Check if we got valid data
            if not unconscious_data or not unconscious_data.get("signifiers"):
                raise ValueError("Invalid response from VLM")
        except Exception as e:
            print(f"Warning: Error extracting signifiers using VLM: {e}")
            print("Falling back to manual signifier extraction...")
            
            unconscious_data = self._extract_fallback_signifiers(dream_data)
        
        return unconscious_data

    # def _extract_fallback_signifiers(self, dream_data):
    #     """Extract signifiers from dreams when VLM fails."""
    #     dream_text = json.dumps(dream_data).lower()
        
    #     # Common signifiers often found in dreams
    #     signifiers = []
    #     relationships = []
        
    #     # Check for common themes in dreams
    #     theme_checks = [
    #         {"name": "John", "associations": ["friendship", "rivalry", "childhood"], 
    #         "significance": "A significant childhood friend representing peer relationships"},
    #         {"name": "Father/Dad", "associations": ["authority", "absence", "scary", "provider"], 
    #         "significance": "Represents paternal authority and its absence"},
    #         {"name": "Mother/Mom", "associations": ["comfort", "home", "protection", "absent"], 
    #         "significance": "Represents maternal care and attachment"},
    #         {"name": "House/Home", "associations": ["security", "identity", "belonging", "refuge"], 
    #         "significance": "Symbol of the self and personal identity"},
    #         {"name": "Losing Teeth", "associations": ["vulnerability", "anxiety", "growth", "change"], 
    #         "significance": "Common dream signifier representing anxiety about appearance or loss"}
    #     ]
        
    #     # Add all themes (simpler than checking for them)
    #     signifiers.extend(theme_checks)
        
    #     # Create signifying chains
    #     chains = [
    #         {
    #             "name": "Family Structure Chain",
    #             "signifiers": ["Mother/Mom", "Father/Dad", "House/Home"],
    #             "explanation": "Chain representing the family structure and domestic environment",
    #             "relation_to_fantasy": "Foundation of the subject's identity formation"
    #         },
    #         {
    #             "name": "Anxiety Displacement Chain",
    #             "signifiers": ["Losing Teeth", "John", "Father/Dad"],
    #             "explanation": "Chain representing displacement of anxiety through peer and authority figures",
    #             "relation_to_fantasy": "Manifestation of fears of inadequacy and judgment"
    #         }
    #     ]
        
    #     # Create object_a
    #     object_a = {
    #         "description": "The unattainable object of desire",
    #         "manifestations": ["approval", "belonging", "security"],
    #         "relation_to_desire": "Acts as the cause of desire, always out of reach"
    #     }
        
    #     # Create symptom
    #     symptom = {
    #         "description": "Anxiety manifesting as fear of loss",
    #         "signifiers_involved": ["Losing Teeth", "Father/Dad", "John"],
    #         "jouissance_pattern": "Repetition of anxiety scenarios involving loss and judgment"
    #     }
        
    #     # Create structural positions
    #     structural_positions = [
    #         {
    #             "position": "Hysteric's Discourse",
    #             "prominence": "Primary",
    #             "evidence": ["Questioning identity", "Seeking approval", "Anxiety about inadequacy"],
    #             "explanation": "The subject primarily speaks from a position of questioning their place"
    #         }
    #     ]
        
    #     return {
    #         "signifiers": signifiers,
    #         "signifying_chains": chains,
    #         "object_a": object_a,
    #         "symptom": symptom,
    #         "structural_positions": structural_positions,
    #         "analysis": "Analysis derived through manual extraction of common dream patterns."
    #     }
    
    def encode_experience(self, text):
        """Extract signifiers from text and add them to the graph."""
        doc = self.nlp(text)
        
        signifiers = []
        for token in doc:
            if token.pos_ in ["NOUN", "VERB", "ADJ"] and not token.is_stop:
                signifiers.append(token.text.lower())
        
        for signifier in signifiers:
            if signifier not in self.signifier_graph.graph:
                self.signifier_graph.add_node(
                    signifier,
                    type='symbolic',
                    activation=1.0,
                    repressed=False,
                    timestamp=datetime.now()
                )
            else:
                self.signifier_graph.graph.nodes[signifier]['activation'] = 1.0
                self.signifier_graph.graph.nodes[signifier]['timestamp'] = datetime.now()
        
        for i in range(len(signifiers) - 1):
            source = signifiers[i]
            target = signifiers[i + 1]
            
            self.signifier_graph.add_edge(
                source, target,
                weight=0.5,
                type='neutral',
                context=[text]
            )
        
        return signifiers
    
    def build_signifier_graph(self, signifiers, signifying_chains):
        """Build a graph representation of unconscious signifiers and their relationships."""
        print("Building signifier graph...")
        
        # Add signifiers as nodes
        for signifier in signifiers:
            name = signifier.get('name', '')
            if not name:
                continue
                
            sig_type = 'symbolic'  # Default
            if any(word in name.lower() for word in ['image', 'picture', 'visual', 'appearance']):
                sig_type = 'imaginary'
            elif any(word in name.lower() for word in ['impossible', 'trauma', 'death', 'void']):
                sig_type = 'real'
                
            repressed = False
            if 'significance' in signifier:
                if any(word in signifier['significance'].lower() for word in 
                       ['repressed', 'hidden', 'denied', 'avoided', 'unconscious']):
                    repressed = True
            
            # Add to graph
            self.signifier_graph.add_node(
                name, 
                type=sig_type,
                activation=0.5,  # Initial activation
                repressed=repressed
            )
        
        # Add edges based on signifying chains
        for chain in signifying_chains:
            chain_signifiers = chain.get('signifiers', [])
            chain_name = chain.get('name', '')
            
            # Create edges between consecutive signifiers in the chain
            for i in range(len(chain_signifiers) - 1):
                source = chain_signifiers[i]
                target = chain_signifiers[i + 1]
                
                # Determine edge type based on chain description
                edge_type = 'neutral'
                if chain_name and 'condensation' in chain_name.lower():
                    edge_type = 'condensation'
                elif chain_name and 'displacement' in chain_name.lower():
                    edge_type = 'displacement'
                
                # Add context from chain explanation
                context = [chain.get('explanation', '')]
                
                # Add to graph if both nodes exist
                if source in self.signifier_graph.graph and target in self.signifier_graph.graph:
                    self.signifier_graph.add_edge(
                        source, target,
                        weight=0.7,  # Strong association within a chain
                        type=edge_type,
                        context=context
                    )
        
        # Apply condensation to find similar signifiers
        self.signifier_graph.condense_nodes()
        
        # Apply displacement to create indirect associations
        for source in list(self.signifier_graph.graph.nodes()):
            for target in list(self.signifier_graph.graph.nodes()):
                if source != target:
                    self.signifier_graph.displace_association(source, target)
        
        print(f"Built signifier graph with {len(self.signifier_graph.graph.nodes())} nodes and {len(self.signifier_graph.graph.edges())} edges")
        return self.signifier_graph
    
    def build_unconscious_memory(self, dream_data, agent_name, agent_dir):
        """Build the unconscious memory from dreams data."""
        # Extract unconscious signifiers
        unconscious_data = self.extract_unconscious_signifiers(dream_data)
        
        # Build graph from extracted signifiers
        self.build_signifier_graph(
            unconscious_data.get('signifiers', []),
            unconscious_data.get('signifying_chains', [])
        )
        
        # Convert graph to serializable format for storage
        graph_data = self._serialize_graph_data()
        
        # Generate visualizations for signifiers
        visualization_results = self.visualize_signifiers(unconscious_data, agent_name, agent_dir)
        
        # Create final unconscious memory structure
        unconscious_memory = {
            'signifiers': unconscious_data.get('signifiers', []),
            'signifying_chains': unconscious_data.get('signifying_chains', []),
            'structural_positions': unconscious_data.get('structural_positions', []),
            'analysis': unconscious_data.get('analysis', ''),
            'object_a': unconscious_data.get('object_a', {}),
            'symptom': unconscious_data.get('symptom', {}),
            'signifier_graph': graph_data,
            'visualizations': visualization_results
        }
        
        # Save unconscious memory to file
        unconscious_path = f"{agent_dir}/unconscious_memory.json"
        save_json(unconscious_memory, unconscious_path)
        print(f"Unconscious memory saved to {unconscious_path}")
        
        return unconscious_memory
    
    def visualize_signifiers(self, unconscious_data, agent_name, agent_dir):
        """Generate visualizations for unconscious signifiers."""
        print("Generating visualizations for signifiers...")
        signifier_images_dir = f"{agent_dir}/signifier_images"
        ensure_directory(signifier_images_dir)
        
        # Extract signifier names
        signifiers = unconscious_data.get("signifiers", [])
        
        # Get signifier names, handling both string and dict formats
        signifier_names = []
        for signifier in signifiers:
            if isinstance(signifier, dict) and "name" in signifier:
                signifier_names.append(signifier["name"])
            elif isinstance(signifier, str):
                signifier_names.append(signifier)
        
        # Limit to 5 signifiers to avoid API overload
        signifier_names = signifier_names[:5]
        
        if not signifier_names:
            print("Warning: No signifiers found to visualize.")
            return {}
        
        print(f"Found {len(signifier_names)} signifiers to visualize: {', '.join(signifier_names)}")
        
        visualization_results = {}
        
        for signifier_name in signifier_names:
            print(f"Visualizing signifier: {signifier_name}")
            
            # Create a safe filename by replacing problematic characters
            safe_name = re.sub(r'[^\w\s-]', '', signifier_name).strip().replace(' ', '_')
            output_path = f"{signifier_images_dir}/{safe_name}"
            
            # Get signifier details for a better prompt
            signifier_details = None
            for signifier in signifiers:
                if isinstance(signifier, dict) and signifier.get("name") == signifier_name:
                    signifier_details = signifier
                    break
            
            # Create a more detailed prompt if we have signifier details
            if signifier_details and "associations" in signifier_details:
                associations = ", ".join(signifier_details["associations"])
                significance = signifier_details.get("significance", "")
                
                prompt = (
                    f"Create a surrealist image representing the unconscious signifier '{signifier_name}' "
                    f"with associations to {associations}. {significance} "
                    f"The image should be dreamlike and symbolic in the style of Salvador Dali or René Magritte."
                )
            else:
                prompt = (
                    f"Create a surrealist image representing the unconscious signifier '{signifier_name}' "
                    f"in the style of Salvador Dali or René Magritte. The image should be dreamlike and symbolic."
                )
            
            # Try direct generation first
            result = self.vlm_interface.direct_image_generation(prompt, output_path)
            
            # If direct generation fails, try the template approach
            if not result["success"]:
                print(f"Direct generation failed for {signifier_name}, trying template approach...")
                result = self.vlm_interface.generate_image(
                    "phase1", 
                    "visualize_signifier", 
                    {"signifier": signifier_name}, 
                    output_path
                )
            
            if result["success"]:
                visualization_results[signifier_name] = {
                    "image_path": result["image_path"],
                    "response_text": result.get("response_text", "")
                }
                print(f"Successfully created visualization for {signifier_name}: {result['image_path']}")
                
                # Add a small delay to avoid rate limiting
                time.sleep(2)
            else:
                print(f"Failed to visualize signifier {signifier_name}: {result.get('error', 'Unknown error')}")
        
        return visualization_results