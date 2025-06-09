def build_signifier_graph(self, signifiers: List[Dict], chains: List[Dict]) -> 'LacanianSignifierGraph':
        """Build a proper Lacanian signifier graph with S1/S2 dynamics."""
        from utils.lacanian_graph import LacanianSignifierGraph
        
        graph = LacanianSignifierGraph()
        
        # Add all signifiers first
        for sig in signifiers:
            if isinstance(sig, dict) and 'name'
            
import json
import re
import time
from datetime import datetime
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from interfaces.vlm_interface import VLMInterface
from utils.file_utils import ensure_directory, save_json
from utils.prompt_utils import render_prompt

class UnconsciousBuilder:
    """Builds unconscious structure from dreams using Lacanian psychoanalytic principles."""
    
    def __init__(self, vlm_interface: VLMInterface):
        self.vlm = vlm_interface
        # Initialize Lacanian structural components
        self.master_signifiers = []  # S1 - anchoring points
        self.knowledge_signifiers = []  # S2 - chain of knowledge
        self.object_a_manifestations = []  # Cause of desire``
        self.jouissance_patterns = []  # Patterns of painful enjoyment
        self.symbolic_positions = {
            "master": 0.0,
            "university": 0.0,
            "hysteric": 0.0,
            "analyst": 0.0
        }
        
    def extract_unconscious_signifiers(self, dream_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract unconscious signifiers using pure VLM analysis with Lacanian framework."""
        print("Extracting unconscious signifiers through Lacanian analysis...")
        
        # First pass: Extract raw signifiers
        result = self.vlm.generate_text("phase1", "extract_unconscious_signifiers", {"dreams": dream_data})
        unconscious_data = self._parse_vlm_response(result)
        
        # Second pass: Identify master signifiers (S1) vs knowledge signifiers (S2)
        unconscious_data = self._differentiate_signifier_types(unconscious_data, dream_data)
        
        # Third pass: Map object a and jouissance patterns
        unconscious_data = self._map_object_a_dynamics(unconscious_data, dream_data)
        
        # Fourth pass: Determine structural positions
        unconscious_data = self._analyze_structural_positions(unconscious_data, dream_data)
        
        return unconscious_data
    
    def _differentiate_signifier_types(self, data: Dict[str, Any], dream_data: Dict[str, Any]) -> Dict[str, Any]:
        """Differentiate between S1 (master signifiers) and S2 (knowledge signifiers)."""
        template_data = {
            "signifiers": json.dumps(data.get('signifiers', []), indent=2),
            "dreams": json.dumps(dream_data, indent=2)[:2000] + "..."
        }
        
        classification = self.vlm.generate_text("phase1", "differentiate_signifiers", template_data)
        parsed = self._parse_vlm_response(classification)
        
        # Enhance original data with classification
        data['signifier_classification'] = parsed
        return data
    
    def _map_object_a_dynamics(self, data: Dict[str, Any], dream_data: Dict[str, Any]) -> Dict[str, Any]:
        """Map object a as cause of desire and identify jouissance patterns."""
        template_data = {
            "dreams": json.dumps(dream_data, indent=2)[:2000] + "...",
            "signifiers": json.dumps(data.get('signifiers', []), indent=2)
        }
        
        object_a_analysis = self.vlm.generate_text("phase1", "map_object_a", template_data)
        parsed = self._parse_vlm_response(object_a_analysis)
        
        # Deep integration of object a
        if 'object_a' in data:
            data['object_a']['structural_analysis'] = parsed
            data['object_a']['void_manifestations'] = parsed.get('void_manifestations', [])
            data['object_a']['circuit_of_desire'] = parsed.get('desire_circuit', {})
        
        return data
    
    def _analyze_structural_positions(self, data: Dict[str, Any], dream_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the subject's position in the four discourses."""
        template_data = {
            "dreams": json.dumps(dream_data, indent=2)[:2000] + "..."
        }
        
        discourse_analysis = self.vlm.generate_text("phase1", "analyze_discourse_positions", template_data)
        parsed = self._parse_vlm_response(discourse_analysis)
        
        # Calculate discourse weights
        if 'discourse_percentages' in parsed:
            total = sum(parsed['discourse_percentages'].values())
            if total > 0:
                for discourse in parsed['discourse_percentages']:
                    data['structural_positions'][discourse] = parsed['discourse_percentages'][discourse] / total
        
        return data
    
    def build_signifier_graph(self, signifiers: List[Dict], chains: List[Dict]) -> 'LacanianSignifierGraph':
        """Build a proper Lacanian signifier graph with S1/S2 dynamics."""
        from utils.lacanian_graph import LacanianSignifierGraph
        
        graph = LacanianSignifierGraph()
        
        # Add all signifiers first
        for sig in signifiers:
            if isinstance(sig, dict) and 'name' in sig:
                sig_name = sig['name']
                
                # Determine if it's a master signifier
                if self._is_master_signifier(sig):
                    graph.add_master_signifier(
                        sig_name,
                        anchoring_function=sig.get('significance', ''),
                        primal_repression=sig.get('repressed', False)
                    )
                else:
                    graph.add_knowledge_signifier(
                        sig_name,
                        sig.get('associations', []),
                        metaphoric_substitutions=sig.get('substitutions', [])
                    )
        
        # Build signifying chains with proper Lacanian logic
        for chain in chains:
            if isinstance(chain, dict) and 'signifiers' in chain:
                chain_signifiers = chain['signifiers']
                
                # Ensure all signifiers in chain exist in graph
                for sig in chain_signifiers:
                    if sig not in graph.graph:
                        # Add missing signifier
                        graph.add_knowledge_signifier(sig, [], [])
                
                # Create the chain
                graph.create_signifying_chain(
                    chain.get('name', 'unnamed_chain'),
                    chain_signifiers,
                    chain_type=self._determine_chain_type(chain),
                    retroactive_meaning=True  # Nachträglichkeit
                )
        
        return graph
    
    def _is_master_signifier(self, signifier: Dict) -> bool:
        """Determine if a signifier functions as S1 (master signifier)."""
        # Master signifiers are often:
        # - Repeated without clear meaning
        # - Points of identification
        # - Organizing principles
        indicators = [
            'identity' in signifier.get('significance', '').lower(),
            'recurring' in signifier.get('significance', '').lower(),
            'central' in signifier.get('significance', '').lower(),
            len(signifier.get('associations', [])) > 5,  # Many connections
            signifier.get('repressed', False)  # Often repressed
        ]
        return sum(indicators) >= 2
    
    def _determine_chain_type(self, chain: Dict) -> str:
        """Determine the type of signifying chain."""
        explanation = chain.get('explanation', '').lower()
        if 'metaphor' in explanation or 'substitut' in explanation:
            return 'metaphoric'
        elif 'metonym' in explanation or 'displace' in explanation:
            return 'metonymic'
        else:
            return 'mixed'
    
    def generate_dream_work_patterns(self, unconscious_data: Dict) -> Dict[str, Any]:
        """Generate patterns of dream-work (condensation, displacement, etc.)."""
        template_data = {
            "signifiers": json.dumps(unconscious_data.get('signifiers', []), indent=2),
            "signifying_chains": json.dumps(unconscious_data.get('signifying_chains', []), indent=2)
        }
        
        dream_work = self.vlm.generate_text("phase1", "analyze_dream_work", template_data)
        return self._parse_vlm_response(dream_work)
    
    def map_jouissance_economy(self, unconscious_data: Dict, dream_data: Dict) -> Dict[str, Any]:
        """Map the economy of jouissance in the unconscious structure."""
        template_data = {
            "dreams": json.dumps(dream_data, indent=2)[:1500] + "...",
            "symptom": json.dumps(unconscious_data.get('symptom', {}), indent=2)
        }
        
        jouissance_map = self.vlm.generate_text("phase1", "map_jouissance_economy", template_data)
        return self._parse_vlm_response(jouissance_map)
    
    def create_fantasy_formula(self, unconscious_data: Dict) -> str:
        """Create the subject's fundamental fantasy formula ($ ◊ a)."""
        signifiers = unconscious_data.get('signifiers', [])
        object_a = unconscious_data.get('object_a', {})
        chains = unconscious_data.get('signifying_chains', [])
        
        template_data = {
            "signifiers": json.dumps(signifiers[:5], indent=2),
            "object_a": json.dumps(object_a, indent=2),
            "chain_names": json.dumps([c['name'] for c in chains], indent=2)
        }
        
        fantasy = self.vlm.generate_text("phase1", "create_fantasy_formula", template_data)
        return fantasy
    
    def _parse_vlm_response(self, response: str) -> Dict[str, Any]:
        """Enhanced parsing that preserves psychoanalytic complexity."""
        try:
            # Try to extract JSON with multiple patterns
            json_patterns = [
                r'```json\s*(.*?)\s*```',
                r'```\s*(.*?)\s*```',
                r'\{[^{}]*\{[^{}]*\}[^{}]*\}',  # Nested JSON
                r'\{[^{}]*\}'  # Simple JSON
            ]
            
            for pattern in json_patterns:
                matches = re.findall(pattern, response, re.DOTALL)
                for match in matches:
                    try:
                        return json.loads(match)
                    except:
                        continue
            
            # If no JSON found, structure the response
            return self._structure_text_response(response)
            
        except Exception as e:
            print(f"Parse error: {e}")
            return {"raw_analysis": response}
    
    def _structure_text_response(self, text: str) -> Dict[str, Any]:
        """Structure non-JSON text into usable format while preserving meaning."""
        lines = text.strip().split('\n')
        structured = {
            "signifiers": [],
            "analysis": "",
            "key_points": []
        }
        
        current_section = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if any(marker in line.lower() for marker in ['signifier:', 'symbol:', 'element:']):
                # Extract signifier
                name = line.split(':', 1)[1].strip() if ':' in line else line
                structured["signifiers"].append({"name": name, "extracted_from": "text"})
            elif line.startswith('-') or line.startswith('•'):
                structured["key_points"].append(line[1:].strip())
            else:
                structured["analysis"] += line + " "
        
        return structured
    
    def visualize_signifiers(self, unconscious_data: Dict, agent_name: str, agent_dir: str) -> Dict[str, Any]:
        """Generate surrealist visualizations that capture unconscious dynamics."""
        print("Generating psychoanalytic visualizations...")
        signifier_images_dir = f"{agent_dir}/signifier_images"
        ensure_directory(signifier_images_dir)
        
        visualization_results = {}
        
        # Select key signifiers for visualization
        key_signifiers = self._select_key_signifiers(unconscious_data)
        
        for signifier in key_signifiers[:5]:  # Limit to 5 most significant
            safe_name = re.sub(r'[^\w\s-]', '', signifier['name']).strip().replace(' ', '_')
            output_path = f"{signifier_images_dir}/{safe_name}"
            
            # Create psychoanalytically informed prompt
            viz_prompt = self._create_visualization_prompt(signifier, unconscious_data)
            
            result = self.vlm.direct_image_generation(viz_prompt, output_path)
            
            if result.get("success"):
                visualization_results[signifier['name']] = {
                    "image_path": result["image_path"],
                    "signifier_data": signifier,
                    "visualization_concept": viz_prompt
                }
                time.sleep(2)  # Rate limiting
        
        return visualization_results
    
    def _select_key_signifiers(self, unconscious_data: Dict) -> List[Dict]:
        """Select most psychoanalytically significant signifiers for visualization."""
        signifiers = unconscious_data.get('signifiers', [])
        
        # Prioritize master signifiers and those related to object a
        priority_signifiers = []
        for sig in signifiers:
            score = 0
            # Master signifier indicators
            if 'master' in str(sig.get('significance', '')).lower():
                score += 3
            # Object a relation
            if any(obj in str(sig).lower() for obj in ['desire', 'lack', 'void', 'impossible']):
                score += 2
            # Repression indicator
            if sig.get('repressed', False):
                score += 2
            # Chain participation
            chain_count = sum(1 for chain in unconscious_data.get('signifying_chains', []) 
                            if sig['name'] in chain.get('signifiers', []))
            score += chain_count
            
            priority_signifiers.append((score, sig))
        
        # Sort by score and return top signifiers
        priority_signifiers.sort(key=lambda x: x[0], reverse=True)
        return [sig for _, sig in priority_signifiers]
    
    def _create_visualization_prompt(self, signifier: Dict, unconscious_data: Dict) -> str:
        """Create visualization prompt that captures psychoanalytic dimensions."""
        # Find related chains and object a manifestations
        related_chains = [
            chain for chain in unconscious_data.get('signifying_chains', [])
            if signifier['name'] in chain.get('signifiers', [])
        ]
        
        object_a_relation = ""
        if 'object_a' in unconscious_data:
            if signifier['name'] in str(unconscious_data['object_a']):
                object_a_relation = "This signifier circles around the void of object a."
        
        # Prepare template data
        template_data = {
            "signifier_name": signifier['name'],
            "significance": signifier.get('significance', ''),
            "associations": ', '.join(signifier.get('associations', [])),
            "object_a_relation": object_a_relation
        }
        
        # Use the template
        return render_prompt("phase1", "visualize_signifier", template_data)
    
    def build_unconscious_memory(self, dream_data: Dict, agent_name: str, agent_dir: str) -> Dict[str, Any]:
        """Build complete unconscious memory with all psychoanalytic structures."""
        print(f"Building Lacanian unconscious structure for {agent_name}...")
        
        # Extract unconscious signifiers with full analysis
        unconscious_data = self.extract_unconscious_signifiers(dream_data)
        
        # Generate dream-work patterns
        dream_work = self.generate_dream_work_patterns(unconscious_data)
        unconscious_data['dream_work_patterns'] = dream_work
        
        # Map jouissance economy
        jouissance = self.map_jouissance_economy(unconscious_data, dream_data)
        unconscious_data['jouissance_economy'] = jouissance
        
        # Create fundamental fantasy formula
        fantasy = self.create_fantasy_formula(unconscious_data)
        unconscious_data['fundamental_fantasy'] = fantasy
        
        # Build signifier graph
        graph = self.build_signifier_graph(
            unconscious_data.get('signifiers', []),
            unconscious_data.get('signifying_chains', [])
        )
        unconscious_data['signifier_graph'] = graph.serialize()
        
        # Generate visualizations
        visualizations = self.visualize_signifiers(unconscious_data, agent_name, agent_dir)
        unconscious_data['visualizations'] = visualizations
        
        # Add metadata
        unconscious_data['metadata'] = {
            'agent_name': agent_name,
            'extraction_date': datetime.now().isoformat(),
            'theoretical_framework': 'Lacanian Psychoanalysis',
            'dream_count': len(dream_data.get('dreams', []))
        }
        
        # Save complete unconscious structure
        unconscious_path = f"{agent_dir}/unconscious_memory.json"
        save_json(unconscious_data, unconscious_path)
        print(f"Unconscious memory saved to {unconscious_path}")
        
        return unconscious_data