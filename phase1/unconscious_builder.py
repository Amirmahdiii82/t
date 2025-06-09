import json
import re
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

from interfaces.vlm_interface import VLMInterface
from utils.file_utils import ensure_directory, save_json
from utils.prompt_utils import render_prompt

# Forward declaration for type hinting to avoid circular import issues at runtime
LacanianSignifierGraph = Any

class UnconsciousBuilder:
    """Builds unconscious structure from dreams using Lacanian psychoanalytic principles."""

    def __init__(self, vlm_interface: VLMInterface):
        """
        Initializes the UnconsciousBuilder with a VLM interface and Lacanian components.
        
        Args:
            vlm_interface (VLMInterface): The interface for interacting with the Vision-Language Model.
        """
        self.vlm = vlm_interface
        # Initialize Lacanian structural components
        self.master_signifiers = []  # S1 - anchoring points
        self.knowledge_signifiers = []  # S2 - chain of knowledge
        self.object_a_manifestations = []  # Cause of desire
        self.jouissance_patterns = []  # Patterns of painful enjoyment
        self.symbolic_positions = {
            "master": 0.0,
            "university": 0.0,
            "hysteric": 0.0,
            "analyst": 0.0
        }

    def extract_unconscious_signifiers(self, dream_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extracts unconscious signifiers using a multi-pass VLM analysis with a Lacanian framework.
        
        Args:
            dream_data (Dict[str, Any]): The raw dream data for an agent.
            
        Returns:
            Dict[str, Any]: A dictionary containing the structured unconscious data.
        """
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
        """
        Differentiates between S1 (master signifiers) and S2 (knowledge signifiers) using the VLM.
        """
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
        """
        Maps object a as the cause of desire and identifies jouissance patterns using the VLM.
        """
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
        """
        Analyzes the subject's position in the four discourses using the VLM.
        """
        template_data = {
            "dreams": json.dumps(dream_data, indent=2)[:2000] + "..."
        }

        discourse_analysis = self.vlm.generate_text("phase1", "analyze_discourse_positions", template_data)
        parsed = self._parse_vlm_response(discourse_analysis)

        # Calculate discourse weights
        if 'discourse_percentages' in parsed:
            total = sum(parsed['discourse_percentages'].values())
            if total > 0:
                for discourse, value in parsed['discourse_percentages'].items():
                    if discourse in data['structural_positions']:
                        data['structural_positions'][discourse] = value / total
        
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
                for sig_name in chain_signifiers:
                    if sig_name not in graph.graph:
                        # Add missing signifier as a knowledge signifier by default
                        graph.add_knowledge_signifier(sig_name, [], [])
                
                # Create the chain
                graph.create_signifying_chain(
                    chain.get('name', 'unnamed_chain'),
                    chain_signifiers,
                    chain_type=self._determine_chain_type(chain),
                    retroactive_meaning=True  # Nachträglichkeit
                )
        
        return graph

    def _is_master_signifier(self, signifier: Dict) -> bool:
        """
        Determines if a signifier functions as S1 (master signifier) based on heuristics.
        """
        # Master signifiers are often repeated, points of identification, or organizing principles.
        indicators = [
            'identity' in signifier.get('significance', '').lower(),
            'recurring' in signifier.get('significance', '').lower(),
            'central' in signifier.get('significance', '').lower(),
            len(signifier.get('associations', [])) > 5,  # Many connections can indicate centrality
            signifier.get('repressed', False)  # Primal repression is a key indicator
        ]
        return sum(indicators) >= 2

    def _determine_chain_type(self, chain: Dict) -> str:
        """Determines the type of signifying chain (metaphoric or metonymic)."""
        explanation = chain.get('explanation', '').lower()
        if 'metaphor' in explanation or 'substitut' in explanation:
            return 'metaphoric'
        elif 'metonym' in explanation or 'displace' in explanation:
            return 'metonymic'
        else:
            return 'mixed'

    def _parse_vlm_response(self, response: str) -> Dict[str, Any]:
        """
        Enhanced parsing for VLM responses, robustly extracting JSON from text.
        """
        try:
            # Try to extract JSON with multiple patterns to handle variations in VLM output
            json_patterns = [
                r'```json\s*(.*?)\s*```',
                r'```\s*(.*?)\s*```',
            ]
            
            for pattern in json_patterns:
                matches = re.findall(pattern, response, re.DOTALL)
                if matches:
                    # Prefer the last, most complete match
                    for match in reversed(matches):
                        try:
                            return json.loads(match)
                        except json.JSONDecodeError:
                            continue
            
            # If no code block JSON found, try to parse the entire response as JSON
            return json.loads(response)

        except Exception:
            # If all parsing fails, return the raw analysis to avoid losing data
            print(f"Could not parse VLM response as JSON, returning raw text.")
            return {"raw_analysis": response}

    def visualize_signifiers(self, unconscious_data: Dict, agent_name: str, agent_dir: str) -> Dict[str, Any]:
        """
        Generates surrealist visualizations for key signifiers to capture unconscious dynamics.
        
        Args:
            unconscious_data (Dict): The agent's full unconscious structure.
            agent_name (str): The name of the agent.
            agent_dir (str): The directory to save the images.

        Returns:
            Dict[str, Any]: A dictionary of the visualization results.
        """
        print("Generating psychoanalytic visualizations...")
        signifier_images_dir = f"{agent_dir}/signifier_images"
        ensure_directory(signifier_images_dir)
        
        visualization_results = {}
        
        # Select key signifiers for visualization
        key_signifiers = self._select_key_signifiers(unconscious_data)
        
        for signifier in key_signifiers[:5]:  # Limit to 5 most significant to manage resources
            safe_name = re.sub(r'[^\w\s-]', '', signifier['name']).strip().replace(' ', '_')
            output_path = f"{signifier_images_dir}/{safe_name}"
            
            # Create a psychoanalytically informed prompt for the VLM's image generator
            viz_prompt = self._create_visualization_prompt(signifier, unconscious_data)
            
            print(f"  - Visualizing '{signifier['name']}'...")
            result = self.vlm.direct_image_generation(viz_prompt, output_path)
            
            if result.get("success"):
                visualization_results[signifier['name']] = {
                    "image_path": result["image_path"],
                    "signifier_data": signifier,
                    "visualization_concept": viz_prompt
                }
                time.sleep(2)  # Add a small delay to avoid overwhelming the API
        
        return visualization_results

    def _select_key_signifiers(self, unconscious_data: Dict) -> List[Dict]:
        """
        Selects the most psychoanalytically significant signifiers for visualization based on a scoring system.
        """
        signifiers = unconscious_data.get('signifiers', [])
        priority_signifiers = []

        for sig in signifiers:
            if not isinstance(sig, dict) or 'name' not in sig:
                continue

            score = 0
            # Master signifier indicators
            if 'master' in str(sig.get('significance', '')).lower() or self._is_master_signifier(sig):
                score += 3
            # Object a relation
            if any(term in str(sig).lower() for term in ['desire', 'lack', 'void', 'impossible']):
                score += 2
            # Repression indicator
            if sig.get('repressed', False):
                score += 2
            # Chain participation
            chain_count = sum(1 for chain in unconscious_data.get('signifying_chains', []) 
                            if sig['name'] in chain.get('signifiers', []))
            score += chain_count
            
            priority_signifiers.append((score, sig))
        
        # Sort by score (descending) and return the signifier dictionaries
        priority_signifiers.sort(key=lambda x: x[0], reverse=True)
        return [sig for _, sig in priority_signifiers]

    def _create_visualization_prompt(self, signifier: Dict, unconscious_data: Dict) -> str:
        """
        Creates a rich, psychoanalytically-informed visualization prompt for the VLM.
        """
        object_a_relation = ""
        if 'object_a' in unconscious_data and signifier['name'] in str(unconscious_data['object_a']):
            object_a_relation = "This signifier circles the void of object a, the impossible object-cause of desire."
        
        template_data = {
            "signifier_name": signifier['name'],
            "significance": signifier.get('significance', 'An element of the unconscious tapestry.'),
            "associations": ', '.join(signifier.get('associations', ['dreams', 'memories'])),
            "object_a_relation": object_a_relation
        }
        
        return render_prompt("phase1", "visualize_signifier", template_data)

    def build_unconscious_memory(self, dream_data: Dict, agent_name: str, agent_dir: str) -> Dict[str, Any]:
        """
        Builds the complete unconscious memory structure for an agent.
        
        This orchestrates the full pipeline: signifier extraction, graph building,
        visualization, and structuring of psychoanalytic concepts.

        Args:
            dream_data (Dict): The raw dream data.
            agent_name (str): The name of the agent being built.
            agent_dir (str): The base directory for saving agent files.

        Returns:
            Dict[str, Any]: The complete, structured unconscious memory.
        """
        print(f"Building Lacanian unconscious structure for {agent_name}...")
        
        # 1. Extract and analyze unconscious signifiers from dreams
        unconscious_data = self.extract_unconscious_signifiers(dream_data)
        
        # 2. Build the signifier graph from the extracted data
        graph = self.build_signifier_graph(
            unconscious_data.get('signifiers', []),
            unconscious_data.get('signifying_chains', [])
        )
        unconscious_data['signifier_graph'] = graph.serialize()
        
        # 3. Generate surrealist visualizations for key signifiers
        visualizations = self.visualize_signifiers(unconscious_data, agent_name, agent_dir)
        unconscious_data['visualizations'] = visualizations
        
        # 4. Add final metadata
        unconscious_data['metadata'] = {
            'agent_name': agent_name,
            'extraction_date': datetime.now().isoformat(),
            'theoretical_framework': 'Lacanian Psychoanalysis',
            'dream_count': len(dream_data.get('dreams', []))
        }
        
        # 5. Save the complete unconscious structure to a file
        unconscious_path = f"{agent_dir}/unconscious_memory.json"
        save_json(unconscious_data, unconscious_path)
        print(f"Unconscious memory for {agent_name} saved to {unconscious_path}")
        
        return unconscious_data