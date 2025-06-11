import json
import re
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from interfaces.vlm_interface import VLMInterface
from utils.file_utils import ensure_directory, save_json
from utils.lacanian_graph import LacanianSignifierGraph

class UnconsciousBuilder:
    """
    Builds unconscious structures from dreams using Lacanian psychoanalytic principles.
    
    ALL unconscious processing uses the VLM interface for consistency.
    """

    def __init__(self, vlm_interface: VLMInterface = None):
        self.vlm = vlm_interface or VLMInterface()
        self.master_signifiers = []
        self.knowledge_signifiers = []
        self.object_a_manifestations = []

    def extract_unconscious_signifiers(self, dream_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract unconscious signifiers using Lacanian framework with VLM."""
        print("Extracting unconscious signifiers using VLM...")

        # Extract raw signifiers using VLM
        result = self.vlm.generate_text("phase1", "extract_unconscious_signifiers", {"dreams": dream_data})
        unconscious_data = self._parse_vlm_response(result)

        # Classify signifiers as S1 vs S2 using VLM
        unconscious_data = self._differentiate_signifier_types(unconscious_data, dream_data)

        # Map object a dynamics using VLM
        unconscious_data = self._map_object_a_dynamics(unconscious_data, dream_data)

        # Analyze structural positions using VLM
        unconscious_data = self._analyze_structural_positions(unconscious_data, dream_data)

        # Analyze dream-work patterns using VLM
        unconscious_data = self._analyze_dream_work(unconscious_data, dream_data)

        # Map jouissance economy using VLM
        unconscious_data = self._map_jouissance_economy(unconscious_data, dream_data)

        # Create fantasy formula using VLM
        unconscious_data = self._create_fantasy_formula(unconscious_data)

        return unconscious_data

    def _differentiate_signifier_types(self, data: Dict[str, Any], dream_data: Dict[str, Any]) -> Dict[str, Any]:
        """Classify signifiers as S1 (master) vs S2 (knowledge) using VLM."""
        template_data = {
            "signifiers": json.dumps(data.get('signifiers', []), indent=2),
            "dreams": json.dumps(dream_data, indent=2)[:2000] + "..."
        }

        print("Differentiating signifier types using VLM...")
        classification = self.vlm.generate_text("phase1", "differentiate_signifiers", template_data)
        parsed = self._parse_vlm_response(classification)

        # Enhance original data with classification
        data['signifier_classification'] = parsed
        return data

    def _map_object_a_dynamics(self, data: Dict[str, Any], dream_data: Dict[str, Any]) -> Dict[str, Any]:
        """Map object a as cause of desire using VLM."""
        template_data = {
            "dreams": json.dumps(dream_data, indent=2)[:2000] + "...",
            "signifiers": json.dumps(data.get('signifiers', []), indent=2)
        }

        print("Mapping object a dynamics using VLM...")
        object_a_analysis = self.vlm.generate_text("phase1", "map_object_a", template_data)
        parsed = self._parse_vlm_response(object_a_analysis)

        # Integrate object a analysis
        if 'object_a' in data:
            data['object_a']['structural_analysis'] = parsed
            data['object_a']['void_manifestations'] = parsed.get('void_manifestations', [])
            data['object_a']['circuit_of_desire'] = parsed.get('desire_circuit', {})

        return data

    def _analyze_structural_positions(self, data: Dict[str, Any], dream_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze subject's position in four discourses using VLM."""
        template_data = {
            "dreams": json.dumps(dream_data, indent=2)[:2000] + "..."
        }

        print("Analyzing discourse positions using VLM...")
        discourse_analysis = self.vlm.generate_text("phase1", "analyze_discourse_positions", template_data)
        parsed = self._parse_vlm_response(discourse_analysis)

        # Calculate discourse weights
        if 'discourse_percentages' in parsed:
            total = sum(parsed['discourse_percentages'].values())
            if total > 0:
                for discourse, value in parsed['discourse_percentages'].items():
                    if discourse in data.get('structural_positions', {}):
                        data['structural_positions'][discourse] = value / total
        
        return data

    def _analyze_dream_work(self, data: Dict[str, Any], dream_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze dream-work mechanisms using VLM."""
        template_data = {
            "signifiers": json.dumps(data.get('signifiers', []), indent=2),
            "signifying_chains": json.dumps(data.get('signifying_chains', []), indent=2)
        }

        print("Analyzing dream-work patterns using VLM...")
        dream_work_analysis = self.vlm.generate_text("phase1", "analyze_dream_work", template_data)
        parsed = self._parse_vlm_response(dream_work_analysis)

        data['dream_work_patterns'] = parsed
        return data

    def _map_jouissance_economy(self, data: Dict[str, Any], dream_data: Dict[str, Any]) -> Dict[str, Any]:
        """Map jouissance economy using VLM."""
        template_data = {
            "dreams": json.dumps(dream_data, indent=2)[:2000] + "...",
            "symptom": json.dumps(data.get('symptom', {}), indent=2)
        }

        print("Mapping jouissance economy using VLM...")
        jouissance_analysis = self.vlm.generate_text("phase1", "map_jouissance_economy", template_data)
        parsed = self._parse_vlm_response(jouissance_analysis)

        data['jouissance_economy'] = parsed
        return data

    def _create_fantasy_formula(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create fundamental fantasy formula using VLM."""
        template_data = {
            "signifiers": json.dumps(data.get('signifiers', []), indent=2),
            "object_a": json.dumps(data.get('object_a', {}), indent=2),
            "chain_names": [chain.get('name', 'unnamed') for chain in data.get('signifying_chains', [])]
        }

        print("Creating fantasy formula using VLM...")
        fantasy_analysis = self.vlm.generate_text("phase1", "create_fantasy_formula", template_data)
        parsed = self._parse_vlm_response(fantasy_analysis)

        data['fantasy_formula'] = parsed
        return data

    def build_signifier_graph(self, signifiers: List[Dict], chains: List[Dict]) -> LacanianSignifierGraph:
        """Build Lacanian signifier graph with S1/S2 dynamics."""
        graph = LacanianSignifierGraph()

        # Add all signifiers
        for sig in signifiers:
            if isinstance(sig, dict) and 'name' in sig:
                sig_name = sig['name']
                
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
        
        # Build signifying chains
        for chain in chains:
            if isinstance(chain, dict) and 'signifiers' in chain:
                chain_signifiers = chain['signifiers']
                
                # Ensure all signifiers exist in graph
                for sig_name in chain_signifiers:
                    if sig_name not in graph.graph:
                        graph.add_knowledge_signifier(sig_name, [], [])
                
                # Create the chain with retroactive meaning
                graph.create_signifying_chain(
                    chain.get('name', 'unnamed_chain'),
                    chain_signifiers,
                    chain_type=self._determine_chain_type(chain),
                    retroactive_meaning=True
                )
        
        return graph

    def _is_master_signifier(self, signifier: Dict) -> bool:
        """Determine if signifier functions as S1 (master signifier)."""
        indicators = [
            'identity' in signifier.get('significance', '').lower(),
            'recurring' in signifier.get('significance', '').lower(),
            'central' in signifier.get('significance', '').lower(),
            len(signifier.get('associations', [])) > 5,
            signifier.get('repressed', False)
        ]
        return sum(indicators) >= 2

    def _determine_chain_type(self, chain: Dict) -> str:
        """Determine type of signifying chain."""
        explanation = chain.get('explanation', '').lower()
        if 'metaphor' in explanation or 'substitut' in explanation:
            return 'metaphoric'
        elif 'metonym' in explanation or 'displace' in explanation:
            return 'metonymic'
        else:
            return 'mixed'

    def _parse_vlm_response(self, response: str) -> Dict[str, Any]:
        """Parse VLM responses, extracting JSON from text."""
        try:
            # Try to extract JSON from code blocks
            json_patterns = [
                r'```json\s*(.*?)\s*```',
                r'```\s*(.*?)\s*```',
            ]
            
            for pattern in json_patterns:
                matches = re.findall(pattern, response, re.DOTALL)
                if matches:
                    for match in reversed(matches):
                        try:
                            return json.loads(match)
                        except json.JSONDecodeError:
                            continue
            
            # Try parsing entire response as JSON
            return json.loads(response)

        except Exception as e:
            print(f"Could not parse VLM response as JSON: {e}")
            raise e  # Don't fallback, let it fail properly

    def visualize_signifiers(self, unconscious_data: Dict, agent_name: str, agent_dir: str) -> Dict[str, Any]:
        """Generate surrealist visualizations for key signifiers using VLM."""
        print("Generating psychoanalytic visualizations using VLM...")
        signifier_images_dir = f"{agent_dir}/signifier_images"
        ensure_directory(signifier_images_dir)
        
        visualization_results = {}
        
        # Select key signifiers for visualization
        key_signifiers = self._select_key_signifiers(unconscious_data)
        
        for signifier in key_signifiers[:5]:  # Limit to 5 for resource management
            safe_name = re.sub(r'[^\w\s-]', '', signifier['name']).strip().replace(' ', '_')
            output_path = f"{signifier_images_dir}/{safe_name}"
            
            # Create psychoanalytic visualization prompt using VLM prompt template
            template_data = {
                "signifier_name": signifier['name'],
                "significance": signifier.get('significance', 'Unknown significance'),
                "associations": ', '.join(signifier.get('associations', [])),
                "unconscious_context_json": json.dumps(unconscious_data, indent=2)[:500] + "..."
            }
            
            print(f"Visualizing '{signifier['name']}' using VLM...")
            
            # Use VLM template for generating visualization prompts
            viz_prompt = self.vlm.generate_text("phase1", "generate_visualization_prompt", template_data)
            
            # Generate the actual image
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
        """Select most psychoanalytically significant signifiers."""
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
        
        # Sort by score and return signifier dictionaries
        priority_signifiers.sort(key=lambda x: x[0], reverse=True)
        return [sig for _, sig in priority_signifiers]

    def build_unconscious_memory(self, dream_data: Dict, agent_name: str, agent_dir: str) -> Dict[str, Any]:
        """
        Build complete unconscious memory structure using VLM for all processing.
        """
        print(f"Building Lacanian unconscious structure for {agent_name} using VLM...")
        
        # Extract and analyze unconscious signifiers
        unconscious_data = self.extract_unconscious_signifiers(dream_data)
        
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
            'dream_count': len(dream_data.get('dreams', [])),
            'processing_method': 'VLM_unified'
        }
        
        # Save unconscious structure
        unconscious_path = f"{agent_dir}/unconscious_memory.json"
        save_json(unconscious_data, unconscious_path)
        print(f"Unconscious memory saved to {unconscious_path}")
        
        return unconscious_data