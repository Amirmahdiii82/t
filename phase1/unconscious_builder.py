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
    
    Extracts signifiers, builds signifying chains, identifies object a manifestations,
    and creates surrealist visualizations of unconscious content.
    """

    def __init__(self, vlm_interface: VLMInterface):
        self.vlm = vlm_interface
        self.master_signifiers = []
        self.knowledge_signifiers = []
        self.object_a_manifestations = []

    def extract_unconscious_signifiers(self, dream_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract unconscious signifiers using Lacanian framework.
        
        Args:
            dream_data: Raw dream narratives and content
            
        Returns:
            Dictionary containing structured unconscious data
        """
        print("Extracting unconscious signifiers...")

        try:
            # Extract raw signifiers using VLM
            result = self.vlm.generate_text("phase1", "extract_unconscious_signifiers", {"dreams": dream_data})
            if not result or result.strip() == "":
                print("Warning: Empty VLM response, using fallback extraction")
                return self._create_fallback_unconscious_structure(dream_data)
            
            unconscious_data = self._parse_vlm_response(result)
            
            # If parsing failed, use fallback
            if not unconscious_data or "raw_analysis" in unconscious_data:
                print("Warning: VLM response parsing failed, using fallback extraction")
                return self._create_fallback_unconscious_structure(dream_data)

            # Enhance with additional analysis
            unconscious_data = self._enhance_unconscious_data(unconscious_data, dream_data)

            return unconscious_data
            
        except Exception as e:
            print(f"Error in unconscious signifier extraction: {e}")
            return self._create_fallback_unconscious_structure(dream_data)

    def _create_fallback_unconscious_structure(self, dream_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a fallback unconscious structure when VLM fails."""
        print("Creating fallback unconscious structure...")
        
        # Extract basic signifiers from text analysis
        dreams_text = str(dream_data).lower()
        
        # Common psychoanalytic signifiers
        potential_signifiers = [
            "mother", "father", "family", "home", "love", "fear", "death", "birth",
            "water", "fire", "light", "dark", "child", "adult", "past", "future",
            "house", "door", "window", "road", "journey", "lost", "found"
        ]
        
        found_signifiers = []
        for sig in potential_signifiers:
            if sig in dreams_text:
                found_signifiers.append({
                    "name": sig,
                    "significance": f"Recurring element related to {sig}",
                    "associations": [sig + "_related", "emotional_" + sig],
                    "relation_to_desire": f"Circulates around the absence of {sig}",
                    "repressed": False
                })
        
        # Ensure we have at least some signifiers
        if not found_signifiers:
            found_signifiers = [
                {
                    "name": "identity",
                    "significance": "Core sense of self",
                    "associations": ["self", "other", "recognition"],
                    "relation_to_desire": "Fundamental question of being",
                    "repressed": False
                },
                {
                    "name": "relationship",
                    "significance": "Connection with others",
                    "associations": ["love", "fear", "trust"],
                    "relation_to_desire": "Desire for recognition",
                    "repressed": False
                }
            ]
        
        return {
            "signifiers": found_signifiers[:10],  # Limit to 10
            "signifying_chains": [
                {
                    "name": "identity_chain",
                    "signifiers": [s["name"] for s in found_signifiers[:3]],
                    "type": "mixed",
                    "explanation": "Chain of identity formation",
                    "relation_to_fantasy": "Supports fundamental fantasy of coherent self"
                }
            ],
            "object_a": {
                "description": "The void that causes desire",
                "manifestations": ["absence", "lack", "impossible_object"],
                "void_manifestations": ["what is missing", "unfulfillable desire"],
                "desire_circuit": "Desire moves around the central void without capturing it"
            },
            "symptom": {
                "description": "Repetitive pattern of enjoyment and suffering",
                "signifiers_involved": [found_signifiers[0]["name"] if found_signifiers else "unknown"],
                "jouissance_pattern": "Painful pleasure in repetition",
                "repetition_structure": "Circular return to same conflicts"
            },
            "structural_positions": {
                "hysteric": 0.4,
                "master": 0.3, 
                "university": 0.2,
                "analyst": 0.1
            },
            "fallback_mode": True
        }

    def _enhance_unconscious_data(self, data: Dict[str, Any], dream_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance unconscious data with additional analysis."""
        try:
            # Classify signifiers as S1 vs S2 if not already done
            if "signifier_classification" not in data:
                data = self._differentiate_signifier_types(data, dream_data)

            # Map object a dynamics if not detailed enough
            if not data.get("object_a", {}).get("void_manifestations"):
                data = self._map_object_a_dynamics(data, dream_data)

            # Analyze structural positions if not present
            if not data.get("structural_positions"):
                data = self._analyze_structural_positions(data, dream_data)

            return data
            
        except Exception as e:
            print(f"Warning: Could not enhance unconscious data: {e}")
            return data

    def _differentiate_signifier_types(self, data: Dict[str, Any], dream_data: Dict[str, Any]) -> Dict[str, Any]:
        """Classify signifiers as S1 (master) vs S2 (knowledge) using fallback logic."""
        signifiers = data.get('signifiers', [])
        
        master_signifiers = []
        knowledge_signifiers = []
        
        for sig in signifiers:
            if not isinstance(sig, dict) or 'name' not in sig:
                continue
                
            # Simple heuristic for S1 vs S2 classification
            significance = sig.get('significance', '').lower()
            if any(word in significance for word in ['identity', 'core', 'central', 'fundamental']):
                master_signifiers.append({
                    "name": sig['name'],
                    "anchoring_function": f"Organizes meaning around {sig['name']}"
                })
            else:
                knowledge_signifiers.append({
                    "name": sig['name'],
                    "chain_position": f"Links to other signifiers in meaning network"
                })
        
        data['signifier_classification'] = {
            "master_signifiers": master_signifiers,
            "knowledge_signifiers": knowledge_signifiers
        }
        
        return data

    def _map_object_a_dynamics(self, data: Dict[str, Any], dream_data: Dict[str, Any]) -> Dict[str, Any]:
        """Map object a as cause of desire using fallback logic."""
        object_a = data.get('object_a', {})
        
        # Enhance object a with fallback data
        object_a.update({
            'structural_analysis': {
                'void_manifestations': ["absence", "lack", "impossible_fulfillment"],
                'desire_circuit': "Desire circulates without capture",
                'jouissance_patterns': ["repetitive seeking", "painful pleasure"]
            },
            'void_manifestations': ["what cannot be obtained", "missing piece"],
            'circuit_of_desire': {
                'movement': 'circular',
                'object': 'always already lost',
                'satisfaction': 'impossible'
            }
        })
        
        data['object_a'] = object_a
        return data

    def _analyze_structural_positions(self, data: Dict[str, Any], dream_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze subject's position in four discourses using fallback logic."""
        # Default distribution based on typical neurotic structure
        data['structural_positions'] = {
            'hysteric': 0.4,  # Default to hysteric as most common neurotic position
            'master': 0.3,
            'university': 0.2,
            'analyst': 0.1
        }
        
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
        """Parse VLM responses, extracting JSON from text with improved error handling."""
        if not response or response.strip() == "":
            return {}
            
        try:
            # Try to extract JSON from code blocks first
            json_patterns = [
                r'```json\s*(.*?)\s*```',
                r'```\s*(.*?)\s*```',
            ]
            
            for pattern in json_patterns:
                matches = re.findall(pattern, response, re.DOTALL)
                if matches:
                    for match in reversed(matches):
                        try:
                            return json.loads(match.strip())
                        except json.JSONDecodeError:
                            continue
            
            # Try parsing entire response as JSON
            return json.loads(response.strip())

        except json.JSONDecodeError:
            # If JSON parsing fails, return indicator for fallback
            return {"raw_analysis": response}
        except Exception:
            return {}

    def visualize_signifiers(self, unconscious_data: Dict, agent_name: str, agent_dir: str) -> Dict[str, Any]:
        """Generate surrealist visualizations for key signifiers."""
        print("Generating psychoanalytic visualizations...")
        signifier_images_dir = f"{agent_dir}/signifier_images"
        ensure_directory(signifier_images_dir)
        
        visualization_results = {}
        
        # Select key signifiers for visualization
        key_signifiers = self._select_key_signifiers(unconscious_data)
        
        # Limit visualizations for production (time/resource constraints)
        max_visualizations = 3
        
        for i, signifier in enumerate(key_signifiers[:max_visualizations]):
            safe_name = re.sub(r'[^\w\s-]', '', signifier['name']).strip().replace(' ', '_')
            output_path = f"{signifier_images_dir}/{safe_name}"
            
            # Create psychoanalytic visualization prompt
            viz_prompt = self._create_visualization_prompt(signifier, unconscious_data)
            
            print(f"Visualizing '{signifier['name']}'...")
            try:
                result = self.vlm.direct_image_generation(viz_prompt, output_path)
                
                if result and result.get("success"):
                    visualization_results[signifier['name']] = {
                        "image_path": result["image_path"],
                        "signifier_data": signifier,
                        "visualization_concept": viz_prompt
                    }
                    time.sleep(1)  # Rate limiting
                else:
                    print(f"Warning: Failed to generate image for {signifier['name']}")
                    
            except Exception as e:
                print(f"Warning: Error generating visualization for {signifier['name']}: {e}")
                continue
        
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

    def _create_visualization_prompt(self, signifier: Dict, unconscious_data: Dict) -> str:
        """Create psychoanalytically-informed visualization prompt."""
        object_a_relation = ""
        if 'object_a' in unconscious_data and signifier['name'] in str(unconscious_data['object_a']):
            object_a_relation = "This signifier circles the void of object a, the impossible object-cause of desire."
        
        prompt = f"""Create a surrealist image representing the unconscious signifier '{signifier['name']}'.

Psychoanalytic significance: {signifier.get('significance', 'An element of the unconscious.')}
Associations: {', '.join(signifier.get('associations', ['dreams', 'memories']))}
{object_a_relation}

Style: Surrealist art in the manner of Salvador Dalí and René Magritte
Mood: Uncanny, dreamlike, symbolic
Elements: Visual distortion, impossible geometries, symbolic condensation
Quality: High-resolution, detailed, evocative of unconscious depths"""
        
        return prompt

    def build_unconscious_memory(self, dream_data: Dict, agent_name: str, agent_dir: str) -> Dict[str, Any]:
        """
        Build complete unconscious memory structure.
        
        Args:
            dream_data: Raw dream narratives
            agent_name: Name of the agent
            agent_dir: Directory for saving agent files
            
        Returns:
            Complete structured unconscious memory
        """
        print(f"Building Lacanian unconscious structure for {agent_name}...")
        
        try:
            # Extract and analyze unconscious signifiers
            unconscious_data = self.extract_unconscious_signifiers(dream_data)
            
            # Build signifier graph
            graph = self.build_signifier_graph(
                unconscious_data.get('signifiers', []),
                unconscious_data.get('signifying_chains', [])
            )
            unconscious_data['signifier_graph'] = graph.serialize()
            
            # Generate visualizations (limited for production)
            try:
                visualizations = self.visualize_signifiers(unconscious_data, agent_name, agent_dir)
                unconscious_data['visualizations'] = visualizations
            except Exception as e:
                print(f"Warning: Visualization generation failed: {e}")
                unconscious_data['visualizations'] = {}
            
            # Add metadata
            unconscious_data['metadata'] = {
                'agent_name': agent_name,
                'extraction_date': datetime.now().isoformat(),
                'theoretical_framework': 'Lacanian Psychoanalysis',
                'dream_count': len(dream_data.get('dreams', [])),
                'fallback_mode': unconscious_data.get('fallback_mode', False)
            }
            
            # Save unconscious structure
            unconscious_path = f"{agent_dir}/unconscious_memory.json"
            save_json(unconscious_data, unconscious_path)
            print(f"✅ Unconscious memory saved to {unconscious_path}")
            
            return unconscious_data
            
        except Exception as e:
            print(f"❌ Error building unconscious memory: {e}")
            # Return minimal structure to prevent complete failure
            minimal_structure = self._create_fallback_unconscious_structure(dream_data)
            
            try:
                unconscious_path = f"{agent_dir}/unconscious_memory.json"
                save_json(minimal_structure, unconscious_path)
                print(f"⚠️ Saved fallback unconscious memory to {unconscious_path}")
            except Exception as save_error:
                print(f"❌ Could not save fallback structure: {save_error}")
            
            return minimal_structure