import json
import re, os
from datetime import datetime
from typing import Dict, List, Any, Optional
from interfaces.llm_interface import LLMInterface
from interfaces.vlm_interface import VLMInterface
from utils.lacanian_graph import LacanianSignifierGraph

class DreamGenerator:
    """
    Generates psychoanalytically authentic dreams based on activated signifiers from daily interactions.
    
    Uses Freudian dream-work mechanisms (condensation, displacement) combined with
    Lacanian signifier theory to create meaningful dream narratives.
    """
    
    def __init__(self, agent_name: str, memory_manager, base_path: str = "base_agents"):
        self.agent_name = agent_name
        self.memory_manager = memory_manager
        self.base_path = base_path
        self.agent_path = os.path.join(base_path, agent_name)
        
        # Initialize interfaces
        self.llm = LLMInterface()
        self.vlm = VLMInterface()
        
        # Load unconscious memory and signifier graph
        self._load_unconscious_memory()
        self._initialize_signifier_graph()
    
    def _load_unconscious_memory(self):
        """Load unconscious memory structures."""
        unconscious_path = os.path.join(self.agent_path, "unconscious_memory.json")
        try:
            with open(unconscious_path, 'r') as f:
                self.unconscious_memory = json.load(f)
        except Exception:
            self.unconscious_memory = {"signifiers": [], "signifying_chains": []}
    
    def _initialize_signifier_graph(self):
        """Initialize signifier graph from unconscious memory."""
        self.signifier_graph = LacanianSignifierGraph()
        
        # Reconstruct graph from saved data if available
        if self.unconscious_memory.get('signifier_graph'):
            graph_data = self.unconscious_memory['signifier_graph']
            
            # Add nodes
            for node in graph_data.get('nodes', []):
                self.signifier_graph.graph.add_node(
                    node['id'],
                    **{k: v for k, v in node.items() if k != 'id'}
                )
            
            # Add edges
            for edge in graph_data.get('edges', []):
                self.signifier_graph.graph.add_edge(
                    edge['source'],
                    edge['target'],
                    **{k: v for k, v in edge.items() if k not in ['source', 'target']}
                )
    
    def generate_dream(self, dream_context: str = "sleep") -> Dict[str, Any]:
        """
        Generate dream based on activated signifiers from recent interactions.
        
        Args:
            dream_context: Context for dream generation
            
        Returns:
            Dictionary containing complete dream data
        """
        try:
            # Get signifiers activated from recent memories
            activated_signifiers = self._get_activated_signifiers_from_memories()
            
            if not activated_signifiers:
                return self._generate_minimal_dream()
            
            # Get active signifying chains
            active_chains = self._get_active_signifying_chains(activated_signifiers)
            
            # Check for return of repressed content
            repressed_returns = self._check_return_of_repressed(activated_signifiers)
            
            # Generate dream narrative using psychoanalytic framework
            dream_data = self._generate_psychoanalytic_dream({
                'activated_signifiers': activated_signifiers,
                'signifying_chains': active_chains,
                'return_of_repressed': repressed_returns,
                'object_a': self.unconscious_memory.get('object_a', {}),
                'symptom': self.unconscious_memory.get('symptom', {}),
                'agent_name': self.agent_name
            })
            
            # Generate dream imagery
            dream_images = self._generate_dream_images(dream_data)
            
            # Create complete dream object
            dream = {
                "id": f"dream_{int(datetime.now().timestamp())}",
                "agent_name": self.agent_name,
                "timestamp": datetime.now().isoformat(),
                "title": dream_data.get("title", "Untitled Dream"),
                "narrative": dream_data.get("narrative", ""),
                "scenes": dream_data.get("scenes", []),
                "activated_signifiers": [s['signifier'] for s in activated_signifiers],
                "signifying_chains": [c['name'] for c in active_chains],
                "manifest_content": dream_data.get("manifest_content", []),
                "latent_content": dream_data.get("latent_content", []),
                "images": dream_images,
                "emotional_state": self.memory_manager.get_emotional_state()
            }
            
            # Save dream
            self._save_dream(dream)
            
            return dream
            
        except Exception as e:
            return {"error": str(e), "agent_name": self.agent_name}
    
    def _get_activated_signifiers_from_memories(self) -> List[Dict[str, Any]]:
        """
        Get signifiers activated from recent memory content with improved activation logic.
        """
        activated = []
        activation_map = {}  # Track activation sources
        
        # Get recent interactions from short-term memory
        recent_memories = self.memory_manager.get_short_term_memory(10)
        
        if not recent_memories:
            return []
        
        # Extract emotional themes from recent interactions
        emotional_themes = self._extract_emotional_themes(recent_memories)
        
        # Build comprehensive text from memories
        memory_segments = []
        for mem in recent_memories:
            content = mem.get("content", "")
            context = mem.get("context", "")
            
            # Weight user inputs higher than agent responses
            weight = 1.0 if context == "user_interaction" else 0.7
            memory_segments.append((content, weight))
        
        # Check which signifiers are activated by weighted content
        for signifier_obj in self.unconscious_memory.get("signifiers", []):
            if not isinstance(signifier_obj, dict):
                continue
                
            signifier_name = signifier_obj.get("name")
            if not signifier_name:
                continue
            
            activation_score = 0.0
            activation_reasons = []
            
            # Check for direct activation in weighted content
            for content, weight in memory_segments:
                content_lower = content.lower()
                
                # Direct mention
                if signifier_name.lower() in content_lower:
                    activation_score += 1.0 * weight
                    activation_reasons.append(f"direct mention (weight: {weight})")
                
                # Check associations
                associations = signifier_obj.get("associations", [])
                for assoc in associations:
                    if isinstance(assoc, str) and assoc.lower() in content_lower:
                        activation_score += 0.5 * weight
                        activation_reasons.append(f"association '{assoc}' (weight: {weight * 0.5})")
            
            # Check emotional resonance
            signifier_emotions = self._get_signifier_emotions(signifier_obj)
            for theme in emotional_themes:
                if theme in signifier_emotions:
                    activation_score += 0.3
                    activation_reasons.append(f"emotional resonance: {theme}")
            
            # Only activate if score is significant
            if activation_score >= 0.5:
                activated.append({
                    'signifier': signifier_name,
                    'activation_type': 'memory_based',
                    'activation_strength': min(1.0, activation_score),
                    'activation_reasons': activation_reasons,
                    'significance': signifier_obj.get('significance', ''),
                    'associations': signifier_obj.get('associations', [])
                })
        
        # Sort by activation strength
        activated.sort(key=lambda x: x['activation_strength'], reverse=True)
        
        # Apply dream-work: condensation and displacement
        activated = self._apply_dream_work_transformations(activated)
        
        return activated[:5]  # Limit to most relevant for focused dream
    
    def _extract_emotional_themes(self, memories: List[Dict]) -> List[str]:
        """Extract dominant emotional themes from recent memories."""
        themes = []
        
        for mem in memories:
            emotional_state = mem.get('emotional_state', {})
            emotion_category = emotional_state.get('emotion_category', '')
            
            if emotion_category and emotion_category != 'neutral':
                themes.append(emotion_category)
        
        # Return unique themes
        return list(set(themes))
    
    def _get_signifier_emotions(self, signifier: Dict) -> List[str]:
        """Extract emotional associations from signifier."""
        emotions = []
        
        # From significance
        significance = signifier.get('significance', '').lower()
        emotion_words = ['fear', 'love', 'anger', 'anxiety', 'joy', 'sadness', 'jealousy']
        for word in emotion_words:
            if word in significance:
                emotions.append(word)
        
        # From associations
        for assoc in signifier.get('associations', []):
            if isinstance(assoc, str):
                assoc_lower = assoc.lower()
                for word in emotion_words:
                    if word in assoc_lower:
                        emotions.append(word)
        
        return emotions
    
    def _apply_dream_work_transformations(self, signifiers: List[Dict]) -> List[Dict]:
        """Apply condensation and displacement to signifiers."""
        if len(signifiers) < 2:
            return signifiers
        
        # Condensation: merge related signifiers
        condensed = []
        processed = set()
        
        for i, sig1 in enumerate(signifiers):
            if i in processed:
                continue
                
            # Look for signifiers to condense with
            for j, sig2 in enumerate(signifiers[i+1:], i+1):
                if j in processed:
                    continue
                    
                # Check if they share associations or are in same chain
                shared_assoc = set(sig1.get('associations', [])) & set(sig2.get('associations', []))
                
                if shared_assoc or self._are_in_same_chain(sig1['signifier'], sig2['signifier']):
                    # Create condensed signifier
                    condensed.append({
                        'signifier': f"{sig1['signifier']}-{sig2['signifier']}",
                        'activation_type': 'condensation',
                        'activation_strength': (sig1['activation_strength'] + sig2['activation_strength']) / 2,
                        'condensed_from': [sig1['signifier'], sig2['signifier']],
                        'significance': f"Condensation of {sig1['signifier']} and {sig2['signifier']}",
                        'associations': list(set(sig1.get('associations', []) + sig2.get('associations', [])))
                    })
                    processed.add(i)
                    processed.add(j)
                    break
            
            if i not in processed:
                condensed.append(sig1)
                processed.add(i)
        
        # Displacement: shift emotional charge
        if len(condensed) > 1:
            # Find signifier with highest emotional charge
            max_charge_idx = 0
            max_charge = condensed[0]['activation_strength']
            
            for i, sig in enumerate(condensed[1:], 1):
                if sig['activation_strength'] > max_charge:
                    max_charge = sig['activation_strength']
                    max_charge_idx = i
            
            # Displace some charge to a random other signifier
            if max_charge_idx != 0:
                target_idx = (max_charge_idx + 1) % len(condensed)
                
                # Transfer some activation
                transfer_amount = condensed[max_charge_idx]['activation_strength'] * 0.3
                condensed[max_charge_idx]['activation_strength'] -= transfer_amount
                condensed[target_idx]['activation_strength'] += transfer_amount
                
                # Note displacement
                condensed[target_idx]['displacement_from'] = condensed[max_charge_idx]['signifier']
                condensed[target_idx]['activation_type'] = 'displacement'
        
        return condensed
    
    def _are_in_same_chain(self, sig1: str, sig2: str) -> bool:
        """Check if two signifiers are in the same signifying chain."""
        for chain in self.unconscious_memory.get('signifying_chains', []):
            if isinstance(chain, dict):
                chain_signifiers = chain.get('signifiers', [])
                if sig1 in chain_signifiers and sig2 in chain_signifiers:
                    return True
        return False
    
    def _get_active_signifying_chains(self, activated_signifiers: List[Dict]) -> List[Dict]:
        """Get signifying chains involving activated signifiers."""
        active_chains = []
        activated_names = []
        
        # Extract base signifier names (handle condensed signifiers)
        for sig in activated_signifiers:
            sig_name = sig['signifier']
            if '-' in sig_name and sig.get('activation_type') == 'condensation':
                # Split condensed signifiers
                activated_names.extend(sig.get('condensed_from', [sig_name]))
            else:
                activated_names.append(sig_name)
        
        for chain in self.unconscious_memory.get('signifying_chains', []):
            if not isinstance(chain, dict):
                continue
                
            chain_signifiers = chain.get('signifiers', [])
            
            # Check if any signifiers in this chain are activated
            activated_in_chain = [s for s in chain_signifiers if s in activated_names]
            
            if activated_in_chain:
                active_chains.append({
                    "name": chain.get('name', 'unnamed_chain'),
                    "signifiers": chain_signifiers,
                    "explanation": chain.get('explanation', ''),
                    "activated_nodes": activated_in_chain,
                    "activation_ratio": len(activated_in_chain) / len(chain_signifiers)
                })
        
        return active_chains
    
    def _check_return_of_repressed(self, activated_signifiers: List[Dict]) -> List[Dict]:
        """Check if any repressed content is returning through dreams."""
        returns = []
        
        for sig_data in activated_signifiers:
            signifier = sig_data['signifier'].split('-')[0]  # Handle condensed signifiers
            
            if signifier in self.signifier_graph.graph:
                node_data = self.signifier_graph.graph.nodes[signifier]
                if node_data.get('repressed', False):
                    returns.append({
                        "signifier": signifier,
                        "significance": sig_data.get("significance", ""),
                        "return_context": "Emerging through dream after activation",
                        "activation_strength": sig_data.get('activation_strength', 0.5)
                    })
        
        return returns
    
    def _generate_psychoanalytic_dream(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate dream using psychoanalytic framework with proper JSON parsing."""
        try:
            # Generate dream with LLM
            result = self.llm.generate("phase2", "generate_dream", context)
            
            if result:
                # Clean up the response - remove markdown blocks and parse JSON
                cleaned_result = self._clean_json_response(result)
                
                try:
                    dream_data = json.loads(cleaned_result)
                    
                    # Ensure all required fields
                    if self._validate_dream_structure(dream_data):
                        return dream_data
                    else:
                        # Fix structure if needed
                        return self._fix_dream_structure(dream_data)
                        
                except json.JSONDecodeError as e:
                    print(f"JSON decode error: {e}")
                    # Try to extract narrative at least
                    return self._extract_narrative_fallback(result)
            
            # Fallback to generating from signifiers
            return self._generate_signifier_based_dream(context)
            
        except Exception as e:
            print(f"Error in dream generation: {e}")
            return self._generate_signifier_based_dream(context)
    
    def _clean_json_response(self, response: str) -> str:
        """Clean LLM response to extract valid JSON."""
        # Remove markdown code blocks
        cleaned = re.sub(r'```json\s*', '', response)
        cleaned = re.sub(r'```\s*', '', cleaned)
        
        # Find the first { or [ and last } or ]
        start_idx = cleaned.find('{')
        if start_idx == -1:
            start_idx = cleaned.find('[')
        
        if start_idx == -1:
            return response
        
        # Find matching closing bracket
        stack = []
        end_idx = -1
        
        for i in range(start_idx, len(cleaned)):
            if cleaned[i] in '{[':
                stack.append(cleaned[i])
            elif cleaned[i] in '}]':
                if stack:
                    stack.pop()
                    if not stack:
                        end_idx = i
                        break
        
        if end_idx != -1:
            return cleaned[start_idx:end_idx + 1]
        
        return cleaned[start_idx:]
    
    def _validate_dream_structure(self, dream_data: Dict) -> bool:
        """Validate that dream data has required structure."""
        required_fields = ['title', 'narrative', 'scenes', 'manifest_content', 'latent_content']
        
        for field in required_fields:
            if field not in dream_data:
                return False
        
        # Check scenes structure
        if not isinstance(dream_data['scenes'], list) or len(dream_data['scenes']) == 0:
            return False
        
        for scene in dream_data['scenes']:
            if not isinstance(scene, dict):
                return False
            if 'narrative' not in scene or 'setting' not in scene:
                return False
        
        return True
    
    def _fix_dream_structure(self, dream_data: Dict) -> Dict:
        """Fix incomplete dream structure."""
        # Ensure all required fields
        fixed = {
            'title': dream_data.get('title', 'Untitled Dream'),
            'narrative': dream_data.get('narrative', ''),
            'scenes': dream_data.get('scenes', []),
            'manifest_content': dream_data.get('manifest_content', []),
            'latent_content': dream_data.get('latent_content', [])
        }
        
        # Fix scenes if needed
        if not fixed['scenes'] and fixed['narrative']:
            # Create a single scene from narrative
            fixed['scenes'] = [{
                'setting': 'Dream space',
                'narrative': fixed['narrative'],
                'symbols': [],
                'signifiers_expressed': [],
                'visual_description': 'Surrealist dreamscape'
            }]
        
        # Fix manifest/latent content if empty
        if not fixed['manifest_content']:
            fixed['manifest_content'] = ['dream imagery', 'unconscious symbols']
        
        if not fixed['latent_content']:
            fixed['latent_content'] = ['hidden desires', 'repressed thoughts']
        
        return fixed
    
    def _extract_narrative_fallback(self, response: str) -> Dict[str, Any]:
        """Extract narrative from response when JSON parsing fails."""
        # Look for narrative markers
        narrative = ""
        
        if "narrative" in response:
            narrative_match = re.search(r'"narrative"\s*:\s*"([^"]+)"', response)
            if narrative_match:
                narrative = narrative_match.group(1)
        
        if not narrative:
            # Take first substantial text block
            sentences = re.split(r'[.!?]+', response)
            narrative = '. '.join(sentences[:3]) + '.'
        
        return {
            'title': 'Dream Fragment',
            'narrative': narrative,
            'scenes': [{
                'setting': 'Unconscious landscape',
                'narrative': narrative,
                'symbols': [],
                'signifiers_expressed': [],
                'visual_description': 'Abstract dream imagery'
            }],
            'manifest_content': ['fragmented images'],
            'latent_content': ['unconscious material']
        }
    
    def _generate_signifier_based_dream(self, context: Dict) -> Dict[str, Any]:
        """Generate dream directly from signifier content."""
        signifiers = context['activated_signifiers']
        
        if not signifiers:
            return self._generate_minimal_dream()
        
        # Build narrative using dream-work principles
        primary_sig = signifiers[0]
        narrative_parts = []
        
        # Opening with primary signifier transformed
        narrative_parts.append(
            f"I found myself in a strange place where {primary_sig['signifier']} "
            f"appeared, but it wasn't quite right - it was somehow different, transformed."
        )
        
        # Add condensation if multiple signifiers
        if len(signifiers) > 1:
            secondary_sig = signifiers[1]
            narrative_parts.append(
                f"Then {primary_sig['signifier']} began to merge with {secondary_sig['signifier']}, "
                f"creating something that was both and neither at the same time."
            )
        
        # Add displacement
        if primary_sig.get('associations'):
            assoc = primary_sig['associations'][0]
            narrative_parts.append(
                f"The feeling I expected wasn't there - instead, {assoc} "
                f"carried all the emotional weight, leaving me confused."
            )
        
        # Add return of repressed if present
        repressed_returns = context.get('return_of_repressed', [])
        if repressed_returns:
            narrative_parts.append(
                f"Something long forgotten pushed through: {repressed_returns[0]['signifier']}. "
                f"I knew I had been avoiding this, but here it was, undeniable."
            )
        
        # Closing with typical dream dissolution
        narrative_parts.append(
            "The dream began to fragment, pieces sliding away from each other "
            "until I couldn't hold onto the meaning anymore."
        )
        
        narrative = " ".join(narrative_parts)
        
        # Create scenes from narrative segments
        scenes = []
        for i, part in enumerate(narrative_parts[:3]):
            scenes.append({
                'setting': f'Dream sequence {i+1}',
                'narrative': part,
                'symbols': [s['signifier'] for s in signifiers[:2]],
                'signifiers_expressed': [s['signifier'] for s in signifiers],
                'visual_description': self._create_visual_description(part, signifiers)
            })
        
        return {
            'title': f"Dream of {primary_sig['signifier']}",
            'narrative': narrative,
            'scenes': scenes,
            'manifest_content': [s['signifier'] for s in signifiers] + ['transformation', 'merging'],
            'latent_content': [s.get('significance', 'unconscious desire') for s in signifiers]
        }
    
    def _generate_minimal_dream(self) -> Dict[str, Any]:
        """Generate minimal dream when no signifiers are activated."""
        return {
            "title": "Empty Dream Space",
            "narrative": "I found myself in a vast, empty space. Nothing seemed to happen, yet there was a sense of waiting, of something just beyond reach. The emptiness itself seemed significant, as if it were holding space for something that couldn't yet appear.",
            "scenes": [{
                "setting": "Void",
                "narrative": "An endless expanse of possibility, neither dark nor light",
                "symbols": ["emptiness", "potential"],
                "signifiers_expressed": [],
                "visual_description": "Minimalist void with subtle gradients suggesting hidden depth"
            }],
            "manifest_content": ["empty space", "waiting", "vastness"],
            "latent_content": ["unfulfilled desire", "potential", "the void of object a"]
        }
    
    def _create_visual_description(self, scene_text: str, signifiers: List[Dict]) -> str:
        """Create visual description for image generation."""
        base = "A surrealist dream scene in the style of Salvador Dalí and René Magritte, featuring "
        
        if signifiers:
            # Use primary signifiers for visual focus
            primary_elements = [s['signifier'] for s in signifiers[:2]]
            base += f"symbolic representations of {' and '.join(primary_elements)}, "
        
        # Add dream-work elements
        if 'merge' in scene_text.lower() or 'transform' in scene_text.lower():
            base += "with elements morphing and merging impossibly, "
        
        if 'feeling' in scene_text.lower() or 'emotional' in scene_text.lower():
            base += "where emotional charge shifts between objects unexpectedly, "
        
        base += "rendered with dreamlike distortions, impossible perspectives, and symbolic condensation"
        
        return base
    
    def _generate_dream_images(self, dream_data: Dict) -> List[Dict[str, Any]]:
        """Generate images for key dream scenes."""
        images = []
        
        # Limit to 2 most significant scenes
        scenes_to_render = dream_data.get('scenes', [])[:2]
        
        for i, scene in enumerate(scenes_to_render):
            visual_desc = scene.get('visual_description', '')
            signifiers = scene.get('signifiers_expressed', [])
            
            if visual_desc:
                # Create comprehensive prompt
                prompt = self._create_image_prompt(visual_desc, signifiers, scene)
                
                # Generate image
                timestamp = int(datetime.now().timestamp())
                image_filename = f"dream_{timestamp}_scene_{i+1}"
                image_output_path = os.path.join(self.agent_path, "dreams", "images", image_filename)
                os.makedirs(os.path.dirname(image_output_path), exist_ok=True)
                
                try:
                    result = self.vlm.direct_image_generation(prompt, image_output_path)
                    if result and result.get("success"):
                        images.append({
                            "scene_number": i + 1,
                            "image_path": result.get("image_path"),
                            "signifiers_depicted": signifiers,
                            "description": visual_desc
                        })
                except Exception as e:
                    print(f"Error generating image for scene {i+1}: {e}")
        
        return images
    
    def _create_image_prompt(self, visual_desc: str, signifiers: List[str], scene: Dict) -> str:
        """Create detailed prompt for dream image generation."""
        prompt_parts = [visual_desc]
        
        # Add specific surrealist techniques
        techniques = [
            "Use perspective distortion and impossible geometry",
            "Include metamorphosis of forms",
            "Employ symbolic juxtaposition",
            "Create uncanny atmosphere with familiar yet strange elements"
        ]
        
        prompt_parts.append(techniques[hash(str(signifiers)) % len(techniques)])
        
        # Add emotional atmosphere based on scene
        if 'merge' in scene.get('narrative', '').lower():
            prompt_parts.append("Show elements blending and merging in impossible ways")
        
        if 'forgotten' in scene.get('narrative', '').lower():
            prompt_parts.append("Include faded or ghostly elements suggesting repressed memories")
        
        # Final touches
        prompt_parts.append("High detail, oil painting texture, dramatic lighting, surrealist masterpiece")
        
        return ". ".join(prompt_parts)
    
    def _save_dream(self, dream: Dict[str, Any]) -> None:
        """Save dream to JSON file."""
        try:
            dreams_dir = os.path.join(self.agent_path, "dreams")
            os.makedirs(dreams_dir, exist_ok=True)
            
            # Save JSON
            dream_filename = f"{dream['id']}.json"
            dream_path = os.path.join(dreams_dir, dream_filename)
            with open(dream_path, 'w') as f:
                json.dump(dream, f, indent=2)
            
            # Save readable version
            self._save_readable_dream(dream, dreams_dir)
            
        except Exception as e:
            print(f"Error saving dream: {e}")
    
    def _save_readable_dream(self, dream: Dict[str, Any], dreams_dir: str) -> None:
        """Save human-readable dream format."""
        try:
            date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            readable_filename = f"dream_{date_str}_readable.txt"
            readable_path = os.path.join(dreams_dir, readable_filename)
            
            with open(readable_path, 'w') as f:
                f.write(f"DREAM RECORD - {dream['agent_name'].upper()}\n")
                f.write(f"Date: {dream['timestamp']}\n")
                f.write("=" * 50 + "\n\n")
                
                f.write(f"TITLE: {dream.get('title', 'Untitled')}\n")
                f.write("-" * 30 + "\n\n")
                
                f.write("ACTIVATED SIGNIFIERS:\n")
                for sig in dream.get('activated_signifiers', []):
                    f.write(f"- {sig}\n")
                
                f.write(f"\nDREAM NARRATIVE:\n\n")
                f.write(dream.get('narrative', 'No narrative recorded'))
                
                if dream.get('scenes'):
                    f.write(f"\n\nDREAM SCENES:\n")
                    for i, scene in enumerate(dream['scenes']):
                        f.write(f"\nScene {i+1}: {scene.get('setting', 'Unknown')}\n")
                        f.write(scene.get('narrative', ''))
                        f.write("\n")
                
                f.write(f"\n\nMANIFEST CONTENT: {', '.join(dream.get('manifest_content', []))}\n")
                f.write(f"LATENT CONTENT: {', '.join(dream.get('latent_content', []))}\n")
                
                f.write(f"\n" + "=" * 50 + "\n")
                
        except Exception as e:
            print(f"Error saving readable dream: {e}")