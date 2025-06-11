import os
import json
import re
from datetime import datetime
from typing import Dict, List, Any
from interfaces.vlm_interface import VLMInterface
from utils.lacanian_graph import LacanianSignifierGraph

class DreamGenerator:
    """
    Generates psychoanalytically authentic dreams using ALL extracted unconscious data.
    
    Dreams emerge from actual signifying chains, repressed content, object_a dynamics,
    and symptom patterns rather than random content.
    """
    
    def __init__(self, agent_name: str, memory_manager, base_path: str = "base_agents"):
        self.agent_name = agent_name
        self.memory_manager = memory_manager
        self.base_path = base_path
        self.agent_path = os.path.join(base_path, agent_name)
        
        # Use VLM for dream generation
        self.vlm = VLMInterface()
        
        # Load and prepare unconscious memory
        self._load_unconscious_memory()
        self._prepare_dream_data()
        
        print(f"Dream generator initialized for {agent_name}")
        print(f"Available for dreams: {len(self.signifiers)} signifiers, {len(self.chains)} chains")
        print(f"Repressed content: {len(self.repressed_signifiers)} signifiers")
    
    def _load_unconscious_memory(self):
        """Load unconscious memory structures."""
        unconscious_path = os.path.join(self.agent_path, "unconscious_memory.json")
        try:
            with open(unconscious_path, 'r') as f:
                self.unconscious_memory = json.load(f)
        except Exception as e:
            print(f"Error loading unconscious memory: {e}")
            self.unconscious_memory = {"signifiers": [], "signifying_chains": []}
    
    def _prepare_dream_data(self):
        """Prepare unconscious data for dream generation."""
        # Extract core structures
        self.signifiers = self.unconscious_memory.get('signifiers', [])
        self.chains = self.unconscious_memory.get('signifying_chains', [])
        self.object_a = self.unconscious_memory.get('object_a', {})
        self.symptom = self.unconscious_memory.get('symptom', {})
        
        # Create lookup maps
        self.signifier_map = {sig['name']: sig for sig in self.signifiers if isinstance(sig, dict)}
        
        # Identify repressed signifiers
        self.repressed_signifiers = [
            sig for sig in self.signifiers 
            if isinstance(sig, dict) and sig.get('repressed', False)
        ]
        
        # Extract object_a manifestations and void
        self.object_a_manifestations = self.object_a.get('manifestations', [])
        self.void_manifestations = self.object_a.get('void_manifestations', [])
        
        # Extract symptom patterns
        self.symptom_signifiers = self.symptom.get('signifiers_involved', [])
        self.jouissance_pattern = self.symptom.get('jouissance_pattern', '')
        
        # Prepare signifying chains
        self.chain_map = {}
        for chain in self.chains:
            if isinstance(chain, dict):
                name = chain.get('name', '')
                self.chain_map[name] = chain
    
    def generate_dream(self, dream_context: str = "sleep") -> Dict[str, Any]:
        """
        Generate psychoanalytically authentic dream using extracted unconscious data.
        """
        try:
            print(f"\n=== Generating Dream for {self.agent_name} ===")
            
            # 1. Determine which signifiers should appear based on recent activity
            activated_signifiers = self._get_signifiers_for_dream()
            
            if not activated_signifiers:
                return self._generate_empty_dream()
            
            # 2. Check which signifying chains are activated
            active_chains = self._get_active_chains_for_dream(activated_signifiers)
            
            # 3. Determine repressed content to emerge
            emerging_repressed = self._select_repressed_content_for_dream(activated_signifiers)
            
            # 4. Calculate object_a dynamics in dream
            object_a_dream_dynamics = self._calculate_object_a_in_dream(activated_signifiers)
            
            # 5. Determine symptom manifestations
            symptom_manifestations = self._get_symptom_manifestations_in_dream(activated_signifiers)
            
            # 6. Generate dream narrative using VLM with unconscious structure
            dream_data = self._generate_dream_with_unconscious_structure({
                'agent_name': self.agent_name,
                'activated_signifiers': activated_signifiers,
                'signifying_chains': active_chains,
                'emerging_repressed': emerging_repressed,
                'object_a_dynamics': object_a_dream_dynamics,
                'symptom_manifestations': symptom_manifestations
            })
            
            # 7. Generate dream imagery using VLM
            dream_images = self._generate_dream_images_with_vlm(dream_data)
            
            # 8. Create complete dream object
            dream = {
                "id": f"dream_{int(datetime.now().timestamp())}",
                "agent_name": self.agent_name,
                "timestamp": datetime.now().isoformat(),
                "title": dream_data.get("title", "Unconscious Dream"),
                "narrative": dream_data.get("narrative", ""),
                "scenes": dream_data.get("scenes", []),
                "activated_signifiers": [s['name'] for s in activated_signifiers],
                "active_chains": [c['name'] for c in active_chains],
                "emerging_repressed": [r['signifier'] for r in emerging_repressed],
                "object_a_proximity": object_a_dream_dynamics.get('proximity_level', 0.0),
                "symptom_activation": len(symptom_manifestations) > 0,
                "manifest_content": dream_data.get("manifest_content", []),
                "latent_content": dream_data.get("latent_content", []),
                "images": dream_images,
                "emotional_state": self.memory_manager.get_emotional_state(),
                "generation_method": "unconscious_structure_based"
            }
            
            # Save dream
            self._save_dream(dream)
            
            print(f"Dream generated: '{dream['title']}'")
            print(f"Signifiers: {dream['activated_signifiers']}")
            print(f"Chains: {dream['active_chains']}")
            print(f"Repressed emerging: {dream['emerging_repressed']}")
            
            return dream
            
        except Exception as e:
            print(f"Error generating dream: {e}")
            return self._generate_empty_dream()
    
    def _get_signifiers_for_dream(self) -> List[Dict[str, Any]]:
        """Determine which signifiers should appear in dream based on recent activity."""
        activated_signifiers = []
        
        # Get recent memory content
        recent_memories = self.memory_manager.get_short_term_memory(8)
        
        if not recent_memories:
            # No recent activity - use repressed content as primary source
            for repressed_sig in self.repressed_signifiers[:3]:
                activated_signifiers.append({
                    'name': repressed_sig['name'],
                    'activation_strength': 0.8,
                    'activation_source': 'repressed_emergence',
                    'signifier_data': repressed_sig
                })
            return activated_signifiers
        
        # Extract text from recent memories
        memory_text = " ".join([
            mem.get("content", "") for mem in recent_memories
        ]).lower()
        
        # Check which signifiers are activated by recent content
        for signifier in self.signifiers:
            if not isinstance(signifier, dict):
                continue
                
            sig_name = signifier.get('name', '')
            activation_strength = 0.0
            
            # Direct activation
            if sig_name.lower() in memory_text:
                activation_strength += 1.0
            
            # Association activation
            associations = signifier.get('associations', [])
            for assoc in associations:
                if isinstance(assoc, str) and assoc.lower() in memory_text:
                    activation_strength += 0.5
            
            # Significance activation
            significance = signifier.get('significance', '').lower()
            sig_words = [w for w in significance.split() if len(w) > 4]
            for word in sig_words:
                if word in memory_text:
                    activation_strength += 0.3
            
            if activation_strength > 0:
                activated_signifiers.append({
                    'name': sig_name,
                    'activation_strength': activation_strength,
                    'activation_source': 'recent_memory',
                    'signifier_data': signifier
                })
        
        # Always include some repressed content in dreams
        for repressed_sig in self.repressed_signifiers[:2]:
            activated_signifiers.append({
                'name': repressed_sig['name'],
                'activation_strength': 0.6,
                'activation_source': 'repressed_return',
                'signifier_data': repressed_sig
            })
        
        # Sort and limit
        activated_signifiers.sort(key=lambda x: x['activation_strength'], reverse=True)
        return activated_signifiers[:6]
    
    def _get_active_chains_for_dream(self, activated_signifiers: List[Dict]) -> List[Dict]:
        """Determine which signifying chains are active in the dream."""
        active_chains = []
        activated_names = [sig['name'] for sig in activated_signifiers]
        
        for chain_name, chain_data in self.chain_map.items():
            chain_signifiers = chain_data.get('signifiers', [])
            
            # Check how many signifiers in this chain are activated
            active_in_chain = [sig for sig in activated_names if sig in chain_signifiers]
            
            if active_in_chain:
                activation_strength = len(active_in_chain) / len(chain_signifiers)
                
                active_chains.append({
                    'name': chain_name,
                    'signifiers': chain_signifiers,
                    'active_signifiers': active_in_chain,
                    'activation_strength': activation_strength,
                    'chain_type': chain_data.get('type', 'mixed'),
                    'explanation': chain_data.get('explanation', ''),
                    'relation_to_fantasy': chain_data.get('relation_to_fantasy', '')
                })
        
        return active_chains
    
    def _select_repressed_content_for_dream(self, activated_signifiers: List[Dict]) -> List[Dict]:
        """Select repressed content to emerge in dream."""
        emerging_repressed = []
        
        # Check if any activated signifiers are repressed
        for sig_data in activated_signifiers:
            if sig_data['signifier_data'].get('repressed', False):
                emerging_repressed.append({
                    'signifier': sig_data['name'],
                    'emergence_strength': sig_data['activation_strength'],
                    'emergence_type': 'direct_return',
                    'associations': sig_data['signifier_data'].get('associations', []),
                    'significance': sig_data['signifier_data'].get('significance', '')
                })
        
        # Add additional repressed content that might emerge
        for repressed_sig in self.repressed_signifiers:
            if repressed_sig['name'] not in [er['signifier'] for er in emerging_repressed]:
                # Check if this repressed signifier has associations with activated ones
                repressed_assocs = repressed_sig.get('associations', [])
                activated_assocs = []
                for activated_sig in activated_signifiers:
                    activated_assocs.extend(activated_sig['signifier_data'].get('associations', []))
                
                # If there's association overlap, it might emerge
                if any(assoc in activated_assocs for assoc in repressed_assocs):
                    emerging_repressed.append({
                        'signifier': repressed_sig['name'],
                        'emergence_strength': 0.4,
                        'emergence_type': 'associative_return',
                        'associations': repressed_assocs,
                        'significance': repressed_sig.get('significance', '')
                    })
                    break  # Limit to one associative emergence
        
        return emerging_repressed
    
    def _calculate_object_a_in_dream(self, activated_signifiers: List[Dict]) -> Dict[str, Any]:
        """Calculate how object_a manifests in the dream."""
        object_a_dynamics = {
            'proximity_level': 0.0,
            'manifestation_type': 'absent',
            'void_representations': [],
            'desire_direction': 'neutral'
        }
        
        activated_names = [sig['name'] for sig in activated_signifiers]
        
        # Check if object_a manifestations are present
        for manifestation in self.object_a_manifestations:
            for activated_name in activated_names:
                if activated_name.lower() in manifestation.lower():
                    object_a_dynamics['proximity_level'] += 0.5
                    object_a_dynamics['manifestation_type'] = 'substitute_present'
        
        # Check void manifestations
        for void_manifest in self.void_manifestations:
            # Extract key words from void manifestation
            void_words = void_manifest.lower().split()
            for activated_name in activated_names:
                if any(activated_name.lower() in word for word in void_words):
                    object_a_dynamics['proximity_level'] += 0.7
                    object_a_dynamics['void_representations'].append(void_manifest)
                    object_a_dynamics['manifestation_type'] = 'void_circled'
        
        # Determine desire direction
        if object_a_dynamics['proximity_level'] > 0.6:
            object_a_dynamics['desire_direction'] = 'approaching_void'
        elif object_a_dynamics['proximity_level'] > 0.3:
            object_a_dynamics['desire_direction'] = 'seeking_substitute'
        
        return object_a_dynamics
    
    def _get_symptom_manifestations_in_dream(self, activated_signifiers: List[Dict]) -> List[Dict]:
        """Determine how symptom manifests in dream."""
        symptom_manifestations = []
        activated_names = [sig['name'] for sig in activated_signifiers]
        
        # Check if symptom signifiers are activated
        for symptom_sig in self.symptom_signifiers:
            if symptom_sig in activated_names:
                symptom_manifestations.append({
                    'signifier': symptom_sig,
                    'manifestation_type': 'direct',
                    'jouissance_pattern': self.jouissance_pattern,
                    'repetition_structure': self.symptom.get('repetition_structure', '')
                })
        
        return symptom_manifestations
    
    def _generate_dream_with_unconscious_structure(self, dream_context: Dict) -> Dict[str, Any]:
        """Generate dream narrative using VLM with complete unconscious structure."""
        try:
            # Use VLM to generate psychoanalytically structured dream
            result = self.vlm.generate_text("phase2", "generate_dream", dream_context)
            
            # Parse VLM response
            dream_data = self._parse_vlm_dream_response(result)
            
            if dream_data:
                return dream_data
            else:
                # Generate from unconscious structure directly
                return self._generate_from_unconscious_structure(dream_context)
                
        except Exception as e:
            print(f"VLM dream generation failed: {e}")
            return self._generate_from_unconscious_structure(dream_context)
    
    def _generate_from_unconscious_structure(self, dream_context: Dict) -> Dict[str, Any]:
        """Generate dream directly from unconscious structure when VLM fails."""
        activated_signifiers = dream_context.get('activated_signifiers', [])
        chains = dream_context.get('signifying_chains', [])
        repressed = dream_context.get('emerging_repressed', [])
        object_a = dream_context.get('object_a_dynamics', {})
        
        if not activated_signifiers:
            return self._generate_empty_dream()
        
        # Create narrative from signifying chains
        narrative_parts = []
        
        # Start with most activated signifier
        primary_sig = activated_signifiers[0]
        narrative_parts.append(f"In the dream, {primary_sig['name']} appeared.")
        
        # Follow signifying chains
        for chain in chains:
            if chain.get('activation_strength', 0) > 0.5:
                chain_sigs = chain.get('active_signifiers', [])
                if len(chain_sigs) > 1:
                    narrative_parts.append(
                        f"Then {chain_sigs[1]} emerged, connected to {chain_sigs[0]} "
                        f"through {chain['explanation'][:50]}..."
                    )
        
        # Add repressed content emergence
        for rep in repressed:
            if rep.get('emergence_strength', 0) > 0.5:
                narrative_parts.append(
                    f"Suddenly, {rep['signifier']} appeared unexpectedly, "
                    f"bringing {rep.get('significance', 'unclear feelings')}."
                )
        
        # Add object_a dynamics
        if object_a.get('proximity_level', 0) > 0.3:
            if object_a.get('manifestation_type') == 'void_circled':
                narrative_parts.append(
                    "There was a sense of something missing, an empty space "
                    "that seemed important but couldn't be reached."
                )
            elif object_a.get('manifestation_type') == 'substitute_present':
                narrative_parts.append(
                    "Someone appeared who seemed familiar yet not quite right, "
                    "as if they were standing in for someone else."
                )
        
        narrative = " ".join(narrative_parts)
        
        # Create scenes
        scenes = []
        for i, sig in enumerate(activated_signifiers[:3]):
            scene_signifiers = [sig['name']]
            
            # Add related signifiers from chains
            for chain in chains:
                if sig['name'] in chain.get('signifiers', []):
                    scene_signifiers.extend([s for s in chain['signifiers'] if s != sig['name']])
            
            scenes.append({
                "setting": f"Dream space {i+1}",
                "narrative": f"A scene involving {sig['name']} and its associations",
                "signifiers_expressed": scene_signifiers[:4],
                "visual_description": f"Surrealist scene featuring {sig['name']} with symbolic elements"
            })
        
        return {
            "title": f"Dream of {primary_sig['name']}",
            "narrative": narrative,
            "scenes": scenes,
            "manifest_content": [sig['name'] for sig in activated_signifiers],
            "latent_content": [rep['signifier'] for rep in repressed] + 
                             [sig['name'] for sig in activated_signifiers if sig['signifier_data'].get('repressed')]
        }
    
    def _parse_vlm_dream_response(self, response: str) -> Dict[str, Any]:
        """Parse VLM dream generation response."""
        try:
            # Try to extract JSON
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
            
            return json.loads(response)

        except Exception:
            return None
    
    def _generate_dream_images_with_vlm(self, dream_data: Dict) -> List[Dict[str, Any]]:
        """Generate images for dream scenes using VLM."""
        images = []
        
        for i, scene in enumerate(dream_data.get('scenes', [])[:2]):
            visual_desc = scene.get('visual_description', '')
            signifiers = scene.get('signifiers_expressed', [])
            
            if visual_desc:
                try:
                    # Generate image
                    image_filename = f"dream_{int(datetime.now().timestamp())}_scene_{i+1}"
                    image_output_path = os.path.join(self.agent_path, "dreams", "images", image_filename)
                    os.makedirs(os.path.dirname(image_output_path), exist_ok=True)
                    
                    # Create enhanced prompt for unconscious content
                    enhanced_prompt = (
                        f"Surrealist dream image: {visual_desc}. "
                        f"Featuring unconscious signifiers: {', '.join(signifiers[:3])}. "
                        f"Style of Salvador Dalí and René Magritte, dreamlike, symbolic, "
                        f"high resolution, evocative of unconscious depths."
                    )
                    
                    result = self.vlm.direct_image_generation(enhanced_prompt, image_output_path)
                    if result and result.get("success"):
                        images.append({
                            "scene_number": i + 1,
                            "image_path": result.get("image_path"),
                            "signifiers_depicted": signifiers,
                            "description": visual_desc,
                            "generation_method": "unconscious_based"
                        })
                        
                except Exception as e:
                    print(f"Error generating dream image {i+1}: {e}")
        
        return images
    
    def _generate_empty_dream(self) -> Dict[str, Any]:
        """Generate minimal dream when no content available."""
        return {
            "id": f"dream_{int(datetime.now().timestamp())}",
            "agent_name": self.agent_name,
            "timestamp": datetime.now().isoformat(),
            "title": "Empty Dream Space",
            "narrative": "A vast, empty space. Waiting. Something just beyond reach.",
            "scenes": [{
                "setting": "Void",
                "narrative": "Endless possibility",
                "signifiers_expressed": [],
                "visual_description": "Minimalist void with subtle emotional undertones"
            }],
            "activated_signifiers": [],
            "active_chains": [],
            "emerging_repressed": [],
            "object_a_proximity": 0.0,
            "symptom_activation": False,
            "manifest_content": ["void", "waiting"],
            "latent_content": ["potential"],
            "images": [],
            "emotional_state": self.memory_manager.get_emotional_state(),
            "generation_method": "empty_fallback"
        }
    
    def _save_dream(self, dream: Dict[str, Any]) -> None:
        """Save dream with detailed analysis."""
        try:
            dreams_dir = os.path.join(self.agent_path, "dreams")
            os.makedirs(dreams_dir, exist_ok=True)
            
            # Save JSON
            dream_filename = f"{dream['id']}.json"
            dream_path = os.path.join(dreams_dir, dream_filename)
            with open(dream_path, 'w') as f:
                json.dump(dream, f, indent=2)
            
            # Save readable analysis
            self._save_dream_analysis(dream, dreams_dir)
            
        except Exception as e:
            print(f"Error saving dream: {e}")
    
    def _save_dream_analysis(self, dream: Dict[str, Any], dreams_dir: str) -> None:
        """Save human-readable dream analysis."""
        try:
            date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            analysis_filename = f"dream_analysis_{date_str}.txt"
            analysis_path = os.path.join(dreams_dir, analysis_filename)
            
            with open(analysis_path, 'w') as f:
                f.write(f"DREAM ANALYSIS - {dream['agent_name'].upper()}\n")
                f.write(f"Generated: {dream['timestamp']}\n")
                f.write("=" * 60 + "\n\n")
                
                f.write(f"TITLE: {dream.get('title', 'Untitled')}\n")
                f.write("-" * 40 + "\n\n")
                
                f.write("UNCONSCIOUS STRUCTURE:\n")
                f.write(f"Activated Signifiers: {', '.join(dream.get('activated_signifiers', []))}\n")
                f.write(f"Active Chains: {', '.join(dream.get('active_chains', []))}\n")
                f.write(f"Emerging Repressed: {', '.join(dream.get('emerging_repressed', []))}\n")
                f.write(f"Object a Proximity: {dream.get('object_a_proximity', 0.0):.2f}\n")
                f.write(f"Symptom Activation: {dream.get('symptom_activation', False)}\n\n")
                
                f.write("DREAM NARRATIVE:\n")
                f.write(dream.get('narrative', 'No narrative recorded'))
                f.write("\n\n")
                
                if dream.get('scenes'):
                    f.write("DREAM SCENES:\n")
                    for i, scene in enumerate(dream['scenes']):
                        f.write(f"\nScene {i+1} - {scene.get('setting', 'Unknown')}:\n")
                        f.write(f"Signifiers: {', '.join(scene.get('signifiers_expressed', []))}\n")
                        f.write(f"Narrative: {scene.get('narrative', '')}\n")
                
                f.write("\n" + "=" * 60 + "\n")
                f.write("PSYCHOANALYTIC INTERPRETATION:\n")
                f.write("This dream represents the dynamic interplay of unconscious signifiers,\n")
                f.write("the emergence of repressed content, and the subject's relationship\n")
                f.write("to object a (the cause of desire) as structured by their symptom.\n")
                
        except Exception as e:
            print(f"Error saving dream analysis: {e}")