import os
import json
import random
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from interfaces.llm_interface import LLMInterface
from interfaces.vlm_interface import VLMInterface

class DreamGenerator:
    """Generate dreams using Freudian dream-work mechanisms and Lacanian symbolic logic."""
    
    def __init__(self, agent_name: str, memory_manager, base_path: str = "base_agents"):
        self.agent_name = agent_name
        self.memory_manager = memory_manager
        self.base_path = base_path
        self.agent_path = os.path.join(base_path, agent_name)
        
        self.llm = LLMInterface()
        self.vlm = VLMInterface()
        
        # Load unconscious structures
        self._load_unconscious_structures()
        
        print(f"Dream Generator initialized for {agent_name}")
    
    def _load_unconscious_structures(self):
        """Load unconscious structures for dream generation."""
        try:
            unconscious = self.memory_manager.unconscious_memory
            self.signifiers = unconscious.get('signifiers', [])
            self.chains = unconscious.get('signifying_chains', [])
            self.object_a = unconscious.get('object_a', {})
            self.symptom = unconscious.get('symptom', {})
            self.dream_work_patterns = unconscious.get('dream_work_patterns', {})
        except:
            self.signifiers = []
            self.chains = []
            self.object_a = {}
            self.symptom = {}
            self.dream_work_patterns = {}
    
    def generate_dream(self, dream_context: str = "sleep") -> Dict[str, Any]:
        """Generate a dream using proper psychoanalytic dream-work."""
        print(f"Generating dream for {self.agent_name}...")
        
        try:
            # 1. Gather day residue (recent memories)
            day_residue = self._gather_day_residue()
            
            # 2. Identify latent dream thoughts
            latent_thoughts = self._identify_latent_thoughts(day_residue)
            
            # 3. Apply dream-work mechanisms
            dream_work = self._apply_dream_work(latent_thoughts)
            
            # 4. Generate manifest dream content
            manifest_dream = self._generate_manifest_content(dream_work)
            
            # 5. Apply secondary revision
            final_dream = self._apply_secondary_revision(manifest_dream)
            
            # 6. Generate dream imagery
            dream_images = self._generate_dream_imagery(final_dream)
            
            # 7. Analyze dream for interpretation
            dream_analysis = self._analyze_dream(final_dream, latent_thoughts)
            
            # Create complete dream object
            dream = {
                "id": f"dream_{int(datetime.now().timestamp())}",
                "agent_name": self.agent_name,
                "timestamp": datetime.now().isoformat(),
                "context": dream_context,
                "day_residue": day_residue,
                "latent_thoughts": latent_thoughts,
                "dream_work": dream_work,
                "manifest_content": final_dream,
                "images": dream_images,
                "analysis": dream_analysis,
                "emotional_state": self.memory_manager.get_emotional_state()
            }
            
            # Save dream
            self._save_dream(dream)
            
            return dream
            
        except Exception as e:
            print(f"Error generating dream: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _gather_day_residue(self) -> List[Dict[str, Any]]:
        """Gather recent experiences that will form day residue."""
        day_residue = []
        
        # Get recent short-term memories
        recent_memories = self.memory_manager.get_short_term_memory(10)
        
        for memory in recent_memories:
            # Extract elements that could become day residue
            residue = {
                'content': memory.get('content', ''),
                'context': memory.get('context', ''),
                'emotional_charge': memory.get('emotional_state', {}).get('emotion_category', 'neutral'),
                'timestamp': memory.get('timestamp', ''),
                'signifier_triggers': []
            }
            
            # Check which signifiers this memory might trigger
            content_lower = residue['content'].lower()
            for signifier in self.signifiers[:20]:  # Check top signifiers
                if isinstance(signifier, dict):
                    sig_name = signifier.get('name', '').lower()
                    if sig_name in content_lower:
                        residue['signifier_triggers'].append(sig_name)
            
            if residue['signifier_triggers'] or residue['emotional_charge'] != 'neutral':
                day_residue.append(residue)
        
        return day_residue
    
    def _identify_latent_thoughts(self, day_residue: List[Dict]) -> Dict[str, Any]:
        """Identify latent dream thoughts from day residue and unconscious."""
        prompt = f"""
        Identify latent dream thoughts from these elements:
        
        Day residue: {json.dumps(day_residue, indent=2)}
        
        Unconscious signifiers: {json.dumps(self.signifiers[:10], indent=2)}
        Object a: {json.dumps(self.object_a, indent=2)}
        Symptom: {json.dumps(self.symptom, indent=2)}
        
        Following Freud's model, identify:
        1. Repressed wishes trying to emerge
        2. Infantile wishes connected to current experiences  
        3. Forbidden desires seeking expression
        4. Anxieties requiring dream-work
        5. Connections between day residue and unconscious material
        
        Return structured latent thoughts.
        """
        
        response = self.llm.generate("phase2", "identify_latent_thoughts", {
            "day_residue": json.dumps(day_residue, indent=2),
            "signifiers": json.dumps(self.signifiers[:10], indent=2),
            "object_a": json.dumps(self.object_a, indent=2),
            "symptom": json.dumps(self.symptom, indent=2)
        })
        
        try:
            latent = json.loads(response)
        except:
            # Structure the response if not JSON
            latent = {
                'wishes': ['unconscious wish for recognition', 'desire for maternal comfort'],
                'anxieties': ['fear of abandonment', 'castration anxiety'],
                'forbidden_content': ['aggressive impulses', 'incestuous desires'],
                'connections': day_residue
            }
        
        return latent
    
    def _apply_dream_work(self, latent_thoughts: Dict) -> Dict[str, Any]:
        """Apply Freudian dream-work mechanisms."""
        dream_work = {
            'condensation': [],
            'displacement': [],
            'representability': [],
            'symbolization': []
        }
        
        # 1. Condensation (Verdichtung)
        dream_work['condensation'] = self._apply_condensation(latent_thoughts)
        
        # 2. Displacement (Verschiebung)
        dream_work['displacement'] = self._apply_displacement(latent_thoughts)
        
        # 3. Considerations of Representability
        dream_work['representability'] = self._apply_representability(latent_thoughts)
        
        # 4. Symbolization
        dream_work['symbolization'] = self._apply_symbolization(latent_thoughts)
        
        return dream_work
    
    def _apply_condensation(self, latent_thoughts: Dict) -> List[Dict]:
        """Apply condensation - multiple ideas in single image."""
        condensations = []
        
        # Combine related signifiers
        wishes = latent_thoughts.get('wishes', [])
        anxieties = latent_thoughts.get('anxieties', [])
        
        # Create condensed images
        if len(wishes) > 1:
            condensations.append({
                'type': 'wish_condensation',
                'elements': wishes[:3],
                'condensed_image': f"A figure that is both {wishes[0]} and {wishes[1]}",
                'overdetermination': len(wishes)
            })
        
        # Condense people/figures
        connections = latent_thoughts.get('connections', [])
        if len(connections) > 1:
            figures = [c.get('content', '') for c in connections if 'person' in c.get('content', '').lower()]
            if len(figures) > 1:
                condensations.append({
                    'type': 'person_condensation',
                    'elements': figures,
                    'condensed_image': "A composite figure with features of multiple people",
                    'interpretation': 'Multiple relationships condensed into one'
                })
        
        # Condense signifying chains
        for chain in self.chains[:3]:
            if isinstance(chain, dict):
                signifiers = chain.get('signifiers', [])
                if len(signifiers) > 2:
                    condensations.append({
                        'type': 'chain_condensation',
                        'chain': chain.get('name', ''),
                        'condensed_signifiers': signifiers[:3],
                        'dream_element': f"An object that represents {', '.join(signifiers[:3])}"
                    })
        
        return condensations
    
    def _apply_displacement(self, latent_thoughts: Dict) -> List[Dict]:
        """Apply displacement - shift emphasis and affect."""
        displacements = []
        
        # Displace affect from forbidden to acceptable
        forbidden = latent_thoughts.get('forbidden_content', [])
        for content in forbidden:
            # Find acceptable substitute
            substitute = self._find_displacement_substitute(content)
            displacements.append({
                'original': content,
                'displaced_to': substitute,
                'mechanism': 'affect_transfer',
                'interpretation': f"Intense feeling about {content} displaced onto {substitute}"
            })
        
        # Displace from important to trivial
        anxieties = latent_thoughts.get('anxieties', [])
        for anxiety in anxieties[:2]:
            trivial_element = self._generate_trivial_element()
            displacements.append({
                'original': anxiety,
                'displaced_to': trivial_element,
                'mechanism': 'significance_reversal',
                'interpretation': f"Core anxiety about {anxiety} appears as concern about {trivial_element}"
            })
        
        return displacements
    
    def _apply_representability(self, latent_thoughts: Dict) -> List[Dict]:
        """Convert abstract thoughts to concrete visual representations."""
        representations = []
        
        # Convert abstract concepts to visual scenes
        abstract_concepts = []
        
        # Extract abstract elements from wishes and anxieties
        for wish in latent_thoughts.get('wishes', []):
            if any(word in str(wish).lower() for word in ['love', 'freedom', 'power', 'death']):
                abstract_concepts.append(wish)
        
        for concept in abstract_concepts:
            visual = self._convert_to_visual(concept)
            representations.append({
                'abstract_thought': concept,
                'visual_representation': visual,
                'mechanism': 'concretization'
            })
        
        # Convert relationships to spatial arrangements
        connections = latent_thoughts.get('connections', [])
        if connections:
            representations.append({
                'abstract_thought': 'relationship dynamics',
                'visual_representation': 'People positioned at various distances in a room',
                'mechanism': 'spatial_metaphor'
            })
        
        return representations
    
    def _apply_symbolization(self, latent_thoughts: Dict) -> List[Dict]:
        """Apply symbolic transformations."""
        symbols = []
        
        # Standard Freudian symbols
        symbol_mappings = {
            'sexual': ['snake', 'tower', 'tunnel', 'flower'],
            'death': ['darkness', 'journey', 'farewell', 'winter'],
            'birth': ['water', 'emergence', 'eggs', 'dawn'],
            'castration': ['cutting', 'loss of teeth', 'broken objects'],
            'mother': ['house', 'earth', 'ocean', 'container']
        }
        
        # Apply symbolization based on latent content
        for category, symbols_list in symbol_mappings.items():
            if any(category in str(thought).lower() for thought in latent_thoughts.get('wishes', [])):
                selected_symbol = random.choice(symbols_list)
                symbols.append({
                    'latent_content': category,
                    'symbol': selected_symbol,
                    'appearance': f"A prominent {selected_symbol} appears in the dream"
                })
        
        # Personal symbols from signifiers
        for signifier in self.signifiers[:5]:
            if isinstance(signifier, dict) and signifier.get('associations'):
                symbols.append({
                    'latent_content': signifier.get('name', ''),
                    'symbol': random.choice(signifier['associations']),
                    'appearance': f"{signifier['name']} appears as {random.choice(signifier['associations'])}"
                })
        
        return symbols
    
    def _generate_manifest_content(self, dream_work: Dict) -> Dict[str, Any]:
        """Generate the manifest dream content from dream-work."""
        prompt = f"""
        Create a dream narrative using these dream-work elements:
        
        Condensations: {json.dumps(dream_work['condensation'], indent=2)}
        Displacements: {json.dumps(dream_work['displacement'], indent=2)}
        Visual representations: {json.dumps(dream_work['representability'], indent=2)}
        Symbols: {json.dumps(dream_work['symbolization'], indent=2)}
        
        Create a surreal but coherent dream narrative that:
        1. Incorporates the condensed images naturally
        2. Shows displaced affects and emphasis
        3. Uses concrete visual scenes
        4. Includes the symbolic elements
        5. Maintains dream-like illogic and scene shifts
        6. Has 2-3 distinct scenes
        
        Format as JSON with scenes and narrative.
        """
        
                    response = self.llm.generate(None, prompt, None)
        
        try:
            manifest = json.loads(response)
        except:
            # Create structured manifest content
            manifest = self._create_default_manifest(dream_work)
        
        return manifest
    
    def _apply_secondary_revision(self, manifest_content: Dict) -> Dict[str, Any]:
        """Apply secondary revision to make dream more coherent."""
        prompt = f"""
        Apply secondary revision to this dream content to make it more story-like:
        
        {json.dumps(manifest_content, indent=2)}
        
        Add:
        1. Narrative connections between scenes
        2. Rational explanations for bizarre elements (dream logic)
        3. Temporal sequence
        4. Character continuity
        
        Keep the surreal quality but add enough coherence to be tellable.
        """
        
                    response = self.llm.generate(None, prompt, None)
        
        try:
            revised = json.loads(response)
            revised['revision_type'] = 'secondary'
        except:
            revised = manifest_content
            revised['revision_type'] = 'minimal'
        
        return revised
    
    def _generate_dream_imagery(self, dream_content: Dict) -> List[Dict[str, Any]]:
        """Generate surrealist images for key dream scenes."""
        images = []
        scenes = dream_content.get('scenes', [])
        
        for i, scene in enumerate(scenes[:3]):  # Limit to 3 images
            # Extract visual elements from scene
            visual_prompt = self._create_dream_image_prompt(scene, dream_content)
            
            image_filename = f"dream_{datetime.now().strftime('%Y%m%d_%H%M%S')}_scene_{i+1}"
            output_path = os.path.join(self.agent_path, "dreams", "images", image_filename)
            
            result = self.vlm.direct_image_generation(visual_prompt, output_path)
            
            if result.get("success"):
                images.append({
                    "scene_index": i,
                    "image_path": result["image_path"],
                    "prompt": visual_prompt,
                    "dream_elements": scene.get('key_elements', [])
                })
        
        return images
    
    def _analyze_dream(self, dream_content: Dict, latent_thoughts: Dict) -> Dict[str, Any]:
        """Provide psychoanalytic interpretation of the dream."""
        template_data = {
            "dream_content": json.dumps(dream_content, indent=2),
            "latent_thoughts": json.dumps(latent_thoughts, indent=2)
        }
        
        interpretation = self.llm.generate("phase2", "analyze_dream", template_data)
        
        return {
            "interpretation": interpretation,
            "wish_fulfillment": "The dream fulfills repressed wishes through symbolic satisfaction",
            "anxiety_management": "Dream-work transforms anxiety into manageable images",
            "desire_structure": "The dream reveals how desire circulates around object a"
        }
    
    def _find_displacement_substitute(self, forbidden_content: str) -> str:
        """Find acceptable substitute for forbidden content."""
        substitutes = {
            'aggressive': ['competitive sports', 'breaking objects', 'loud arguments'],
            'sexual': ['dancing', 'eating', 'swimming'],
            'death': ['travel', 'transformation', 'sleep'],
            'incest': ['close friendship', 'mentorship', 'admiration']
        }
        
        for key, subs in substitutes.items():
            if key in forbidden_content.lower():
                return random.choice(subs)
        
        return 'mundane daily activity'
    
    def _generate_trivial_element(self) -> str:
        """Generate trivial element for displacement."""
        trivial_elements = [
            'a missing button',
            'a crooked picture frame',
            'an untied shoelace',
            'a spelling error',
            'a misplaced book',
            'a wrong turn',
            'a forgotten name',
            'a stain on clothing'
        ]
        return random.choice(trivial_elements)
    
    def _convert_to_visual(self, abstract_concept: str) -> str:
        """Convert abstract concept to visual representation."""
        visual_mappings = {
            'love': 'two trees with intertwined branches',
            'freedom': 'birds flying from an open cage',
            'power': 'a towering figure casting long shadows',
            'death': 'a wilting flower in a vast field',
            'anxiety': 'a maze with no visible exit',
            'desire': 'a person reaching for a distant light'
        }
        
        concept_lower = str(abstract_concept).lower()
        for key, visual in visual_mappings.items():
            if key in concept_lower:
                return visual
        
        return 'an abstract geometric shape pulsing with energy'
    
    def _create_default_manifest(self, dream_work: Dict) -> Dict[str, Any]:
        """Create default manifest content if parsing fails."""
        scenes = []
        
        # Create scene from condensations
        if dream_work['condensation']:
            condensation = dream_work['condensation'][0]
            scenes.append({
                'setting': 'A fluid, shifting space',
                'narrative': f"I see {condensation.get('condensed_image', 'a composite figure')}",
                'key_elements': condensation.get('elements', []),
                'emotional_tone': 'uncanny'
            })
        
        # Create scene from displacement
        if dream_work['displacement']:
            displacement = dream_work['displacement'][0]
            scenes.append({
                'setting': 'An ordinary location that feels significant',
                'narrative': f"I'm deeply concerned about {displacement.get('displaced_to', 'something trivial')}",
                'key_elements': [displacement.get('original', ''), displacement.get('displaced_to', '')],
                'emotional_tone': 'anxious'
            })
        
        return {
            'title': 'Dream of Hidden Desires',
            'scenes': scenes,
            'overall_narrative': 'A dream where meanings shift and nothing is quite what it seems'
        }
    
    def _create_dream_image_prompt(self, scene: Dict, full_dream: Dict) -> str:
        """Create prompt for dream image generation."""
        elements = scene.get('key_elements', [])
        setting = scene.get('setting', 'surreal dreamscape')
        tone = scene.get('emotional_tone', 'mysterious')
        
        prompt = f"""
        Create a surrealist dream image with these elements:
        
        Setting: {setting}
        Key elements: {', '.join(elements[:5])}
        Emotional tone: {tone}
        
        Style: Combine Salvador Dalí's melting reality with René Magritte's impossible objects and 
        Remedios Varo's mystical symbolism.
        
        The image should:
        - Feel like a genuine dream - illogical but emotionally coherent
        - Use visual condensation (multiple meanings in one image)
        - Include subtle distortions and impossibilities
        - Have rich symbolic content open to interpretation
        - Capture the uncanny feeling of dreams
        
        Make it detailed, dreamlike, and psychologically evocative.
        """
        
        return prompt
    
    def _save_dream(self, dream: Dict[str, Any]) -> None:
        """Save dream to file system."""
        try:
            dreams_dir = os.path.join(self.agent_path, "dreams")
            os.makedirs(dreams_dir, exist_ok=True)
            
            # Save dream data
            dream_path = os.path.join(dreams_dir, f"{dream['id']}.json")
            with open(dream_path, 'w') as f:
                json.dump(dream, f, indent=2)
            
            print(f"Dream saved to {dream_path}")
            
            # Update dream log
            log_path = os.path.join(dreams_dir, "dream_log.json")
            try:
                with open(log_path, 'r') as f:
                    log = json.load(f)
            except:
                log = []
            
            log.append({
                'id': dream['id'],
                'timestamp': dream['timestamp'],
                'summary': dream.get('manifest_content', {}).get('overall_narrative', 'A dream')
            })
            
            with open(log_path, 'w') as f:
                json.dump(log, f, indent=2)
                
        except Exception as e:
            print(f"Error saving dream: {e}")
    
    def get_recent_dreams(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent dreams with full psychoanalytic structure."""
        dreams = []
        dreams_dir = os.path.join(self.agent_path, "dreams")
        
        if os.path.exists(dreams_dir):
            dream_files = [f for f in os.listdir(dreams_dir) 
                          if f.endswith('.json') and f != 'dream_log.json']
            dream_files.sort(reverse=True)
            
            for dream_file in dream_files[:limit]:
                try:
                    with open(os.path.join(dreams_dir, dream_file), 'r') as f:
                        dreams.append(json.load(f))
                except:
                    continue
        
        return dreams