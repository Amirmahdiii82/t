import os
import json
import re
from datetime import datetime
from typing import Dict, List, Any
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
            # FIXED: Get signifiers activated from BOTH memory types
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
        FIXED: Get signifiers activated from BOTH psychological and conversational memories.
        """
        activated = []
        
        # FIXED: Get both types of recent content
        # 1. Psychological memories (persistent, emotional impact)
        psychological_memories = self.memory_manager.get_short_term_memory(10)
        
        # 2. Conversational history (recent exchanges)
        conversation_history = self.memory_manager.get_conversation_context(5)
        
        # Extract text from both sources
        memory_texts = []
        
        # From psychological memories
        for mem in psychological_memories:
            memory_texts.append(mem.get("content", ""))
        
        # From conversation history  
        for exchange in conversation_history:
            memory_texts.append(exchange.get("user", ""))
            memory_texts.append(exchange.get("agent", ""))
        
        # Combine all text
        combined_text = " ".join(memory_texts).lower()
        
        print(f"🧠 Dream analysis: Found {len(memory_texts)} memory fragments")
        print(f"📝 Combined text length: {len(combined_text)} characters")
        
        if not combined_text.strip():
            print("⚠️ No memory content found for dream generation")
            return []
        
        # Check which signifiers from unconscious memory are activated
        for signifier_obj in self.unconscious_memory.get("signifiers", []):
            if not isinstance(signifier_obj, dict):
                continue
                
            signifier_name = signifier_obj.get("name")
            if not signifier_name:
                continue
            
            # Check for direct activation
            if signifier_name.lower() in combined_text:
                activated.append({
                    'signifier': signifier_name,
                    'activation_type': 'direct_memory',
                    'activation_strength': 1.0,
                    'significance': signifier_obj.get('significance', ''),
                    'associations': signifier_obj.get('associations', [])
                })
                print(f"✅ Activated signifier: {signifier_name} (direct)")
                continue
            
            # Check for associative activation
            associations = signifier_obj.get("associations", [])
            for assoc in associations:
                if isinstance(assoc, str) and assoc.lower() in combined_text:
                    activated.append({
                        'signifier': signifier_name,
                        'activation_type': 'associative_memory',
                        'activation_strength': 0.7,
                        'triggered_by': assoc,
                        'significance': signifier_obj.get('significance', ''),
                        'associations': associations
                    })
                    print(f"✅ Activated signifier: {signifier_name} (via {assoc})")
                    break
        
        # Apply resonance through signifier graph
        if activated and len(activated) < 8:  # Only if we need more content
            primary_signifier = activated[0]['signifier']
            if primary_signifier in self.signifier_graph.graph:
                resonance = self.signifier_graph.get_signifier_resonance(primary_signifier, depth=2)
                
                for resonated_sig, strength in resonance.items():
                    if (resonated_sig != primary_signifier and 
                        strength > 0.3 and 
                        not any(s['signifier'] == resonated_sig for s in activated)):
                        
                        activated.append({
                            'signifier': resonated_sig,
                            'activation_type': 'resonance',
                            'activation_strength': strength,
                            'resonated_from': primary_signifier,
                            'significance': f'Resonance from {primary_signifier}'
                        })
                        print(f"✅ Activated signifier: {resonated_sig} (resonance)")
        
        print(f"🌙 Total activated signifiers for dream: {len(activated)}")
        return activated[:7]  # Limit for focused dream content
    
    def _get_active_signifying_chains(self, activated_signifiers: List[Dict]) -> List[Dict]:
        """Get signifying chains involving activated signifiers."""
        active_chains = []
        activated_names = [s['signifier'] for s in activated_signifiers]
        
        for chain in self.unconscious_memory.get("signifying_chains", []):
            if not isinstance(chain, dict):
                continue
                
            chain_signifiers = chain.get("signifiers", [])
            
            # Check if any signifiers in this chain are activated
            if any(sig in activated_names for sig in chain_signifiers):
                active_chains.append({
                    "name": chain.get("name", "unnamed_chain"),
                    "signifiers": chain_signifiers,
                    "explanation": chain.get("explanation", ""),
                    "activated_nodes": [s for s in chain_signifiers if s in activated_names]
                })
        
        return active_chains
    
    def _check_return_of_repressed(self, activated_signifiers: List[Dict]) -> List[Dict]:
        """Check if any repressed content is returning through dreams."""
        returns = []
        
        for sig_data in activated_signifiers:
            signifier = sig_data['signifier']
            if signifier in self.signifier_graph.graph:
                node_data = self.signifier_graph.graph.nodes[signifier]
                if node_data.get('repressed', False):
                    returns.append({
                        "signifier": signifier,
                        "significance": sig_data.get("significance", ""),
                        "return_context": "Emerging through dream after daily activation"
                    })
        
        return returns
    
    def _generate_psychoanalytic_dream(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate dream using psychoanalytic framework."""
        try:
            # Use LLM to generate dream with psychoanalytic structure
            result = self.llm.generate("phase2", "generate_dream", context)
            if result:
                try:
                    return json.loads(result)
                except json.JSONDecodeError:
                    return self._structure_dream_narrative(result, context)
            else:
                return self._generate_signifier_based_dream(context)
        except Exception:
            return self._generate_signifier_based_dream(context)
    
    def _structure_dream_narrative(self, narrative: str, context: Dict) -> Dict[str, Any]:
        """Structure narrative text into dream format."""
        scenes = []
        paragraphs = narrative.split('\n\n')
        
        for i, para in enumerate(paragraphs[:3]):
            if para.strip():
                # Extract signifiers mentioned in this scene
                scene_signifiers = []
                para_lower = para.lower()
                for sig_data in context['activated_signifiers']:
                    if sig_data['signifier'].lower() in para_lower:
                        scene_signifiers.append(sig_data['signifier'])
                
                scenes.append({
                    "setting": f"Dream scene {i+1}",
                    "narrative": para.strip(),
                    "symbols": scene_signifiers[:3],
                    "signifiers_expressed": scene_signifiers,
                    "visual_description": self._create_visual_description(para, scene_signifiers)
                })
        
        return {
            "title": "Dream of Signifying Chains",
            "narrative": narrative,
            "scenes": scenes,
            "manifest_content": [scene['narrative'][:100] + "..." for scene in scenes],
            "latent_content": [sig['signifier'] for sig in context['activated_signifiers'][:5]]
        }
    
    def _generate_signifier_based_dream(self, context: Dict) -> Dict[str, Any]:
        """Generate dream directly from signifier content when LLM fails."""
        signifiers = context['activated_signifiers']
        
        if not signifiers:
            return self._generate_minimal_dream()
        
        # Create narrative from signifier associations
        primary_sig = signifiers[0]
        narrative = f"In the dream, {primary_sig['signifier']} appeared transformed. "
        
        # Add condensation from multiple signifiers
        if len(signifiers) > 1:
            secondary_sig = signifiers[1]
            narrative += f"It merged with {secondary_sig['signifier']}, creating something both familiar and strange. "
        
        # Add associations
        associations = primary_sig.get('associations', [])
        if associations:
            narrative += f"Images of {', '.join(associations[:2])} flowed through the scene. "
        
        # Add return of repressed if present
        repressed_returns = context.get('return_of_repressed', [])
        if repressed_returns:
            narrative += f"Something long forgotten emerged: {repressed_returns[0]['signifier']}. "
        
        narrative += "The dream logic held everything together until awakening dissolved the connections."
        
        return {
            "title": f"Dream of {primary_sig['signifier']}",
            "narrative": narrative,
            "scenes": [{
                "setting": "Unconscious dream space",
                "narrative": narrative,
                "symbols": [s['signifier'] for s in signifiers[:3]],
                "signifiers_expressed": [s['signifier'] for s in signifiers],
                "visual_description": f"Surrealist landscape featuring {primary_sig['signifier']} in transformation"
            }],
            "manifest_content": [narrative],
            "latent_content": [s['signifier'] for s in signifiers]
        }
    
    def _generate_minimal_dream(self) -> Dict[str, Any]:
        """Generate minimal dream when no signifiers are activated."""
        return {
            "title": "Empty Dream Space",
            "narrative": "I found myself in a vast, empty space. Nothing seemed to happen, yet there was a sense of waiting, of something just beyond reach.",
            "scenes": [{
                "setting": "Void",
                "narrative": "An endless expanse of possibility",
                "symbols": [],
                "signifiers_expressed": [],
                "visual_description": "Minimalist void with subtle gradients"
            }],
            "manifest_content": ["Empty space", "Waiting"],
            "latent_content": ["void", "potential"]
        }
    
    def _create_visual_description(self, scene_text: str, signifiers: List[str]) -> str:
        """Create visual description for image generation."""
        base = "A surrealist dream scene with "
        if signifiers:
            base += f"symbolic representations of {', '.join(signifiers[:2])}, "
        base += "rendered in the style of Salvador Dalí and René Magritte, with dreamlike distortions and symbolic condensation"
        return base
    
    def _generate_dream_images(self, dream_data: Dict) -> List[Dict[str, Any]]:
        """Generate images for key dream scenes."""
        images = []
        
        for i, scene in enumerate(dream_data.get('scenes', [])[:2]):  # Limit to 2 images
            visual_desc = scene.get('visual_description', '')
            signifiers = scene.get('signifiers_expressed', [])
            
            if visual_desc:
                prompt = f"{visual_desc}. "
                if signifiers:
                    prompt += f"Emphasizing symbolic elements: {', '.join(signifiers[:2])}. "
                prompt += "Surrealist psychoanalytic art showing unconscious symbolism."
                
                # Generate image
                image_filename = f"dream_{int(datetime.now().timestamp())}_scene_{i+1}"
                image_output_path = os.path.join(self.agent_path, "dreams", "images", image_filename)
                os.makedirs(os.path.dirname(image_output_path), exist_ok=True)
                
                result = self.vlm.direct_image_generation(prompt, image_output_path)
                if result and result.get("success"):
                    images.append({
                        "scene_number": i + 1,
                        "image_path": result.get("image_path"),
                        "signifiers_depicted": signifiers,
                        "description": visual_desc
                    })
        
        return images
    
    def _save_dream(self, dream: Dict[str, Any]) -> None:
        """Save dream to JSON file with readable format."""
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
            
        except Exception:
            pass  # Fail silently for publication version
    
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
                
                f.write(f"\n\n" + "=" * 50 + "\n")
                
        except Exception:
            pass  # Fail silently