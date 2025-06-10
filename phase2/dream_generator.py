import os
import json
import random
import re
from datetime import datetime
from typing import Dict, List, Any
from interfaces.llm_interface import LLMInterface
from interfaces.vlm_interface import VLMInterface
from utils.lacanian_graph import LacanianSignifierGraph

class DreamGenerator:
    def __init__(self, agent_name: str, memory_manager, base_path: str = "base_agents"):
        """Initialize dream generator with psychoanalytic framework."""
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
        
        print(f"Dream Generator initialized for {agent_name}")
    
    def _load_unconscious_memory(self):
        """Load unconscious memory with signifiers and chains."""
        unconscious_path = os.path.join(self.agent_path, "unconscious_memory.json")
        try:
            with open(unconscious_path, 'r') as f:
                self.unconscious_memory = json.load(f)
            print(f"Loaded unconscious memory with {len(self.unconscious_memory.get('signifiers', []))} signifiers")
        except Exception as e:
            print(f"Error loading unconscious memory: {e}")
            self.unconscious_memory = {"signifiers": [], "signifying_chains": []}
    
    def _initialize_signifier_graph(self):
        """Initialize or load the signifier graph."""
        self.signifier_graph = LacanianSignifierGraph()
        
        # Reconstruct graph from saved data if available
        if self.unconscious_memory.get('signifier_graph'):
            graph_data = self.unconscious_memory['signifier_graph']
            
            # Add nodes
            for node in graph_data.get('nodes', []):
                self.signifier_graph.graph.add_node(
                    node['id'],
                    type=node.get('type', 'symbolic'),
                    activation=node.get('activation', 0.0),
                    repressed=node.get('repressed', False)
                )
            
            # Add edges
            for edge in graph_data.get('edges', []):
                self.signifier_graph.graph.add_edge(
                    edge['source'],
                    edge['target'],
                    weight=edge.get('weight', 0.5),
                    type=edge.get('type', 'neutral'),
                    context=edge.get('context', [])
                )
            
            print(f"Loaded signifier graph with {len(self.signifier_graph.graph.nodes())} nodes")
    
    def generate_dream(self, dream_context: str = "sleep") -> Dict[str, Any]:
        """Generate dream based on activated signifiers from daily interactions."""
        print(f"Generating dream for {self.agent_name} based on activated signifiers...")
        
        try:
            # Get activated signifiers from recent interactions
            activated_signifiers = self._get_activated_signifiers()
            
            # Get signifying chains involving activated signifiers
            active_chains = self._get_active_signifying_chains(activated_signifiers)
            
            # Check for return of the repressed
            repressed_returns = self._check_return_of_repressed(activated_signifiers)
            
            # Get dream context including object_a and symptom
            dream_context_data = self._gather_psychoanalytic_context(
                activated_signifiers, active_chains, repressed_returns
            )
            
            # Generate dream narrative using psychoanalytic framework
            dream_data = self._generate_psychoanalytic_dream(dream_context_data)
            
            # Generate dream imagery for key signifiers
            dream_images = self._generate_signifier_images(dream_data)
            
            # Analyze dream through Lacanian lens
            dream_analysis = self._analyze_dream_lacanian(dream_data, dream_context_data)
            
            # Create structured dream object
            dream = {
                "id": f"dream_{int(datetime.now().timestamp())}",
                "agent_name": self.agent_name,
                "timestamp": datetime.now().isoformat(),
                "mode": dream_context,
                "title": dream_data.get("title", "Untitled Dream"),
                "narrative": dream_data.get("narrative", ""),
                "scenes": dream_data.get("scenes", []),
                "activated_signifiers": activated_signifiers,
                "signifying_chains": active_chains,
                "return_of_repressed": repressed_returns,
                "central_fantasy": dream_data.get("central_fantasy", ""),
                "manifest_content": dream_data.get("manifest_content", []),
                "latent_content": dream_data.get("latent_content", []),
                "images": dream_images,
                "analysis": dream_analysis,
                "emotional_state": self.memory_manager.get_emotional_state(),
                "recent_interactions": self._get_recent_interaction_summary()
            }
            
            # Update signifier graph with dream activation patterns
            self._update_signifier_activation_from_dream(dream)
            
            # Save dream
            self._save_dream(dream)
            self._save_readable_dream(dream)
            
            print(f"Dream generated: {dream['id']}")
            return dream
            
        except Exception as e:
            print(f"Error generating dream: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _get_activated_signifiers(self) -> List[Dict[str, Any]]:
        """Get signifiers activated by recent interactions."""
        activated = []
        recent_memories = self.memory_manager.get_short_term_memory(10)
        
        if not recent_memories:
            print("❗CRITICAL DEBUG: No recent memories found by DreamGenerator. The list is empty.")
            return []

        recent_text = " ".join([mem.get("content", "") for mem in recent_memories])
        recent_text_lower = recent_text.lower()
        
        print("\n" + "="*25 + " DREAM SIGNIFIER DEBUG " + "="*25)
        print(f"Analyzing text: '{recent_text_lower}'")
        
        all_master_signifiers = self.unconscious_memory.get("signifiers", [])
        print(f"Searching for {len(all_master_signifiers)} master signifiers and their associations...")

        found_signifiers = False
        for signifier_obj in all_master_signifiers:
            parent_name = signifier_obj.get("name")
            if not parent_name:
                continue

            search_terms = [parent_name.lower()] + [assoc.lower() for assoc in signifier_obj.get("associations", [])]
            
            for term in search_terms:
                if re.search(r'\b' + re.escape(term) + r'\b', recent_text_lower):
                    found_signifiers = True
                    print(f"✅ MATCH FOUND: Text term '{term}' activated master signifier '{parent_name}'.")
                    
                    if not any(s['name'] == parent_name for s in activated):
                        activation = 0.5
                        if parent_name in self.signifier_graph.graph:
                            activation = self.signifier_graph.graph.nodes[parent_name].get('activation', 0.5)

                        activated.append({
                            "name": parent_name,
                            "significance": signifier_obj.get("significance"),
                            "associations": signifier_obj.get("associations"),
                            "activation_level": activation,
                            "relation_to_desire": signifier_obj.get("relation_to_desire")
                        })
                    break 
        
        if not found_signifiers:
            print("❌ NO MATCH: No direct signifiers or their associations were found in the recent interaction text.")

        if activated:
            start_node = max(activated, key=lambda x: x['activation_level'])['name']
            print(f"🧠 Spreading activation from most active node: '{start_node}'")
            
            # --- START OF THE FINAL, CORRECT FIX ---
            # This replaces the line that caused the crash. It calls the correct method
            # from your lacanian_graph.py file.
            try:
                spread_results = self.signifier_graph.get_signifier_resonance(start_node, depth=3)

                for node_id, activation_score in spread_results.items():
                    if node_id == start_node: continue # Skip the start node itself

                    if not any(s['name'] == node_id for s in activated):
                        # Use the resonance score as the activation level
                        original_signifier_data = next((s for s in self.unconscious_memory.get("signifiers", []) if s['name'] == node_id), None)
                        if not original_signifier_data:
                             # If the spread-to node isn't a master signifier, create a basic entry
                            original_signifier_data = {"name": node_id, "significance": "Activated via resonance", "associations": [], "relation_to_desire": ""}
                        
                        print(f"  -> Activated '{node_id}' via resonance (score: {activation_score:.2f}).")
                        activated.append({
                            "name": node_id, "significance": original_signifier_data.get("significance"),
                            "associations": original_signifier_data.get("associations"), "activation_level": activation_score,
                            "relation_to_desire": original_signifier_data.get("relation_to_desire", ""), "activation_type": "spread"
                        })
            except Exception as e:
                print(f"⚠️ Error during resonance spreading: {e}")
            # --- END OF THE FINAL, CORRECT FIX ---
        
        print(f"Total activated signifiers to be used in dream: {len(activated)}")
        print("="*75 + "\n")

        return activated[:7]
    
    def _get_active_signifying_chains(self, activated_signifiers: List[Dict]) -> List[Dict]:
        """Get signifying chains that involve activated signifiers."""
        active_chains = []
        activated_names = [s['name'].lower() for s in activated_signifiers]
        
        for chain in self.unconscious_memory.get("signifying_chains", []):
            chain_signifiers = [s.lower() for s in chain.get("signifiers", [])]
            
            if any(sig in chain_signifiers for sig in activated_names):
                active_chains.append({
                    "name": chain.get("name"), "signifiers": chain.get("signifiers"),
                    "explanation": chain.get("explanation"), "relation_to_fantasy": chain.get("relation_to_fantasy"),
                    "activated_nodes": [s for s in chain.get("signifiers", []) if s.lower() in activated_names]
                })
        
        return active_chains
    
    def _check_return_of_repressed(self, activated_signifiers: List[Dict]) -> List[Dict]:
        """Check for return of repressed signifiers."""
        repressed_returns = []
        
        for signifier in activated_signifiers:
            sig_name = signifier['name']
            if sig_name in self.signifier_graph.graph:
                if self.signifier_graph.graph.nodes[sig_name].get('repressed', False):
                    repressed_returns.append({
                        "signifier": sig_name, "significance": signifier.get("significance"),
                        "activation_level": signifier.get("activation_level"),
                        "return_context": "Appeared in dream after being activated in daily interaction"
                    })
        
        return repressed_returns
    
    def _gather_psychoanalytic_context(self, activated_signifiers, active_chains, repressed_returns):
        """Gather full psychoanalytic context for dream generation."""
        return {
            "agent_name": self.agent_name, "activated_signifiers": activated_signifiers,
            "signifying_chains": active_chains, "return_of_repressed": repressed_returns,
            "object_a": self.unconscious_memory.get("object_a", {}), "symptom": self.unconscious_memory.get("symptom", {}),
            "structural_positions": self.unconscious_memory.get("structural_positions", []),
            "recent_memories": self.memory_manager.get_short_term_memory(5),
            "emotional_state": self.memory_manager.get_emotional_state(),
            "unconscious_chains": self.unconscious_memory.get("signifying_chains", [])
        }
    
    def _generate_psychoanalytic_dream(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate dream using psychoanalytic framework."""
        try:
            result = self.llm.generate("phase2", "generate_dream", context)
            if result:
                try:
                    dream_data = json.loads(result)
                    return dream_data
                except json.JSONDecodeError:
                    return self._structure_dream_narrative(result, context)
            else:
                return self._generate_signifier_based_fallback(context)
        except Exception as e:
            print(f"Error in psychoanalytic dream generation: {e}")
            return self._generate_signifier_based_fallback(context)
    
    def _structure_dream_narrative(self, narrative: str, context: Dict) -> Dict[str, Any]:
        """Structure a text narrative into dream format."""
        scenes = []
        paragraphs = narrative.split('\n\n')
        
        for i, para in enumerate(paragraphs[:3]):
            if para.strip():
                scene_signifiers = []
                para_lower = para.lower()
                for sig in context['activated_signifiers']:
                    if sig['name'].lower() in para_lower:
                        scene_signifiers.append(sig['name'])
                
                scenes.append({
                    "setting": f"Dream space {i+1}", "narrative": para.strip(), "symbols": scene_signifiers[:3],
                    "signifiers_expressed": scene_signifiers, "visual_description": self._create_visual_description(para, scene_signifiers)
                })
        
        return {
            "title": "Dream of Signifying Chains", "narrative": narrative, "scenes": scenes,
            "central_fantasy": "The subject's relationship to desire and the Other",
            "manifest_content": [scene['narrative'][:100] + "..." for scene in scenes],
            "latent_content": [sig['name'] for sig in context['activated_signifiers'][:5]]
        }
    
    def _create_visual_description(self, scene_text: str, signifiers: List[str]) -> str:
        """Create visual description for image generation."""
        base = "A surrealist dream scene with "
        if signifiers:
            base += f"symbolic representations of {', '.join(signifiers[:2])}, "
        base += "rendered in the style of Dali and Magritte, with impossible geometries and symbolic condensation"
        return base
    
    def _generate_signifier_based_fallback(self, context: Dict) -> Dict[str, Any]:
        """Generate fallback dream based on signifiers."""
        signifiers = context['activated_signifiers']
        
        if not signifiers:
            narrative = "I found myself in an empty space, searching for something I couldn't name."
        else:
            sig1 = signifiers[0]['name']
            narrative = f"I dreamed of {sig1}, but it kept transforming. "
            if len(signifiers) > 1:
                sig2 = signifiers[1]['name']
                narrative += f"Every time I reached for it, it became {sig2}. "
            if context.get('return_of_repressed'):
                narrative += "Something long forgotten suddenly appeared, filling me with an inexplicable feeling. "
            narrative += "The dream logic made perfect sense until I woke, leaving only fragments of meaning."
        
        return {
            "title": "Dream of Transforming Signifiers", "narrative": narrative,
            "scenes": [{"setting": "Undefined dream space", "narrative": narrative, "symbols": [s['name'] for s in signifiers[:3]],
                        "signifiers_expressed": [s['name'] for s in signifiers], "visual_description": "Surrealist landscape with morphing symbolic objects"}],
            "central_fantasy": "The impossibility of grasping the object of desire",
            "manifest_content": [narrative], "latent_content": [s['name'] for s in signifiers]
        }
    
    def _generate_signifier_images(self, dream_data: Dict) -> List[Dict[str, Any]]:
        """Generate images for dream scenes focusing on signifiers."""
        images = []
        
        for i, scene in enumerate(dream_data.get('scenes', [])[:2]):
            visual_desc = scene.get('visual_description', '')
            signifiers = scene.get('signifiers_expressed', [])
            
            if visual_desc:
                prompt = f"{visual_desc}. "
                if signifiers:
                    prompt += f"Emphasizing the unconscious signifiers: {', '.join(signifiers[:2])}. "
                prompt += "In the style of surrealist psychoanalytic art, showing condensation and displacement."
                image_filename = f"dream_{int(datetime.now().timestamp())}_scene_{i+1}"
                image_output_path = os.path.join(self.agent_path, "dreams", "images", image_filename)
                os.makedirs(os.path.dirname(image_output_path), exist_ok=True)
                result = self.vlm.direct_image_generation(prompt, image_output_path)
                if result and result.get("success"):
                    images.append({
                        "scene_number": i + 1, "image_path": result.get("image_path"),
                        "signifiers_depicted": signifiers, "description": visual_desc
                    })
        return images
    
    def _analyze_dream_lacanian(self, dream_data: Dict, context: Dict) -> Dict[str, Any]:
        """Analyze dream through Lacanian framework."""
        analysis = {
            "primary_process": "Condensation and displacement of signifiers observed", "desire_manifestation": "",
            "other_relation": "", "jouissance_points": [], "symbolic_order_disruption": ""
        }
        if context['object_a']:
            analysis["desire_manifestation"] = f"The dream circles around {context['object_a'].get('description', 'the absent object')}, never quite reaching it"
        if any('father' in s['name'].lower() or 'mother' in s['name'].lower() for s in context['activated_signifiers']):
            analysis["other_relation"] = "Parental figures appear as representatives of the big Other"
        for scene in dream_data.get('scenes', []):
            if any(word in scene.get('narrative', '').lower() for word in ['anxiety', 'pleasure', 'fear', 'excitement']):
                analysis["jouissance_points"].append({"scene": scene.get('setting'), "manifestation": "Excessive affect indicating jouissance"})
        return analysis
    
    def _update_signifier_activation_from_dream(self, dream: Dict) -> None:
        """Update signifier graph based on dream activation."""
        for signifier in dream.get('activated_signifiers', []):
            sig_name = signifier['name']
            if sig_name in self.signifier_graph.graph:
                current = self.signifier_graph.graph.nodes[sig_name].get('activation', 0.5)
                self.signifier_graph.graph.nodes[sig_name]['activation'] = min(1.0, current + 0.1)
        
        signifier_names = [s['name'] for s in dream.get('activated_signifiers', [])]
        for i in range(len(signifier_names) - 1):
            if (signifier_names[i] in self.signifier_graph.graph and signifier_names[i+1] in self.signifier_graph.graph):
                self.signifier_graph.graph.add_edge(
                    signifier_names[i], signifier_names[i+1], weight=0.6,
                    type='dream_condensation', context=[f"Co-occurred in dream {dream['id']}"]
                )
    
    def _get_recent_interaction_summary(self) -> List[str]:
        """Get summary of recent interactions."""
        recent = self.memory_manager.get_short_term_memory(5)
        return [mem.get('content', '')[:100] + "..." for mem in recent if mem.get('content')]
    
    def _save_dream(self, dream: Dict[str, Any]) -> None:
        """Save dream to JSON file."""
        try:
            dreams_dir = os.path.join(self.agent_path, "dreams")
            os.makedirs(dreams_dir, exist_ok=True)
            dream_filename = f"{dream['id']}.json"
            dream_path = os.path.join(dreams_dir, dream_filename)
            with open(dream_path, 'w') as f:
                json.dump(dream, f, indent=2)
            print(f"Dream JSON saved to {dream_path}")
        except Exception as e:
            print(f"Error saving dream: {e}")
    
    def _save_readable_dream(self, dream: Dict[str, Any]) -> None:
        """Save human-readable version focusing on psychoanalytic elements."""
        try:
            dreams_dir = os.path.join(self.agent_path, "dreams")
            os.makedirs(dreams_dir, exist_ok=True)
            date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            readable_filename = f"dream_{date_str}_readable.txt"
            readable_path = os.path.join(dreams_dir, readable_filename)
            
            with open(readable_path, 'w') as f:
                f.write(f"PSYCHOANALYTIC DREAM RECORD - {dream['agent_name'].upper()}\n")
                f.write(f"Date: {dream['timestamp']}\n")
                f.write("=" * 70 + "\n\n")
                f.write(f"DREAM TITLE: {dream.get('title', 'Untitled')}\n")
                f.write("-" * 50 + "\n\n")
                f.write("ACTIVATED SIGNIFIERS:\n")
                for sig in dream.get('activated_signifiers', []):
                    f.write(f"- {sig['name']}: {sig['significance']}\n")
                    f.write(f"  Activation: {sig.get('activation_level', 0):.2f}")
                    if sig.get('activation_type') == 'spread':
                        f.write(" (via spread activation)")
                    f.write("\n")
                
                f.write("\n" + "-" * 50 + "\n")
                f.write("DREAM NARRATIVE:\n\n")
                f.write(dream.get('narrative', 'No narrative recorded'))
                f.write("\n\n" + "-" * 50 + "\n")
                
                if dream.get('scenes'):
                    f.write("DREAM SCENES:\n")
                    for i, scene in enumerate(dream['scenes']):
                        f.write(f"\nScene {i+1}: {scene.get('setting', 'Unknown setting')}\n")
                        f.write(scene.get('narrative', ''))
                        f.write(f"\nSignifiers present: {', '.join(scene.get('signifiers_expressed', []))}\n")
                
                if dream.get('signifying_chains'):
                    f.write("\n" + "-" * 50 + "\n")
                    f.write("ACTIVE SIGNIFYING CHAINS:\n")
                    for chain in dream['signifying_chains']:
                        f.write(f"\n{chain['name']}:\n")
                        f.write(f"  Chain: {' → '.join(chain['signifiers'])}\n")
                        f.write(f"  Activated nodes: {', '.join(chain.get('activated_nodes', []))}\n")
                
                if dream.get('return_of_repressed'):
                    f.write("\n" + "-" * 50 + "\n")
                    f.write("RETURN OF THE REPRESSED:\n")
                    for rep in dream['return_of_repressed']:
                        f.write(f"- {rep['signifier']}: {rep.get('return_context', '')}\n")
                
                f.write("\n" + "=" * 70 + "\n")
                f.write("LACANIAN ANALYSIS:\n")
                analysis = dream.get('analysis', {})
                f.write(f"Primary Process: {analysis.get('primary_process', 'Not analyzed')}\n")
                f.write(f"Desire Manifestation: {analysis.get('desire_manifestation', 'Not analyzed')}\n")
                f.write(f"Relation to Other: {analysis.get('other_relation', 'Not analyzed')}\n")
                
                if analysis.get('jouissance_points'):
                    f.write("\nJouissance Points:\n")
                    for jp in analysis['jouissance_points']:
                        f.write(f"- {jp.get('scene', '')}: {jp.get('manifestation', '')}\n")
                
            print(f"Readable dream saved to {readable_path}")
            
        except Exception as e:
            print(f"Error saving readable dream: {e}")
