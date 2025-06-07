import os
import json
import random
import re
import traceback
from datetime import datetime
from typing import Dict, List, Any
from interfaces.llm_interface import LLMInterface
from interfaces.vlm_interface import VLMInterface

class DreamGenerator:
    def __init__(self, agent_name: str, memory_manager, base_path: str = "base_agents"):
        """Initialize dream generator."""
        self.agent_name = agent_name
        self.memory_manager = memory_manager
        self.base_path = base_path
        self.agent_path = os.path.join(base_path, agent_name)
        
        # Initialize interfaces
        self.llm = LLMInterface()
        self.vlm = VLMInterface()
        
        # Dream generation settings
        self.dream_length_range = (100, 500)  # words
        self.max_dream_elements = 5
        
        print(f"Dream Generator initialized for {agent_name}")
    
    def generate_dream(self, dream_context: str = "sleep") -> Dict[str, Any]:
        """Generate a dream based on current memory state."""
        print(f"Generating dream for {self.agent_name}...")
        
        try:
            # Get memory context for dream generation
            memory_context = self._gather_dream_context()
            
            # Generate dream narrative
            dream_narrative = self._generate_dream_narrative(memory_context, dream_context)
            
            # Generate dream imagery
            dream_images = self._generate_dream_imagery(dream_narrative)
            
            # Create dream object
            dream = {
                "id": f"dream_{int(datetime.now().timestamp())}",
                "agent_name": self.agent_name,
                "timestamp": datetime.now().isoformat(),
                "context": dream_context,
                "narrative": dream_narrative,
                "images": dream_images,
                "memory_sources": memory_context,
                "emotional_state": self.memory_manager.get_emotional_state(),
                "unconscious_elements": self._extract_unconscious_elements(dream_narrative)
            }
            
            # Save dream
            self._save_dream(dream)
            
            print(f"Dream generated and saved: {dream['id']}")
            return dream
            
        except Exception as e:
            print(f"Error generating dream: {e}")
            traceback.print_exc()
            return None
    
    def _gather_dream_context(self) -> Dict[str, Any]:
        """Gather memory context for dream generation."""
        context = {
            "recent_memories": [],
            "emotional_memories": [],
            "relationships": [],
            "unconscious_signifiers": [],
            "current_emotional_state": None
        }
        
        try:
            # Get recent short-term memories
            short_term = self.memory_manager.get_short_term_memory(5)
            context["recent_memories"] = [mem["content"] for mem in short_term]
            
            # Get emotionally significant memories
            emotional_state = self.memory_manager.get_emotional_state()
            emotion_category = emotional_state.get("emotion_category", "neutral")
            
            # Search for memories related to current emotional state
            emotional_memories = self.memory_manager.retrieve_memories(emotion_category, 3)
            context["emotional_memories"] = emotional_memories
            
            # Get key relationships
            relationships = self.memory_manager.retrieve_relationships("important", 3)
            context["relationships"] = relationships
            
            # Get unconscious signifiers
            signifiers = self.memory_manager.get_unconscious_signifiers(5)
            context["unconscious_signifiers"] = signifiers
            
            # Current emotional state
            context["current_emotional_state"] = emotional_state
            
        except Exception as e:
            print(f"Error gathering dream context: {e}")
        
        return context
    
    def _generate_dream_narrative(self, memory_context: Dict[str, Any], dream_context: str) -> str:
        """Generate dream narrative using LLM."""
        try:
            # Prepare prompt data
            prompt_data = {
                "agent_name": self.agent_name,
                "dream_context": dream_context,
                "recent_memories": memory_context.get("recent_memories", []),
                "emotional_memories": memory_context.get("emotional_memories", []),
                "relationships": memory_context.get("relationships", []),
                "unconscious_signifiers": memory_context.get("unconscious_signifiers", []),
                "emotional_state": memory_context.get("current_emotional_state", {}),
                "dream_length": random.randint(*self.dream_length_range)
            }
            
            # Generate dream narrative
            narrative = self.llm.generate("phase2", "generate_dream", prompt_data)
            
            if narrative:
                return narrative
            else:
                return self._generate_fallback_dream(memory_context)
                
        except Exception as e:
            print(f"Error generating dream narrative: {e}")
            return self._generate_fallback_dream(memory_context)
    
    def _generate_fallback_dream(self, memory_context: Dict[str, Any]) -> str:
        """Generate a simple fallback dream."""
        elements = []
        
        # Add recent memory elements
        if memory_context.get("recent_memories"):
            elements.extend(memory_context["recent_memories"][:2])
        
        # Add relationship elements
        if memory_context.get("relationships"):
            for rel in memory_context["relationships"][:2]:
                if isinstance(rel, dict):
                    name = rel.get("name", "someone")
                    elements.append(f"interacting with {name}")
        
        # Add emotional elements
        emotional_state = memory_context.get("current_emotional_state", {})
        emotion = emotional_state.get("emotion_category", "neutral")
        elements.append(f"feeling {emotion}")
        
        # Combine elements into a simple narrative
        if elements:
            dream = f"I dreamed about {', '.join(elements[:3])}. "
            dream += f"The dream felt {emotion} and reflected my recent experiences."
        else:
            dream = f"I had a {emotion} dream that I can't quite remember clearly."
        
        return dream
    
    def _generate_dream_imagery(self, dream_narrative: str) -> List[Dict[str, Any]]:
        """Generate visual imagery for the dream using advanced element extraction."""
        images = []
        
        try:
            # Extract visual elements using advanced NLP techniques
            visual_elements = self._extract_visual_elements_advanced(dream_narrative)
            
            # Generate images for key elements
            for i, element in enumerate(visual_elements[:3]):  # Limit to 3 images
                try:
                    # Create more sophisticated image prompt
                    image_prompt = self._create_image_prompt(element, dream_narrative)
                    
                    # Generate image path
                    image_filename = f"dream_image_{int(datetime.now().timestamp())}_{i+1}"
                    image_output_path = os.path.join(self.agent_path, "signifier_images", image_filename)
                    
                    # Generate image using VLM
                    image_result = self.vlm.direct_image_generation(image_prompt, image_output_path)
                    
                    if image_result and image_result.get("success"):
                        images.append({
                            "id": f"dream_image_{i+1}",
                            "element": element,
                            "prompt": image_prompt,
                            "image_path": image_result.get("image_path"),
                            "thumbnail_path": image_result.get("thumbnail_path"),
                            "timestamp": datetime.now().isoformat()
                        })
                        print(f"Generated dream image for element: {element}")
                    else:
                        print(f"Failed to generate image for element: {element}")
                        
                except Exception as e:
                    print(f"Error generating image for element '{element}': {e}")
                    continue
        
        except Exception as e:
            print(f"Error generating dream imagery: {e}")
        
        return images
    
    def _extract_visual_elements_advanced(self, narrative: str) -> List[str]:
        """Extract visual elements from dream narrative using advanced techniques."""
        visual_elements = []
        
        try:
            # Use LLM to extract visual elements
            extraction_prompt = f"""
            Analyze this dream narrative and extract the 3-5 most vivid visual elements that would make compelling surrealist images:
            
            Dream: {narrative}
            
            Focus on:
            - Key objects, places, or scenes
            - Symbolic or metaphorical elements
            - Emotionally charged imagery
            - Surreal or impossible elements
            
            Return only a simple list of visual elements, one per line.
            """
            
            llm_response = self.llm.generate(None, extraction_prompt)
            
            if llm_response:
                # Parse the response to extract elements
                lines = llm_response.strip().split('\n')
                for line in lines:
                    line = line.strip()
                    # Remove bullet points, numbers, etc.
                    line = re.sub(r'^[-*•\d\.\)\s]+', '', line)
                    if line and len(line) > 3:
                        visual_elements.append(line)
                        if len(visual_elements) >= 5:
                            break
            
            # Fallback to regex-based extraction if LLM fails
            if not visual_elements:
                visual_elements = self._extract_visual_elements_regex(narrative)
            
        except Exception as e:
            print(f"Error in advanced visual element extraction: {e}")
            # Fallback to simple extraction
            visual_elements = self._extract_visual_elements_simple(narrative)
        
        return visual_elements[:5]
    
    def _extract_visual_elements_regex(self, narrative: str) -> List[str]:
        """Extract visual elements using regex patterns."""
        visual_elements = []
        
        # Patterns for visual elements
        patterns = [
            r'\b(flying|falling|running|walking|swimming|climbing)\b',
            r'\b(house|building|room|kitchen|bedroom|bathroom|garden|forest|ocean|mountain|city|street)\b',
            r'\b(car|plane|train|boat|bicycle|horse|animal|dog|cat|bird)\b',
            r'\b(fire|water|light|darkness|shadow|mirror|window|door|stairs|bridge)\b',
            r'\b(person|people|man|woman|child|family|friend|stranger)\b',
            r'\b(red|blue|green|yellow|black|white|bright|dark|colorful)\b.*?\b(object|thing|item)\b'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, narrative, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = ' '.join(match)
                if match and match not in visual_elements:
                    visual_elements.append(match)
                    if len(visual_elements) >= 5:
                        break
            if len(visual_elements) >= 5:
                break
        
        return visual_elements
    
    def _extract_visual_elements_simple(self, narrative: str) -> List[str]:
        """Simple fallback extraction method."""
        # Common visual dream elements
        common_elements = [
            "flying through clouds", "falling into darkness", "mysterious house",
            "endless corridor", "flowing water", "bright light", "shadowy figure",
            "familiar face", "strange landscape", "floating objects"
        ]
        
        narrative_lower = narrative.lower()
        found_elements = []
        
        for element in common_elements:
            if any(word in narrative_lower for word in element.split()):
                found_elements.append(element)
                if len(found_elements) >= 3:
                    break
        
        # If still no elements, use generic ones
        if not found_elements:
            found_elements = ["dream landscape", "symbolic imagery", "surreal scene"]
        
        return found_elements
    
    def _create_image_prompt(self, element: str, full_narrative: str) -> str:
        """Create a sophisticated image generation prompt."""
        # Extract emotional tone from narrative
        emotional_words = re.findall(r'\b(anxious|peaceful|frightening|joyful|sad|angry|confused|excited|calm|disturbing)\b', 
                                    full_narrative, re.IGNORECASE)
        emotional_tone = emotional_words[0] if emotional_words else "dreamlike"
        
        # Create sophisticated prompt
        prompt = f"""
        Create a surrealist dream image representing "{element}" with a {emotional_tone} atmosphere.
        
        Style: Salvador Dalí meets René Magritte - dreamlike, symbolic, with impossible elements and fluid reality.
        
        Visual elements:
        - {element} as the central focus
        - Dreamlike distortions and impossible perspectives
        - Rich symbolic imagery
        - {emotional_tone} mood and lighting
        - Surreal color palette
        - Floating or morphing elements
        
        The image should feel like it emerged from the unconscious mind, with symbolic weight and psychoanalytic depth.
        """
        
        return prompt
    
    def _extract_unconscious_elements(self, dream_narrative: str) -> List[Dict[str, Any]]:
        """Extract unconscious elements from dream narrative."""
        unconscious_elements = []
        
        try:
            # Get unconscious signifiers
            signifiers = self.memory_manager.get_unconscious_signifiers(10)
            
            # Check which signifiers appear in the dream
            narrative_lower = dream_narrative.lower()
            
            for signifier in signifiers:
                if isinstance(signifier, dict):
                    name = signifier.get("name", "").lower()
                    if name and name in narrative_lower:
                        unconscious_elements.append({
                            "signifier": signifier.get("name"),
                            "significance": signifier.get("significance", ""),
                            "associations": signifier.get("associations", [])
                        })
        
        except Exception as e:
            print(f"Error extracting unconscious elements: {e}")
        
        return unconscious_elements
    
    def _save_dream(self, dream: Dict[str, Any]) -> None:
        """Save dream to file."""
        try:
            # Create dreams directory if it doesn't exist
            dreams_dir = os.path.join(self.agent_path, "dreams")
            os.makedirs(dreams_dir, exist_ok=True)
            
            # Save dream
            dream_filename = f"{dream['id']}.json"
            dream_path = os.path.join(dreams_dir, dream_filename)
            
            with open(dream_path, 'w') as f:
                json.dump(dream, f, indent=2)
            
            print(f"Dream saved to {dream_path}")
            
        except Exception as e:
            print(f"Error saving dream: {e}")
    
    def get_recent_dreams(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent dreams."""
        dreams = []
        
        try:
            dreams_dir = os.path.join(self.agent_path, "dreams")
            
            if os.path.exists(dreams_dir):
                dream_files = [f for f in os.listdir(dreams_dir) if f.endswith('.json')]
                dream_files.sort(reverse=True)  # Most recent first
                
                for dream_file in dream_files[:limit]:
                    dream_path = os.path.join(dreams_dir, dream_file)
                    try:
                        with open(dream_path, 'r') as f:
                            dream = json.load(f)
                            dreams.append(dream)
                    except Exception as e:
                        print(f"Error loading dream {dream_file}: {e}")
                        continue
        
        except Exception as e:
            print(f"Error getting recent dreams: {e}")
        
        return dreams