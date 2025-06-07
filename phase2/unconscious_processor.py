import json
from typing import Dict, List, Any
from interfaces.llm_interface import LLMInterface

class UnconsciousProcessor:
    def __init__(self, agent_name: str, memory_manager, dream_generator):
        """Initialize unconscious processor."""
        self.agent_name = agent_name
        self.memory_manager = memory_manager
        self.dream_generator = dream_generator
        self.llm = LLMInterface()
        
        print(f"Unconscious Processor initialized for {agent_name}")
    
    def process_input(self, user_input: str, context: str = "dialogue") -> Dict[str, Any]:
        """Process input through unconscious lens."""
        try:
            # Get unconscious signifiers
            signifiers = self.memory_manager.get_unconscious_signifiers(10)
            emotional_state = self.memory_manager.get_emotional_state()
            
            # Analyze input for unconscious patterns
            unconscious_analysis = self._analyze_unconscious_patterns(user_input, signifiers)
            
            # Generate unconscious influence on response
            unconscious_influence = {
                "emotional_coloring": self._get_emotional_coloring(user_input, emotional_state),
                "symbolic_associations": unconscious_analysis.get("symbolic_associations", []),
                "repressed_elements": unconscious_analysis.get("repressed_elements", []),
                "projection_tendencies": self._analyze_projections(user_input, signifiers)
            }
            
            return unconscious_influence
            
        except Exception as e:
            print(f"Error in unconscious processing: {e}")
            return {"emotional_coloring": "neutral", "symbolic_associations": [], "repressed_elements": [], "projection_tendencies": []}
    
    def _analyze_unconscious_patterns(self, user_input: str, signifiers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze input for unconscious patterns."""
        try:
            analysis_context = {
                "user_input": user_input,
                "signifiers": signifiers,
                "agent_name": self.agent_name
            }
            
            analysis = self.llm.generate("phase2", "unconscious_analysis", analysis_context)
            
            # Try to parse as JSON, fallback to text analysis
            try:
                return json.loads(analysis)
            except:
                return {
                    "symbolic_associations": self._extract_symbolic_associations(user_input, signifiers),
                    "repressed_elements": [],
                    "analysis_text": analysis
                }
                
        except Exception as e:
            print(f"Error analyzing unconscious patterns: {e}")
            return {"symbolic_associations": [], "repressed_elements": []}
    
    def _extract_symbolic_associations(self, user_input: str, signifiers: List[Dict[str, Any]]) -> List[str]:
        """Extract symbolic associations from input."""
        associations = []
        user_input_lower = user_input.lower()
        
        for signifier in signifiers:
            if isinstance(signifier, dict):
                name = signifier.get("name", "").lower()
                if name and name in user_input_lower:
                    associations.append(signifier.get("significance", name))
        
        return associations
    
    def _get_emotional_coloring(self, user_input: str, emotional_state: Dict[str, Any]) -> str:
        """Get emotional coloring for the response."""
        emotion_category = emotional_state.get("emotion_category", "neutral")
        
        # Map emotions to response colorings
        coloring_map = {
            "joy": "optimistic",
            "sadness": "melancholic",
            "anger": "defensive",
            "fear": "cautious",
            "surprise": "curious",
            "disgust": "critical",
            "neutral": "balanced"
        }
        
        return coloring_map.get(emotion_category, "balanced")
    
    def _analyze_projections(self, user_input: str, signifiers: List[Dict[str, Any]]) -> List[str]:
        """Analyze potential projections in the input."""
        projections = []
        
        # Simple keyword-based projection analysis
        projection_keywords = ["you are", "you seem", "you always", "you never", "people like you"]
        user_input_lower = user_input.lower()
        
        for keyword in projection_keywords:
            if keyword in user_input_lower:
                projections.append(f"Potential projection detected: '{keyword}'")
        
        return projections
    
    def generate_unconscious_response_influence(self, conscious_response: str) -> str:
        """Generate unconscious influence on conscious response."""
        try:
            emotional_state = self.memory_manager.get_emotional_state()
            signifiers = self.memory_manager.get_unconscious_signifiers(5)
            
            influence_context = {
                "conscious_response": conscious_response,
                "emotional_state": emotional_state,
                "signifiers": signifiers,
                "agent_name": self.agent_name
            }
            
            influenced_response = self.llm.generate("phase2", "unconscious_influence", influence_context)
            
            return influenced_response if influenced_response else conscious_response
            
        except Exception as e:
            print(f"Error generating unconscious influence: {e}")
            return conscious_response