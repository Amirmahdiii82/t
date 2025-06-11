import json
import os
from typing import Dict, List, Any, Optional
from interfaces.llm_interface import LLMInterface

class ConsciousProcessor:
    """
    Processes conscious thought patterns with integration of unconscious influences.
    
    Handles rational response generation while being modulated by unconscious
    signifier activation and emotional state.
    """
    
    def __init__(self, agent_name: str, memory_manager):
        self.agent_name = agent_name
        self.memory_manager = memory_manager
        self.llm = LLMInterface()
        self.unconscious_processor = None
        
    def set_unconscious_processor(self, unconscious_processor):
        """Set reference to unconscious processor for integration."""
        self.unconscious_processor = unconscious_processor
    
    def process_input(self, user_input: str, unconscious_influence: Optional[Dict] = None) -> str:
        """
        Generate conscious response influenced by unconscious dynamics.
        
        Args:
            user_input: User's message
            unconscious_influence: Unconscious processing results
            
        Returns:
            Generated response as string
        """
        try:
            # Gather conscious context from memory
            conscious_context = self._gather_conscious_context(user_input)
            
            # Generate initial response
            initial_response = self._generate_conscious_response(user_input, conscious_context)
            
            # Apply unconscious influence if available
            if unconscious_influence and self.unconscious_processor:
                print("=== Applying Unconscious Modulation ===")
                final_response = self._apply_unconscious_modulation(
                    initial_response, unconscious_influence
                )
                print("Response modulated by unconscious dynamics")
            else:
                final_response = initial_response
            
            return final_response
            
        except Exception as e:
            print(f"Warning: Conscious processing error: {e}")
            return self._generate_fallback_response(user_input)
    
    def _gather_conscious_context(self, user_input: str) -> Dict[str, Any]:
        """Gather relevant conscious memories and relationships."""
        return {
            'memories': self.memory_manager.retrieve_memories(user_input, 5),
            'relationships': self.memory_manager.retrieve_relationships(user_input, 3),
            'persona': self.memory_manager.get_persona(),
            'emotional_state': self.memory_manager.get_emotional_state(),
            'recent_interactions': self.memory_manager.get_short_term_memory(5)
        }
    
    def _generate_conscious_response(self, user_input: str, context: Dict) -> str:
        """Generate rational response using available context."""
        template_data = self._format_template_data(user_input, context)
        
        # Use template if available, otherwise direct prompt
        if os.path.exists("phase2/prompts/agent_response.mustache"):
            try:
                response = self.llm.generate("phase2", "agent_response", template_data)
            except Exception as e:
                print(f"Warning: Could not load template phase2/agent_response: {e}")
                prompt = self._create_direct_prompt(template_data)
                response = self.llm.generate(None, prompt, None)
        else:
            prompt = self._create_direct_prompt(template_data)
            response = self.llm.generate(None, prompt, None)
        
        return response if response else self._generate_fallback_response(user_input)
    
    def _apply_unconscious_modulation(self, conscious_response: str, unconscious_influence: Dict) -> str:
        """Apply unconscious influence to conscious response."""
        # Get discourse position for speech style modulation
        discourse_scores = unconscious_influence.get('discourse_position', {})
        primary_discourse = max(discourse_scores.items(), key=lambda x: x[1])[0] if discourse_scores else 'hysteric'
        
        print(f"Applying {primary_discourse} discourse position")
        
        # Apply discourse-specific modulation
        if primary_discourse == 'master':
            response = self._apply_master_discourse_style(conscious_response)
        elif primary_discourse == 'hysteric':
            response = self._apply_hysteric_discourse_style(conscious_response)
        elif primary_discourse == 'university':
            response = self._apply_university_discourse_style(conscious_response)
        elif primary_discourse == 'analyst':
            response = self._apply_analyst_discourse_style(conscious_response)
        else:
            response = conscious_response
        
        # Apply signifier influence through subtle word choices
        active_signifiers = unconscious_influence.get('active_signifiers', [])
        if active_signifiers:
            response = self._apply_signifier_influence(response, active_signifiers)
        
        return response
    
    def _apply_master_discourse_style(self, response: str) -> str:
        """Apply master's discourse style (assertive, definitive)."""
        replacements = [
            ("I think", "I know"),
            ("maybe", "certainly"),
            ("could be", "is"),
            ("Perhaps", "Clearly"),
            ("might", "will"),
            ("I'm not sure", "I'm confident")
        ]
        
        for old, new in replacements:
            response = response.replace(old, new)
        
        return response
    
    def _apply_hysteric_discourse_style(self, response: str) -> str:
        """Apply hysteric's discourse style (questioning, uncertain)."""
        # Add questioning elements without being too obvious
        if '.' in response and '?' not in response[-20:]:
            sentences = response.split('.')
            if len(sentences) > 1 and sentences[-2].strip():
                # Sometimes add questioning
                if len(sentences[-2]) > 20:
                    sentences[-2] = sentences[-2].strip() + ", don't you think?"
                response = '.'.join(sentences)
        
        # Make some statements less certain
        response = response.replace("I am", "I feel I am")
        response = response.replace("This is", "This seems to be")
        
        return response
    
    def _apply_university_discourse_style(self, response: str) -> str:
        """Apply university discourse style (knowledge-focused, explanatory)."""
        knowledge_phrases = [
            "Research shows that",
            "It's well established that",
            "Studies indicate",
            "From what I understand"
        ]
        
        # Sometimes add knowledge authority
        if len(response) > 50 and ',' in response:
            parts = response.split(',', 1)
            if len(parts[0]) > 20 and not any(phrase in response for phrase in knowledge_phrases):
                phrase = knowledge_phrases[hash(response) % len(knowledge_phrases)]
                response = f"{phrase} {response[0].lower()}{response[1:]}"
        
        return response
    
    def _apply_analyst_discourse_style(self, response: str) -> str:
        """Apply analyst's discourse style (reflective, interpretive)."""
        if not any(phrase in response.lower() for phrase in ['you said', 'i notice', 'i hear']):
            reflective_openings = [
                "I notice that ",
                "It's interesting how ",
                "I'm hearing that ",
                "What I'm picking up is that "
            ]
            
            # Sometimes apply reflective style
            if len(response) > 30:
                opening = reflective_openings[hash(response) % len(reflective_openings)]
                response = opening + response[0].lower() + response[1:]
        
        return response
    
    def _apply_signifier_influence(self, response: str, active_signifiers: List[Dict]) -> str:
        """Subtly incorporate active signifiers through word associations."""
        # Get the most activated signifier
        if not active_signifiers:
            return response
        
        # Sort by activation strength
        sorted_signifiers = sorted(active_signifiers, 
                                 key=lambda x: x.get('activation_strength', 0), 
                                 reverse=True)
        
        primary_signifier = sorted_signifiers[0]['signifier']
        
        # Subtle influence through related concepts (not obvious insertion)
        signifier_associations = {
            'father': ['authority', 'guidance', 'structure', 'leadership'],
            'mother': ['care', 'nurturing', 'comfort', 'support'],
            'anxiety': ['uncertainty', 'concern', 'tension', 'worry'],
            'love': ['connection', 'warmth', 'understanding', 'bond'],
            'family': ['closeness', 'belonging', 'home', 'relationships'],
            'fear': ['caution', 'concern', 'uncertainty', 'hesitation'],
            'identity': ['self', 'understanding', 'clarity', 'purpose'],
            'relationship': ['connection', 'interaction', 'understanding', 'communication']
        }
        
        primary_lower = primary_signifier.lower()
        if primary_lower in signifier_associations:
            associations = signifier_associations[primary_lower]
            # Subtly favor these concepts in word choice
            for association in associations:
                if association in response.lower():
                    # Already present - unconscious influence working
                    break
            else:
                # Try to incorporate one association subtly
                if len(associations) > 0:
                    chosen_assoc = associations[0]
                    # Very subtle replacement
                    response = response.replace("help", f"help and {chosen_assoc}")
                    response = response.replace("support", f"{chosen_assoc}")
                    response = response.replace("understand", f"understand and provide {chosen_assoc}")
        
        return response
    
    def _format_template_data(self, user_input: str, context: Dict) -> Dict[str, Any]:
        """Format data for template rendering."""
        persona = context.get('persona', {})
        emotional_state = context.get('emotional_state', {})
        neurochemical = emotional_state.get('neurochemical_state', {})
        
        return {
            'agent_name': self.agent_name,
            'user_message': user_input,
            
            # Persona information
            'persona_name': persona.get('name', self.agent_name),
            'persona_age': persona.get('age'),
            'persona_occupation': persona.get('occupation'),
            'personality_traits': persona.get('personality_traits', []),
            'background': persona.get('background'),
            
            # Emotional state
            'emotional_description': emotional_state.get('emotional_description', 'neutral'),
            'emotional_pleasure': f"{emotional_state.get('pleasure', 0.0):.2f}",
            'emotional_arousal': f"{emotional_state.get('arousal', 0.0):.2f}",
            'emotional_dominance': f"{emotional_state.get('dominance', 0.0):.2f}",
            
            # Neurochemical levels
            'neurochemical_dopamine': f"{neurochemical.get('dopamine', 0.5):.2f}",
            'neurochemical_serotonin': f"{neurochemical.get('serotonin', 0.5):.2f}",
            'neurochemical_oxytocin': f"{neurochemical.get('oxytocin', 0.5):.2f}",
            'neurochemical_cortisol': f"{neurochemical.get('cortisol', 0.3):.2f}",
            
            # Boolean flags for template conditionals
            'neurochemical_dopamine_high': neurochemical.get('dopamine', 0.5) > 0.6,
            'neurochemical_serotonin_high': neurochemical.get('serotonin', 0.5) > 0.6,
            'neurochemical_oxytocin_high': neurochemical.get('oxytocin', 0.5) > 0.6,
            'neurochemical_cortisol_high': neurochemical.get('cortisol', 0.3) > 0.5,
            
            # Memory content
            'has_memories': len(context.get('memories', [])) > 0,
            'relevant_memories': [self._format_memory(m) for m in context.get('memories', [])[:5]],
            'has_relationships': len(context.get('relationships', [])) > 0,
            'relevant_relationships': [self._format_relationship(r) for r in context.get('relationships', [])[:3]],
            
            # Conversation history
            'conversation_history': self._format_conversation_history(context.get('recent_interactions', []))
        }
    
    def _create_direct_prompt(self, template_data: Dict) -> str:
        """Create direct prompt when template unavailable."""
        prompt = f"You are {template_data['agent_name']} responding to: \"{template_data['user_message']}\"\n\n"
        
        # Add persona info if available
        if template_data.get('persona_name'):
            prompt += f"Your name is {template_data['persona_name']}. "
        if template_data.get('persona_occupation'):
            prompt += f"Your occupation is {template_data['persona_occupation']}. "
        if template_data.get('personality_traits'):
            traits = ', '.join(template_data['personality_traits'][:3])
            prompt += f"Your personality includes: {traits}. "
        
        # Add emotional state
        prompt += f"\nYour current emotional state: {template_data['emotional_description']}\n\n"
        
        # Add relevant memories if available
        if template_data.get('relevant_memories'):
            prompt += "This conversation brings to mind:\n"
            for memory in template_data['relevant_memories'][:3]:
                prompt += f"- {memory}\n"
        
        prompt += f"\nRespond naturally as {template_data['agent_name']} based on your personality and current emotional state."
        
        return prompt
    
    def _format_conversation_history(self, recent_interactions: List[Dict]) -> List[Dict]:
        """Format conversation history for template."""
        formatted = []
        user_msg = None
        
        for entry in recent_interactions[-6:]:
            content = entry.get('content', '')
            context = entry.get('context', '')
            
            if context == 'user_interaction' or context == 'dialogue':
                user_msg = content
            elif context == 'agent_response' and user_msg:
                formatted.append({"user": user_msg, "agent": content})
                user_msg = None
        
        return formatted[-2:] if formatted else []
    
    def _format_memory(self, memory: Dict) -> str:
        """Format memory for display."""
        if isinstance(memory, dict):
            title = memory.get('title', '')
            description = memory.get('description', '')
            return f"{title}: {description}" if title else description
        return str(memory)
    
    def _format_relationship(self, relationship: Dict) -> str:
        """Format relationship for display."""
        if isinstance(relationship, dict):
            name = relationship.get('name', 'Someone')
            rel_type = relationship.get('relationship_type', 'Unknown')
            significance = relationship.get('emotional_significance', '')
            return f"{name} ({rel_type}): {significance}" if significance else f"{name} ({rel_type})"
        return str(relationship)
    
    def _generate_fallback_response(self, user_input: str) -> str:
        """Generate fallback response when processing fails."""
        fallback_responses = [
            "I need a moment to process that.",
            "That's an interesting perspective.",
            "I'm reflecting on what you've shared.",
            "Let me think about that for a moment.",
            "I appreciate you sharing that with me.",
            "Could you tell me more about that?"
        ]
        return fallback_responses[hash(user_input) % len(fallback_responses)]