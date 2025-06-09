import json
import os
from typing import Dict, List, Any, Optional
from interfaces.llm_interface import LLMInterface

class ConsciousProcessor:
    """Process conscious interactions with proper psychoanalytic integration."""
    
    def __init__(self, agent_name: str, memory_manager):
        self.agent_name = agent_name
        self.memory_manager = memory_manager
        self.llm = LLMInterface()
        
        # Initialize the unconscious processor reference (set by brain)
        self.unconscious_processor = None
        
        print(f"Conscious Processor initialized for {agent_name}")
    
    def set_unconscious_processor(self, unconscious_processor):
        """Set reference to unconscious processor for integration."""
        self.unconscious_processor = unconscious_processor
    
    def process_input(self, user_input: str, context: str = "dialogue") -> str:
        """Process input with full conscious-unconscious integration."""
        try:
            # First, process through unconscious
            unconscious_influence = None
            if self.unconscious_processor:
                unconscious_influence = self.unconscious_processor.process_input(user_input, context)
            
            # Gather conscious context
            conscious_context = self._gather_conscious_context(user_input)
            
            # Check for defense mechanisms based on unconscious activity
            defenses = self._check_defense_mechanisms(unconscious_influence)
            
            # Generate initial conscious response
            initial_response = self._generate_conscious_response(
                user_input, conscious_context, defenses
            )
            
            # Apply unconscious influence to create final response
            if unconscious_influence and self.unconscious_processor:
                final_response = self._integrate_unconscious_influence(
                    initial_response, unconscious_influence
                )
            else:
                final_response = initial_response
            
            # Check for and handle potential acting out
            final_response = self._check_acting_out(final_response, unconscious_influence)
            
            return final_response
            
        except Exception as e:
            print(f"Error in conscious processing: {e}")
            return self._generate_defensive_response(user_input)
    
    def _gather_conscious_context(self, user_input: str) -> Dict[str, Any]:
        """Gather relevant conscious memories and relationships."""
        context = {
            'memories': self.memory_manager.retrieve_memories(user_input, 5),
            'relationships': self.memory_manager.retrieve_relationships(user_input, 3),
            'persona': self.memory_manager.get_persona(),
            'emotional_state': self.memory_manager.get_emotional_state(),
            'recent_interactions': self.memory_manager.get_short_term_memory(5)
        }
        return context
    
    def _check_defense_mechanisms(self, unconscious_influence: Optional[Dict]) -> Dict[str, Any]:
        """Identify active defense mechanisms based on unconscious activity."""
        defenses = {
            'active': [],
            'intensity': 0.0,
            'primary_defense': None
        }
        
        if not unconscious_influence:
            return defenses
        
        # Check for resistance
        resistance = unconscious_influence.get('resistance', {})
        if resistance.get('present', False):
            defenses['active'].append('resistance')
            defenses['intensity'] = max(defenses['intensity'], resistance.get('intensity', 0.5))
        
        # Check for repression activation
        repressed_returns = unconscious_influence.get('repressed_returns', [])
        if repressed_returns:
            defenses['active'].append('repression')
            defenses['intensity'] = max(defenses['intensity'], 0.7)
        
        # Check for projection (based on transference)
        transference = unconscious_influence.get('transference', {})
        if transference.get('type') == 'negative' and transference.get('intensity', 0) > 0.6:
            defenses['active'].append('projection')
            defenses['intensity'] = max(defenses['intensity'], transference['intensity'])
        
        # Check for rationalization (high jouissance with low awareness)
        jouissance = unconscious_influence.get('jouissance_effects', {})
        if jouissance.get('level', 0) > 0.6 and jouissance.get('symptom_activation', False):
            defenses['active'].append('rationalization')
            defenses['intensity'] = max(defenses['intensity'], 0.6)
        
        # Check for displacement
        chain_effects = unconscious_influence.get('chain_activations', {})
        if any(point.get('slippage_type') == 'displacement' for point in chain_effects.get('slippage_points', [])):
            defenses['active'].append('displacement')
            defenses['intensity'] = max(defenses['intensity'], 0.5)
        
        # Determine primary defense
        if defenses['active']:
            if 'repression' in defenses['active']:
                defenses['primary_defense'] = 'repression'
            elif 'resistance' in defenses['active']:
                defenses['primary_defense'] = 'resistance'
            else:
                defenses['primary_defense'] = defenses['active'][0]
        
        return defenses
    
    def _generate_conscious_response(self, user_input: str, context: Dict, defenses: Dict) -> str:
        """Generate conscious response considering defenses."""
        # Format template data
        template_data = self._format_template_data(user_input, context)
        
        # Add defense mechanism guidance
        if defenses['primary_defense']:
            defense_guidance = self._get_defense_guidance(defenses)
            template_data['psychological_state'] = defense_guidance
        
        # Generate response using proper template or direct prompt
        if os.path.exists("phase2/prompts/agent_response.mustache"):
            # Use template if available
            response = self.llm.generate("phase2", "agent_response", template_data)
        else:
            # Use direct prompt as fallback
            prompt = f"""
You are {self.agent_name} responding to: "{user_input}"

Your psychological state:
- Emotional: {template_data.get('emotional_description', 'neutral')}
- Defenses: {defenses.get('primary_defense', 'none')} (intensity: {defenses['intensity']})

Relevant memories: {json.dumps(template_data.get('relevant_memories', [])[:3], indent=2)}

Respond authentically as {self.agent_name}, letting your defenses naturally shape your response:
- If repressing, avoid certain topics subtly
- If resisting, redirect or intellectualize
- If projecting, attribute feelings to others
- If rationalizing, provide logical explanations for emotional reactions

Keep the defense mechanisms subtle and natural to the conversation.
"""
            response = self.llm.generate(None, prompt, None)
        
        return response if response else self._generate_defensive_response(user_input)
    
    def _integrate_unconscious_influence(self, conscious_response: str, unconscious_influence: Dict) -> str:
        """Integrate unconscious influence into conscious response."""
        # Get discourse position for speech style
        discourse = max(unconscious_influence.get('discourse_position', {}).items(), 
                       key=lambda x: x[1])[0] if unconscious_influence.get('discourse_position') else 'hysteric'
        
        # Check for potential slips
        slips = unconscious_influence.get('slips_and_parapraxes', [])
        
        # Apply discourse-specific modifications
        if discourse == 'master':
            # Assertive, commanding style
            response = self._apply_master_discourse(conscious_response)
        elif discourse == 'hysteric':
            # Questioning, seeking style
            response = self._apply_hysteric_discourse(conscious_response)
        elif discourse == 'university':
            # Knowledge-focused, explaining style
            response = self._apply_university_discourse(conscious_response)
        elif discourse == 'analyst':
            # Reflective, interpreting style
            response = self._apply_analyst_discourse(conscious_response)
        
        # Insert slips if present
        if slips and len(slips) > 0:
            response = self._insert_slip(response, slips[0])
        
        # Apply fantasy influence if activated
        fantasy = unconscious_influence.get('fantasy_activation', {})
        if fantasy.get('activated', False):
            response = self._apply_fantasy_influence(response, fantasy)
        
        return response
    
    def _apply_master_discourse(self, response: str) -> str:
        """Apply master's discourse style (S1 → S2)."""
        # Make more assertive and definitive
        replacements = [
            ("I think", "I know"),
            ("maybe", "certainly"),
            ("could be", "is"),
            ("Perhaps", "Clearly"),
            ("It seems", "It is")
        ]
        
        for old, new in replacements:
            response = response.replace(old, new)
        
        # Add commanding elements
        if not response.endswith('.'):
            response += '.'
        
        return response
    
    def _apply_hysteric_discourse(self, response: str) -> str:
        """Apply hysteric's discourse style ($ → S1)."""
        # Add questioning and uncertainty
        if '.' in response and not '?' in response:
            sentences = response.split('.')
            if len(sentences) > 1:
                # Convert last sentence to question
                last = sentences[-2].strip()  # -2 because split leaves empty after last .
                if last:
                    sentences[-2] = last + ", don't you think?"
                response = '.'.join(sentences)
        
        # Add self-questioning
        additions = [
            " But what do I know?",
            " Or am I wrong?",
            " Does that make sense?"
        ]
        
        if len(response) < 200 and not response.endswith('?'):
            response += additions[hash(response) % len(additions)]
        
        return response
    
    def _apply_university_discourse(self, response: str) -> str:
        """Apply university discourse style (S2 → a)."""
        # Add explanatory and knowledge-focused elements
        knowledge_phrases = [
            "Research shows that",
            "It's well established that",
            "Studies indicate",
            "The evidence suggests"
        ]
        
        # Insert knowledge reference if appropriate
        if len(response) > 50 and ',' in response:
            parts = response.split(',', 1)
            if len(parts[0]) > 20:
                phrase = knowledge_phrases[hash(response) % len(knowledge_phrases)]
                response = f"{phrase} {response[0].lower()}{response[1:]}"
        
        return response
    
    def _apply_analyst_discourse(self, response: str) -> str:
        """Apply analyst's discourse style (a → $)."""
        # Add reflective elements
        if not any(phrase in response.lower() for phrase in ['you said', 'you mentioned', 'I notice']):
            reflective_openings = [
                "I notice that ",
                "It's interesting how ",
                "I'm hearing that "
            ]
            
            opening = reflective_openings[hash(response) % len(reflective_openings)]
            response = opening + response[0].lower() + response[1:]
        
        return response
    
    def _insert_slip(self, response: str, slip: Dict) -> str:
        """Insert a Freudian slip into the response."""
        if slip['type'] == 'substitution':
            intended = slip.get('intended', '')
            slip_word = slip.get('slip', '')
            
            if intended in response:
                # Insert slip with correction
                response = response.replace(
                    intended,
                    f"{slip_word}... I mean, {intended}",
                    1
                )
        elif slip['type'] == 'return_of_repressed':
            # Add an unexpected phrase
            position = len(response) // 2
            response = response[:position] + "... wait, what was I saying? ... " + response[position:]
        
        return response
    
    def _apply_fantasy_influence(self, response: str, fantasy: Dict) -> str:
        """Apply influence of activated fundamental fantasy."""
        # Fantasy activation makes speech more circular around object a
        if fantasy.get('strength', 0) > 0.5:
            # Add circular phrases
            circular_additions = [
                " But then again...",
                " Though I wonder...",
                " It always comes back to...",
                " Somehow this reminds me..."
            ]
            
            addition = circular_additions[hash(response) % len(circular_additions)]
            if len(response) < 300:
                response += addition
        
        return response
    
    def _check_acting_out(self, response: str, unconscious_influence: Optional[Dict]) -> str:
        """Check for and handle acting out of unconscious material."""
        if not unconscious_influence:
            return response
        
        # High jouissance + high resistance = potential acting out
        jouissance = unconscious_influence.get('jouissance_effects', {})
        resistance = unconscious_influence.get('resistance', {})
        
        if (jouissance.get('level', 0) > 0.7 and 
            resistance.get('present', False) and 
            resistance.get('intensity', 0) > 0.6):
            
            # Add a defensive ending to prevent full acting out
            defensive_endings = [
                " But let's not dwell on that.",
                " Anyway, that's not important right now.",
                " Though I'm not sure why I'm telling you this.",
                " But that's beside the point."
            ]
            
            ending = defensive_endings[hash(response) % len(defensive_endings)]
            response += ending
        
        return response
    
    def _get_defense_guidance(self, defenses: Dict) -> str:
        """Get guidance for how defenses shape response."""
        primary = defenses.get('primary_defense', '')
        intensity = defenses.get('intensity', 0)
        
        if primary == 'repression':
            return f"Avoiding certain topics (repression active at {intensity:.1%})"
        elif primary == 'resistance':
            return f"Redirecting away from difficult material (resistance at {intensity:.1%})"
        elif primary == 'projection':
            return f"Attributing own feelings to others (projection at {intensity:.1%})"
        elif primary == 'rationalization':
            return f"Explaining away emotional reactions (rationalization at {intensity:.1%})"
        elif primary == 'displacement':
            return f"Focusing on less threatening topics (displacement at {intensity:.1%})"
        else:
            return "Open and non-defensive"
    
    def _generate_defensive_response(self, user_input: str) -> str:
        """Generate a defensive response when processing fails."""
        defensive_responses = [
            f"I'm not sure I understand what you're getting at.",
            f"That's an interesting way to put it. What makes you say that?",
            f"I need to think about that for a moment.",
            f"Hmm, I'm not quite following. Could you elaborate?",
            f"That's... well, I'm not sure how to respond to that."
        ]
        
        return defensive_responses[hash(user_input) % len(defensive_responses)]
    
    def _format_template_data(self, user_input: str, context: Dict) -> Dict[str, Any]:
        """Format data for response generation."""
        persona = context.get('persona', {})
        emotional_state = context.get('emotional_state', {})
        memories = context.get('memories', [])
        relationships = context.get('relationships', [])
        
        # Extract neurochemical data
        neurochemical = emotional_state.get('neurochemical_state', {})
        
        return {
            'agent_name': self.agent_name,
            'user_message': user_input,
            'persona_name': persona.get('name', self.agent_name),
            'personality_traits': persona.get('personality_traits', []),
            'emotional_description': emotional_state.get('emotional_description', 'neutral'),
            'emotional_pleasure': emotional_state.get('pleasure', 0.0),
            'emotional_arousal': emotional_state.get('arousal', 0.0),
            'emotional_dominance': emotional_state.get('dominance', 0.0),
            'neurochemical_dopamine': neurochemical.get('dopamine', 0.5),
            'neurochemical_serotonin': neurochemical.get('serotonin', 0.5),
            'neurochemical_oxytocin': neurochemical.get('oxytocin', 0.5),
            'neurochemical_cortisol': neurochemical.get('cortisol', 0.3),
            'relevant_memories': [self._format_memory(m) for m in memories[:5]],
            'relevant_relationships': [self._format_relationship(r) for r in relationships[:3]]
        }
    
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