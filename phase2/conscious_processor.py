import json
import os
from typing import Dict, List, Any, Optional
from interfaces.llm_interface import LLMInterface

class ConsciousProcessor:
    """
    Processes conscious thought patterns with deep integration of unconscious influences.
    
    Uses object_a dynamics, symptom patterns, signifying chains, and discourse positions
    to create psychoanalytically authentic responses.
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
        Generate conscious response deeply influenced by unconscious dynamics.
        """
        try:
            # Gather conscious context from memory
            conscious_context = self._gather_conscious_context(user_input)
            
            # Generate base response
            base_response = self._generate_base_response(user_input, conscious_context)
            
            # Apply unconscious modifications if available
            if unconscious_influence and self.unconscious_processor:
                final_response = self._apply_unconscious_modulation(
                    base_response, unconscious_influence, user_input
                )
            else:
                final_response = base_response
            
            return final_response
            
        except Exception as e:
            print(f"Error in conscious processing: {e}")
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
    
    def _generate_base_response(self, user_input: str, context: Dict) -> str:
        """Generate base rational response using available context."""
        template_data = self._format_template_data(user_input, context)
        
        # Use template if available, otherwise direct prompt
        if os.path.exists("phase2/prompts/agent_response.mustache"):
            response = self.llm.generate("phase2", "agent_response", template_data)
        else:
            prompt = self._create_direct_prompt(template_data)
            response = self.llm.generate(None, prompt, None)
        
        return response if response else self._generate_fallback_response(user_input)
    
    def _apply_unconscious_modulation(self, base_response: str, unconscious_influence: Dict, user_input: str) -> str:
        """Apply deep unconscious influence to conscious response."""
        print(f"\n=== Applying Unconscious Modulation ===")
        
        response = base_response
        
        # 1. Apply object_a dynamics - treat user as substitute/void
        response = self._apply_object_a_dynamics(response, unconscious_influence, user_input)
        
        # 2. Apply symptom patterns - repetitive behaviors
        response = self._apply_symptom_patterns(response, unconscious_influence)
        
        # 3. Apply discourse position - speech style
        response = self._apply_discourse_position(response, unconscious_influence)
        
        # 4. Apply signifying chain effects - associative responses
        response = self._apply_signifying_chain_effects(response, unconscious_influence)
        
        # 5. Apply repressed content emergence - slips, hesitations
        response = self._apply_repressed_content_effects(response, unconscious_influence)
        
        print(f"Response modulated by unconscious dynamics")
        return response
    
    def _apply_object_a_dynamics(self, response: str, unconscious_influence: Dict, user_input: str) -> str:
        """Apply object_a relationship dynamics to response."""
        object_a_effects = unconscious_influence.get('object_a_effects', {})
        proximity_level = object_a_effects.get('proximity_level', 0.0)
        desire_direction = object_a_effects.get('desire_direction', 'neutral')
        
        if proximity_level > 0.3:
            print(f"Object a proximity: {proximity_level:.2f}, direction: {desire_direction}")
            
            if desire_direction == 'seeking_substitute':
                # Treat user as potential substitute for missing object
                response = self._add_seeking_behaviors(response, user_input)
                print("Applied seeking substitute behavior")
                
            elif desire_direction == 'circling_void':
                # Circle around the void without directly approaching
                response = self._add_circling_behaviors(response)
                print("Applied circling void behavior")
        
        return response
    
    def _add_seeking_behaviors(self, response: str, user_input: str) -> str:
        """Add seeking/attachment behaviors when user triggers object_a substitute."""
        # Add subtle seeking patterns
        seeking_patterns = [
            ("I think", "I hope you understand"),
            ("Maybe", "I really need to know"),
            ("It's possible", "It's important that"),
            ("I suppose", "I'm hoping"),
        ]
        
        for old, new in seeking_patterns:
            if old in response:
                response = response.replace(old, new, 1)
                break
        
        # Add questions that seek connection/reassurance
        if not response.endswith('?') and len(response.split('.')) > 1:
            sentences = response.split('.')
            if len(sentences) > 1 and sentences[-2].strip():
                # Add seeking question
                seeking_questions = [
                    "Does that make sense to you?",
                    "Do you understand what I mean?",
                    "What do you think about that?",
                    "Can you help me with this?"
                ]
                import random
                question = seeking_questions[hash(user_input) % len(seeking_questions)]
                sentences[-2] = sentences[-2].strip() + f". {question}"
                response = '.'.join(sentences)
        
        return response
    
    def _add_circling_behaviors(self, response: str) -> str:
        """Add circling behaviors that approach but don't directly address the void."""
        # Add hesitation and indirection
        if response and not response.startswith(("Well", "You know", "I mean")):
            hesitations = ["Well, ", "You know, ", "I mean, ", "It's like... "]
            hesitation = hesitations[hash(response) % len(hesitations)]
            response = hesitation + response[0].lower() + response[1:]
        
        # Add trailing uncertainty
        if response.endswith('.'):
            uncertainties = ["...", " or something like that.", " - if that makes sense."]
            uncertainty = uncertainties[hash(response) % len(uncertainties)]
            response = response[:-1] + uncertainty
        
        return response
    
    def _apply_symptom_patterns(self, response: str, unconscious_influence: Dict) -> str:
        """Apply symptom-based repetitive patterns."""
        symptom_effects = unconscious_influence.get('symptom_effects', {})
        activation_level = symptom_effects.get('activation_level', 0.0)
        jouissance_pattern = symptom_effects.get('jouissance_pattern', '')
        
        if activation_level > 0.3:
            print(f"Symptom activation: {activation_level:.2f}")
            
            # Apply repetitive patterns based on the agent's specific symptom
            if 'vulnerability' in jouissance_pattern.lower():
                response = self._add_vulnerability_patterns(response)
                print("Applied vulnerability symptom pattern")
                
            elif 'attention' in jouissance_pattern.lower():
                response = self._add_attention_seeking_patterns(response)
                print("Applied attention-seeking symptom pattern")
        
        return response
    
    def _add_vulnerability_patterns(self, response: str) -> str:
        """Add vulnerability-based symptom patterns."""
        # Add self-deprecating or helpless elements
        vulnerability_insertions = [
            ("I know", "I'm not sure I know"),
            ("I can", "I'm not sure I can"),
            ("It's clear", "It's confusing to me"),
            ("Obviously", "I'm probably wrong, but"),
        ]
        
        for old, new in vulnerability_insertions:
            if old in response:
                response = response.replace(old, new, 1)
                break
        
        return response
    
    def _add_attention_seeking_patterns(self, response: str) -> str:
        """Add attention-seeking symptom patterns."""
        # Add elements that invite care or attention
        attention_patterns = [
            ("I'm fine", "I'm struggling a bit"),
            ("It's okay", "It's been difficult"),
            ("No problem", "It's challenging for me"),
        ]
        
        for old, new in attention_patterns:
            if old in response:
                response = response.replace(old, new, 1)
                break
        
        return response
    
    def _apply_discourse_position(self, response: str, unconscious_influence: Dict) -> str:
        """Apply discourse position to speech style."""
        discourse_position = unconscious_influence.get('discourse_position', {})
        primary_discourse = max(discourse_position.items(), key=lambda x: x[1])[0] if discourse_position else 'hysteric'
        
        print(f"Applying {primary_discourse} discourse position")
        
        if primary_discourse == 'hysteric':
            response = self._apply_hysteric_discourse(response)
        elif primary_discourse == 'master':
            response = self._apply_master_discourse(response)
        elif primary_discourse == 'university':
            response = self._apply_university_discourse(response)
        elif primary_discourse == 'analyst':
            response = self._apply_analyst_discourse(response)
        
        return response
    
    def _apply_hysteric_discourse(self, response: str) -> str:
        """Apply hysteric discourse - questioning, uncertain, seeking."""
        # Add questioning and uncertainty
        if '.' in response and not '?' in response:
            sentences = response.split('.')
            if len(sentences) > 1 and sentences[-2].strip():
                # Turn statement into question
                questions = ["don't you think?", "wouldn't you say?", "right?"]
                question = questions[hash(response) % len(questions)]
                sentences[-2] = sentences[-2].strip() + f", {question}"
                response = '.'.join(sentences)
        
        # Add hysteric hesitation
        hysteric_markers = [
            ("I think", "I wonder if"),
            ("It is", "Could it be that it's"),
            ("This means", "Does this mean"),
        ]
        
        for old, new in hysteric_markers:
            if old in response:
                response = response.replace(old, new, 1)
                break
        
        return response
    
    def _apply_master_discourse(self, response: str) -> str:
        """Apply master discourse - assertive, commanding."""
        # Make more definitive
        master_patterns = [
            ("I think", "I know"),
            ("maybe", "certainly"),
            ("could be", "is"),
            ("Perhaps", "Clearly"),
            ("It seems", "It is"),
        ]
        
        for old, new in master_patterns:
            response = response.replace(old, new)
        
        return response
    
    def _apply_university_discourse(self, response: str) -> str:
        """Apply university discourse - knowledge-focused, explanatory."""
        if len(response) > 50 and not response.startswith(("Research", "Studies", "It's established")):
            knowledge_phrases = [
                "Research shows that ",
                "It's well established that ",
                "Studies indicate that "
            ]
            phrase = knowledge_phrases[hash(response) % len(knowledge_phrases)]
            response = phrase + response[0].lower() + response[1:]
        
        return response
    
    def _apply_analyst_discourse(self, response: str) -> str:
        """Apply analyst discourse - interpretive, reflective."""
        if not any(phrase in response.lower() for phrase in ['you said', 'i notice', 'interesting']):
            analytic_openings = [
                "I notice that ",
                "It's interesting how ",
                "What I'm hearing is that "
            ]
            opening = analytic_openings[hash(response) % len(analytic_openings)]
            response = opening + response[0].lower() + response[1:]
        
        return response
    
    def _apply_signifying_chain_effects(self, response: str, unconscious_influence: Dict) -> str:
        """Apply signifying chain effects - associative responses."""
        activated_chains = unconscious_influence.get('activated_chains', [])
        
        for chain in activated_chains:
            if chain.get('activation_strength', 0) > 0.5:
                chain_signifiers = chain.get('active_signifiers', [])
                if len(chain_signifiers) > 1:
                    # Subtle influence - if one signifier is mentioned, slightly favor others
                    for signifier in chain_signifiers[1:]:  # Skip first
                        if signifier.lower() not in response.lower():
                            # Very subtle insertion - only if it makes sense
                            if self._can_insert_signifier(response, signifier):
                                response = self._subtly_insert_signifier(response, signifier)
                                print(f"Chain effect: inserted {signifier} from {chain['name']} chain")
                                break
        
        return response
    
    def _can_insert_signifier(self, response: str, signifier: str) -> bool:
        """Check if signifier can be naturally inserted."""
        # Very conservative - only insert if response is long enough and signifier is short
        return len(response) > 100 and len(signifier.split()) == 1
    
    def _subtly_insert_signifier(self, response: str, signifier: str) -> str:
        """Subtly insert signifier into response."""
        # Find a natural insertion point
        words = response.split()
        if len(words) > 10:
            # Insert in middle third of response
            start_idx = len(words) // 3
            end_idx = 2 * len(words) // 3
            insert_idx = start_idx + hash(signifier) % (end_idx - start_idx)
            
            # Create natural insertion
            insertions = [
                f"like {signifier}",
                f"or {signifier}",
                f"- {signifier} -",
                f"({signifier})"
            ]
            insertion = insertions[hash(response) % len(insertions)]
            words.insert(insert_idx, insertion)
            
            return ' '.join(words)
        
        return response
    
    def _apply_repressed_content_effects(self, response: str, unconscious_influence: Dict) -> str:
        """Apply effects of repressed content emergence."""
        repressed_returns = unconscious_influence.get('repressed_returns', [])
        
        if repressed_returns:
            # Add slips, hesitations, corrections
            for repressed in repressed_returns:
                if repressed.get('return_strength', 0) > 0.7:
                    # Strong return - add hesitation or slip
                    response = self._add_repressed_slip(response, repressed['signifier'])
                    print(f"Repressed slip added for: {repressed['signifier']}")
                    break
        
        return response
    
    def _add_repressed_slip(self, response: str, repressed_signifier: str) -> str:
        """Add Freudian slip or hesitation related to repressed content."""
        if len(response.split()) > 5:
            words = response.split()
            # Insert hesitation in first third
            hesitation_idx = len(words) // 4
            
            hesitations = [
                "... um ...",
                "- wait -",
                "... no, I mean ...",
                "... how do I put this ..."
            ]
            hesitation = hesitations[hash(repressed_signifier) % len(hesitations)]
            words.insert(hesitation_idx, hesitation)
            
            return ' '.join(words)
        
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
        return f"""You are {template_data['agent_name']} responding to: "{template_data['user_message']}"

Your current emotional state: {template_data['emotional_description']}

Respond authentically as {template_data['agent_name']} based on your personality and current emotional state."""
    
    def _format_conversation_history(self, recent_interactions: List[Dict]) -> List[Dict]:
        """Format conversation history for template."""
        formatted = []
        user_msg = None
        
        for entry in recent_interactions[-6:]:
            content = entry.get('content', '')
            context = entry.get('context', '')
            
            if context == 'user_interaction':
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
            "Let me think about that for a moment."
        ]
        return fallback_responses[hash(user_input) % len(fallback_responses)]