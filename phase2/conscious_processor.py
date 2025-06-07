from typing import Dict, List
from interfaces.llm_interface import LLMInterface
import random
import traceback

class ConsciousProcessor:
    def __init__(self, agent_name: str, memory_manager):
        """Initialize conscious processor."""
        self.agent_name = agent_name
        self.memory_manager = memory_manager
        self.llm = LLMInterface()
        
        print(f"Conscious Processor initialized for {agent_name}")
    
    def process_input(self, user_input: str, context: str = "dialogue") -> str:
        """Process user input and generate conscious response."""
        try:
            # Get relevant memories and context
            relevant_memories = self.memory_manager.retrieve_memories(user_input, 5)
            relevant_relationships = self.memory_manager.retrieve_relationships(user_input, 3)
            persona = self.memory_manager.get_persona()
            emotional_state = self.memory_manager.get_emotional_state()
            short_term_context = self.memory_manager.get_short_term_memory(10)
            
            # Format the data to match your agent_response.mustache template structure
            template_data = self._format_template_data(
                user_input, persona, emotional_state, 
                relevant_memories, relevant_relationships, short_term_context
            )
            
            print(f"Template data prepared with {len(relevant_memories)} memories and {len(relevant_relationships)} relationships")
            
            # Generate response using existing agent_response.mustache template
            response = self.llm.generate("phase2", "agent_response", template_data)
            
            if response:
                return response.strip()
            else:
                return self._generate_fallback_response(user_input, relevant_memories, relevant_relationships)
                
        except Exception as e:
            print(f"Error in conscious processing: {e}")
            traceback.print_exc()
            return self._generate_fallback_response(user_input, [], [])
    
    def _format_template_data(self, user_input: str, persona: Dict, emotional_state: Dict, 
                             memories: List[Dict], relationships: List[Dict], conversation_history: List[Dict]) -> Dict:
        """Format data to match the agent_response.mustache template structure."""
        
        # Extract persona information
        persona_name = persona.get('name', self.agent_name)
        persona_age = persona.get('age', '')
        persona_occupation = persona.get('occupation', '')
        personality_traits = persona.get('personality_traits', [])
        if not isinstance(personality_traits, list):
            personality_traits = [str(personality_traits)] if personality_traits else []
        
        interests = persona.get('interests', [])
        if not isinstance(interests, list):
            interests = [str(interests)] if interests else []
            
        background = persona.get('background', '')
        
        # Format emotional state
        emotion_category = emotional_state.get('emotion_category', 'neutral')
        pleasure = emotional_state.get('pleasure', 0.0)
        arousal = emotional_state.get('arousal', 0.0) 
        dominance = emotional_state.get('dominance', 0.0)
        
        # Format memories for template
        formatted_memories = []
        for memory in memories:
            if isinstance(memory, dict):
                # Extract the meaningful content from memory
                title = memory.get('title', '')
                description = memory.get('description', '')
                content = memory.get('content', '')
                
                # Create a readable memory string
                if title and description:
                    memory_text = f"{title}: {description}"
                elif title:
                    memory_text = title
                elif description:
                    memory_text = description
                elif content:
                    memory_text = content
                else:
                    memory_text = str(memory)
                    
                formatted_memories.append(memory_text)
            else:
                formatted_memories.append(str(memory))
        
        # Format relationships for template
        formatted_relationships = []
        for relationship in relationships:
            if isinstance(relationship, dict):
                name = relationship.get('name', 'Someone')
                rel_type = relationship.get('relationship_type', 'Unknown')
                significance = relationship.get('emotional_significance', '')
                
                # Create a readable relationship string
                if significance:
                    rel_text = f"{name} ({rel_type}): {significance}"
                else:
                    rel_text = f"{name} ({rel_type})"
                    
                formatted_relationships.append(rel_text)
            else:
                formatted_relationships.append(str(relationship))
        
        # Format conversation history
        formatted_conversation = []
        for entry in conversation_history[-5:]:  # Last 5 entries
            if isinstance(entry, dict):
                content = entry.get('content', '')
                context = entry.get('context', '')
                
                if context == 'user_interaction':
                    formatted_conversation.append({"user": content, "agent": ""})
                elif context == 'agent_response' and formatted_conversation:
                    formatted_conversation[-1]["agent"] = content
        
        # Remove incomplete conversation entries
        formatted_conversation = [conv for conv in formatted_conversation if conv.get("agent")]
        
        # Prepare template data matching your mustache template structure
        template_data = {
            "agent_name": self.agent_name,
            "user_message": user_input,
            
            # Persona section
            "persona_name": persona_name,
            "persona_age": persona_age if persona_age else None,
            "persona_occupation": persona_occupation if persona_occupation else None,
            "personality_traits": personality_traits if personality_traits else None,
            "interests": interests if interests else None,
            "background": background if background else None,
            
            # Emotional state
            "emotional_description": emotion_category,
            "emotional_pleasure": f"{pleasure:.2f}",
            "emotional_arousal": f"{arousal:.2f}",
            "emotional_dominance": f"{dominance:.2f}",
            
            # Memories
            "has_memories": len(formatted_memories) > 0,
            "memory_count": len(formatted_memories),
            "relevant_memories": formatted_memories,
            
            # Relationships  
            "has_relationships": len(formatted_relationships) > 0,
            "relationship_count": len(formatted_relationships),
            "relevant_relationships": formatted_relationships,
            
            # Conversation history
            "conversation_history": formatted_conversation if formatted_conversation else None
        }
        
        return template_data
    
    def _generate_fallback_response(self, user_input: str, memories: List = None, relationships: List = None) -> str:
        """Generate a fallback response when template processing fails."""
        persona = self.memory_manager.get_persona()
        agent_name = persona.get("name", self.agent_name)
        
        # Try to use memories and relationships in fallback
        if memories and len(memories) > 0:
            # Look for relevant information about the user's question
            user_lower = user_input.lower()
            for memory in memories:
                if isinstance(memory, dict):
                    title = memory.get('title', '').lower()
                    description = memory.get('description', '').lower()
                    if any(word in title or word in description for word in user_lower.split()):
                        return f"I remember something about that... {memory.get('title', '')}. {memory.get('description', '')[:100]}..."
        
        if relationships and len(relationships) > 0:
            # Look for people mentioned in the question
            user_lower = user_input.lower()
            for relationship in relationships:
                if isinstance(relationship, dict):
                    name = relationship.get('name', '').lower()
                    if name in user_lower:
                        rel_type = relationship.get('relationship_type', 'someone I know')
                        significance = relationship.get('emotional_significance', '')
                        return f"{relationship.get('name')} is {rel_type}. {significance}"
        
        # Generic fallback
        fallback_responses = [
            f"I'm {agent_name}, and I'm thinking about what you just said.",
            f"That's interesting. As {agent_name}, I'd like to understand more about that.",
            f"I hear you. Let me process that for a moment.",
            f"That reminds me of something, but I'm having trouble accessing that memory right now.",
            f"I'm {agent_name}, and I appreciate you sharing that with me."
        ]
        
        return random.choice(fallback_responses)