import json
import re
from interfaces.llm_interface import LLMInterface 
from utils.file_utils import save_json 
from utils.rag_system import RAGSystem 

class ConsciousBuilder:
    def __init__(self, llm_interface=None):
        self.llm = llm_interface or LLMInterface()
    
    def extract_json_from_response(self, text):
        """
        Extract JSON from a response that might contain markdown or other text.
        Prioritizes markdown blocks, then attempts to find the first valid JSON
        object or array using raw_decode.
        """
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text, re.DOTALL)
        if json_match:
            json_str_candidate = json_match.group(1).strip()
            try:
                parsed_json = json.loads(json_str_candidate)
                print("Successfully parsed JSON from markdown block")
                return parsed_json
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON from markdown block: {e}. Content: '{json_str_candidate[:200]}...'")

        text_to_search = text.strip()
        decoder = json.JSONDecoder()
        
        potential_starts = []
        for i, char in enumerate(text_to_search):
            if char == '{' or char == '[':
                potential_starts.append(i)
        
        if not potential_starts:
            print("Warning: No '{' or '[' found for potential JSON start in the text. Could not parse JSON from LLM response.")
            return {"raw_text": text}

        for start_idx in sorted(potential_starts):
            substring_to_decode = text_to_search[start_idx:]
            try:
                parsed_json, _ = decoder.raw_decode(substring_to_decode)
                print(f"Successfully parsed JSON using raw_decode starting at text index {start_idx} (relative to stripped text).")
                return parsed_json
            except json.JSONDecodeError:
                pass 
        
        print("Warning: Could not parse JSON using markdown or raw_decode. Returning raw text for advanced extraction if applicable.")
        return {"raw_text": text}

    def extract_persona(self, dream_data):
        """Extract persona from dream data."""
        print("Extracting persona...")
        result = self.llm.generate("phase1", "extract_persona", {"dreams": dream_data})
        return self.extract_json_from_response(result)
    
    def extract_relationships(self, dream_data):
        """Extract relationships from dream data with improved parsing."""
        print("Extracting relationships...")
        result = self.llm.generate("phase1", "extract_relationships", {"dreams": dream_data})
        
        print(f"Raw LLM response length: {len(result)}")
        
        relationships_data = self.extract_json_from_response(result)
        
        if isinstance(relationships_data, list):
            print(f"✅ Successfully parsed {len(relationships_data)} relationships from JSON array")
            return relationships_data
        elif isinstance(relationships_data, dict):
            if "raw_text" in relationships_data:
                raw_text = relationships_data["raw_text"]
                print("🔍 Attempting advanced extraction from raw text (this may be slow)...")
                
                extracted_relationships = self._extract_relationships_from_raw_text_advanced(raw_text)
                if extracted_relationships:
                    print(f"✅ Extracted {len(extracted_relationships)} relationships from raw text using advanced methods")
                    return extracted_relationships
                else:
                    print("⚠️ Could not extract relationships using advanced methods from raw text, wrapping as single relationship")
                    return [{"raw_relationships": raw_text}]
            else:
                # Single relationship object
                return [relationships_data]
        else:
            print("❌ Unexpected relationship data format after initial parsing, wrapping in list")
            return [{"raw_relationships": str(relationships_data)}]
    
    def _extract_relationships_from_raw_text_advanced(self, raw_text):
        """Advanced extraction of relationships from raw text. (This method can be slow)"""
        try:
            print("Advanced Method 1: Looking for JSON arrays...")
            array_patterns = [
                r'\[\s*\{[\s\S]*?\}\s*(?:,\s*\{[\s\S]*?\}\s*)*\]',  # Array of objects
                r'\[[\s\S]*?\]'  # Any array
            ]
            
            for pattern in array_patterns:
                matches = re.findall(pattern, raw_text, re.DOTALL)
                for match in matches:
                    try:
                        parsed = json.loads(match)
                        if isinstance(parsed, list) and len(parsed) > 0:
                            valid_relationships = []
                            for item in parsed:
                                if isinstance(item, dict) and ('name' in item or 'relationship_type' in item):
                                    valid_relationships.append(item)
                            
                            if valid_relationships:
                                print(f"✅ Advanced Method 1 success: Found {len(valid_relationships)} valid relationships")
                                return valid_relationships
                                
                    except json.JSONDecodeError:
                        continue
            
            print("Advanced Method 2: Looking for individual relationship objects...")
            object_pattern = r'\{\s*"name"\s*:\s*"[^"]+",[\s\S]*?\}' # This can be slow
            matches = re.findall(object_pattern, raw_text, re.DOTALL) # Added DOTALL
            
            relationships = []
            for match_str in matches: 
                try:
                    parsed = json.loads(match_str) # Use match_str
                    if isinstance(parsed, dict) and 'name' in parsed:
                        relationships.append(parsed)
                        
                except json.JSONDecodeError as e:
                    continue
            
            if relationships:
                print(f"✅ Advanced Method 2 success: Found {len(relationships)} relationships")
                return relationships
            
            print("Advanced Method 3: Pattern-based extraction (heuristic)...")
            relationships = []
            name_pattern = r'"name"\s*:\s*"([^"]+)"'
            names = re.findall(name_pattern, raw_text)
            
            if names:
                for name in names:
                    temp_rel = {"name": name, "relationship_type": "Unknown_Heuristic", "extraction_method": "pattern_based_heuristic"}
                    
                    type_pattern = rf'"name"\s*:\s*"{re.escape(name)}"[^}}]*(?:\{{[^}}]*\}}[^}}]*)*"relationship_type"\s*:\s*"([^"]+)"'
                    type_match = re.search(type_pattern, raw_text, re.DOTALL)
                    if type_match:
                        temp_rel["relationship_type"] = type_match.group(1)
                    
                    relationships.append(temp_rel)
                
                if relationships:
                    print(f"✅ Advanced Method 3 success: Created {len(relationships)} relationships from patterns (heuristic)")
                    return relationships
            
            print("❌ All advanced extraction methods failed")
            return None
            
        except Exception as e:
            print(f"❌ Error in advanced relationship extraction: {e}")
            return None
    
    def extract_long_term_memory(self, dream_data):
        """Extract long-term memories from dream data."""
        print("Extracting long-term memories...")
        result = self.llm.generate("phase1", "extract_long_term_memory", {"dreams": dream_data})
        memories_data = self.extract_json_from_response(result)
        
        if isinstance(memories_data, list):
            print(f"Successfully parsed {len(memories_data)} memories from JSON array")
            return memories_data
        elif isinstance(memories_data, dict):
            if "raw_text" in memories_data:
                raw_text = memories_data["raw_text"]
                print("Attempting to extract JSON from raw text for memories (this may be slow)...")
                
                # Simplified extraction for memories from raw text
                array_matches = re.findall(r'\[[\s\S]*?\]', raw_text, re.DOTALL)
                for match in array_matches:
                    try:
                        parsed = json.loads(match)
                        if isinstance(parsed, list) and len(parsed) > 0:
                            print(f"Successfully extracted {len(parsed)} memories from raw text")
                            return parsed
                    except json.JSONDecodeError:
                        continue
                
                print("Could not extract JSON array of memories from raw text, wrapping as single memory")
                return [{"raw_memories": raw_text}]
            else:
                return [memories_data]
        else:
            print("Unexpected memory data format, wrapping in list")
            return [{"raw_memories": str(memories_data)}]
    
    def build_conscious_memory(self, dream_data, agent_name, agent_dir):
        """Build the conscious memory for an agent."""
        print(f"Building conscious memory for {agent_name}...")
        
        print("Step 1: Extracting persona...")
        persona = self.extract_persona(dream_data)
        print(f"Persona extracted: {persona.get('name', 'Unknown') if isinstance(persona, dict) else 'Raw Text'}")
        
        print("Step 2: Extracting relationships...")
        relationships = self.extract_relationships(dream_data)
        print(f"Relationships extracted: {len(relationships)} relationships")
        
        if relationships:
            for i, rel in enumerate(relationships[:3]):
                if isinstance(rel, dict):
                    name = rel.get('name', 'Unknown')
                    rel_type = rel.get('relationship_type', 'Unknown')
                    print(f"  Relationship {i+1}: {name} ({rel_type})")
                else:
                    print(f"  Relationship {i+1} (raw): {str(rel)[:50]}...")
        
        print("Step 3: Extracting long-term memories...")
        memories = self.extract_long_term_memory(dream_data)
        print(f"Memories extracted: {len(memories)} memories")

        if memories:
            for i, mem in enumerate(memories[:3]):
                if isinstance(mem, dict):
                    title = mem.get('title', 'No title')
                    print(f"  Memory {i+1}: {title}")
                else:
                    print(f"  Memory {i+1} (raw): {str(mem)[:50]}...")
        
        conscious = {
            "persona": persona,
            "relationships": relationships,
            "memories": memories,
            "short_term_memory": [],
            "current_state": "dormant"
        }
        
        conscious_path = f"{agent_dir}/conscious_memory.json"
        save_json(conscious, conscious_path)
        print(f"Conscious memory saved to {conscious_path}")
        
        print("Step 4: Creating RAG system for memories and relationships...")
        try:
            rag = RAGSystem(agent_name, "base_agents")
            
            valid_memories = self._validate_and_clean_memories(memories)
            valid_relationships = self._validate_and_clean_relationships(relationships)
            
            if valid_memories:
                print(f"Adding {len(valid_memories)} valid memories to RAG system...")
                if rag.add_memories(valid_memories):
                    print("✓ Memories successfully added to RAG system")
                else:
                    print("❌ Failed to add memories to RAG system")
            
            if valid_relationships:
                print(f"Adding {len(valid_relationships)} valid relationships to RAG system...")
                if rag.add_relationships(valid_relationships):
                    print("✓ Relationships successfully added to RAG system")
                else:
                    print("❌ Failed to add relationships to RAG system")
            
            stats = rag.get_collection_stats()
            print(f"RAG system final stats: {stats['memories_count']} memories, {stats['relationships_count']} relationships")
            if stats['memories_count'] == 0 and valid_memories: print("⚠ WARNING: Memories were provided but not added to RAG!")
            if stats['relationships_count'] == 0 and valid_relationships: print("⚠ WARNING: Relationships were provided but not added to RAG!")

        except Exception as e:
            print(f"❌ Error creating RAG system: {e}")
            import traceback
            traceback.print_exc()
        
        return conscious

    def _validate_and_clean_memories(self, memories):
        """Validate and clean memory data for RAG system."""
        valid_memories = []
        if not isinstance(memories, list): # Ensure memories is a list
            print(f"Warning: Memories data is not a list ({type(memories)}), attempting to wrap.")
            memories = [memories] if memories else []

        for i, memory in enumerate(memories):
            try:
                if isinstance(memory, dict):
                    if memory.get('title') or memory.get('description') or memory.get('content'):
                        cleaned_memory = {
                            "title": memory.get('title', f'Memory {i+1}'),
                            "description": memory.get('description', memory.get('content', '')),
                            "emotions": memory.get('emotions', []),
                            "associated_people": memory.get('associated_people', []),
                            "significance": memory.get('significance', ''),
                            "memory_type": memory.get('memory_type', 'general')
                        }
                        if not isinstance(cleaned_memory['emotions'], list): cleaned_memory['emotions'] = []
                        if not isinstance(cleaned_memory['associated_people'], list): cleaned_memory['associated_people'] = []
                        valid_memories.append(cleaned_memory)
                    elif memory.get('raw_memories') or memory.get('raw_text'):
                        raw_content = memory.get('raw_memories', memory.get('raw_text', ''))
                        valid_memories.append({
                            "title": f"Extracted Raw Memory {i+1}", "description": str(raw_content),
                            "emotions": [], "associated_people": [], "significance": "Extracted from raw content", "memory_type": "extracted_raw"
                        })
                elif isinstance(memory, str): 
                     valid_memories.append({
                        "title": f"Text Memory {i+1}", "description": memory,
                        "emotions": [], "associated_people": [], "significance": "Converted from string", "memory_type": "text_blob"
                    })
                else:
                    print(f"Skipping memory item {i} due to unexpected type: {type(memory)}")
            except Exception as e:
                print(f"Error processing memory {i}: {e}")
        print(f"Validated {len(valid_memories)} out of {len(memories)} initial memory items.")
        return valid_memories

    def _validate_and_clean_relationships(self, relationships):
        """Validate and clean relationship data for RAG system."""
        valid_relationships = []
        if not isinstance(relationships, list):
            print(f"Warning: Relationships data is not a list ({type(relationships)}), attempting to wrap.")
            relationships = [relationships] if relationships else []

        for i, relationship in enumerate(relationships):
            try:
                if isinstance(relationship, dict):
                    if relationship.get('name') or relationship.get('relationship_type'):
                        cleaned_relationship = {
                            "name": relationship.get('name', f'Person {i+1}'),
                            "relationship_type": relationship.get('relationship_type', relationship.get('relation', 'Unknown')),
                            "emotional_significance": relationship.get('emotional_significance', relationship.get('significance', '')),
                            "key_interactions": relationship.get('key_interactions', []),
                            "unresolved_elements": relationship.get('unresolved_elements', [])
                        }
                        if not isinstance(cleaned_relationship['key_interactions'], list): cleaned_relationship['key_interactions'] = []
                        if not isinstance(cleaned_relationship['unresolved_elements'], list): cleaned_relationship['unresolved_elements'] = []
                        valid_relationships.append(cleaned_relationship)
                    elif relationship.get('raw_relationships') or relationship.get('raw_text'):
                        raw_content = relationship.get('raw_relationships', relationship.get('raw_text', ''))
                        # Potentially call _extract_relationships_from_raw_text_advanced again, or simplify
                        # For now, creating a placeholder from raw content to avoid deep recursion here
                        valid_relationships.append({
                            "name": f"Extracted Raw Person {i+1}", "relationship_type": "Unknown_Raw",
                            "emotional_significance": str(raw_content)[:200], "key_interactions": [], "unresolved_elements": []
                        })
                elif isinstance(relationship, str): # Handle plain string relationships
                    valid_relationships.append({
                        "name": f"Text Relationship {i+1}", "relationship_type": "Unknown_Text",
                        "emotional_significance": relationship, "key_interactions": [], "unresolved_elements": []
                    })
                else:
                    print(f"Skipping relationship item {i} due to unexpected type: {type(relationship)}")
            except Exception as e:
                print(f"Error processing relationship {i}: {e}")
        print(f"Validated {len(valid_relationships)} out of {len(relationships)} initial relationship items.")
        return valid_relationships