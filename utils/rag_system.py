import chromadb
import json
import os
import re
from datetime import datetime
from typing import List, Dict, Any, Optional

class RAGSystem:
    def __init__(self, agent_name: str, base_path: str = "base_agents"):
        """Initialize RAG system with enhanced error handling and debugging."""
        self.agent_name = agent_name
        self.agent_path = os.path.join(base_path, agent_name)
        self.vector_db_path = os.path.join(self.agent_path, "vector_db")
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=self.vector_db_path)
        
        # Initialize collections with error handling
        try:
            self.memories_collection = self.client.get_or_create_collection(
                name="memories",
                metadata={"hnsw:space": "cosine"}
            )
            self.relationships_collection = self.client.get_or_create_collection(
                name="relationships", 
                metadata={"hnsw:space": "cosine"}
            )
            print(f"RAG System initialized for {agent_name}")
            print(f"Memories collection: {self.memories_collection.count()} documents")
            print(f"Relationships collection: {self.relationships_collection.count()} documents")
        except Exception as e:
            print(f"Error initializing RAG collections: {e}")
            self.memories_collection = None
            self.relationships_collection = None

    def add_memories(self, memories: List[Dict[str, Any]]) -> bool:
        """Add memories to the vector database with enhanced error handling."""
        if not self.memories_collection:
            print("Error: Memories collection not initialized")
            return False
            
        try:
            documents = []
            metadatas = []
            ids = []
            
            for i, memory in enumerate(memories):
                # Prepare document content
                if isinstance(memory, dict):
                    # Create searchable text content
                    searchable_content = self._create_searchable_content(memory)
                    documents.append(json.dumps(memory, ensure_ascii=False))
                    
                    # Create metadata
                    metadata = {
                        "type": "memory",
                        "title": memory.get("title", f"Memory_{i}"),
                        "memory_type": memory.get("memory_type", "general"),
                        "emotions": ",".join(memory.get("emotions", [])),
                        "people": ",".join(memory.get("associated_people", [])),
                        "timestamp": datetime.now().isoformat(),
                        "searchable_content": searchable_content
                    }
                    metadatas.append(metadata)
                    ids.append(f"memory_{self.agent_name}_{i}_{datetime.now().timestamp()}")
                else:
                    # Handle non-dict memories
                    documents.append(str(memory))
                    metadatas.append({
                        "type": "memory",
                        "title": f"Memory_{i}",
                        "timestamp": datetime.now().isoformat()
                    })
                    ids.append(f"memory_{self.agent_name}_{i}_{datetime.now().timestamp()}")
            
            # Add to collection
            self.memories_collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            print(f"Successfully added {len(memories)} memories to vector database")
            return True
            
        except Exception as e:
            print(f"Error adding memories to vector database: {e}")
            return False

    def add_relationships(self, relationships: List[Dict[str, Any]]) -> bool:
        """Add relationships to the vector database with enhanced error handling."""
        if not self.relationships_collection:
            print("Error: Relationships collection not initialized")
            return False
            
        try:
            documents = []
            metadatas = []
            ids = []
            
            for i, relationship in enumerate(relationships):
                if isinstance(relationship, dict):
                    # Create searchable content
                    searchable_content = self._create_relationship_searchable_content(relationship)
                    documents.append(json.dumps(relationship, ensure_ascii=False))
                    
                    # Create metadata
                    metadata = {
                        "type": "relationship",
                        "name": relationship.get("name", f"Person_{i}"),
                        "relationship_type": relationship.get("relationship_type", "unknown"),
                        "emotional_significance": relationship.get("emotional_significance", ""),
                        "timestamp": datetime.now().isoformat(),
                        "searchable_content": searchable_content
                    }
                    metadatas.append(metadata)
                    ids.append(f"relationship_{self.agent_name}_{i}_{datetime.now().timestamp()}")
                else:
                    # Handle non-dict relationships
                    documents.append(str(relationship))
                    metadatas.append({
                        "type": "relationship",
                        "name": f"Person_{i}",
                        "timestamp": datetime.now().isoformat()
                    })
                    ids.append(f"relationship_{self.agent_name}_{i}_{datetime.now().timestamp()}")
            
            # Add to collection
            self.relationships_collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            print(f"Successfully added {len(relationships)} relationships to vector database")
            return True
            
        except Exception as e:
            print(f"Error adding relationships to vector database: {e}")
            return False

    def search_memories(self, query: str, n_results: int = 5, similarity_threshold: float = 0.2) -> List[Dict[str, Any]]:
        """Enhanced memory search with similarity thresholding, proper error handling, and debugging."""
        if not self.memories_collection:
            print("Warning: Memories collection not initialized")
            return []
            
        try:
            count = self.memories_collection.count()
            if count == 0:
                print("Warning: No memories found in collection")
                return []
            
            print(f"Searching {count} memories for query: '{query}' with n_results={n_results}")
            
            results = self.memories_collection.query(
                query_texts=[query],
                n_results=min(n_results, count), # Ensure n_results does not exceed available documents
                include=['documents', 'metadatas', 'distances']
            )
            
            processed_memories = []
            if results and 'documents' in results and results['documents']:
                documents = results['documents'][0] if results['documents'] else []
                metadatas = results['metadatas'][0] if results.get('metadatas') else []
                distances = results['distances'][0] if results.get('distances') else []
                
                print(f"Raw search returned {len(documents)} results")
                
                for i, doc in enumerate(documents):
                    try:
                        if doc.strip().startswith('{') or doc.strip().startswith('['):
                            memory = json.loads(doc)
                            if i < len(metadatas):
                                memory['_metadata'] = metadatas[i]
                            if i < len(distances):
                                memory['_similarity_score'] = 1 - distances[i]
                        else:
                            memory = {"content": doc, "type": "text_memory"}
                            if i < len(metadatas):
                                memory.update(metadatas[i])
                            if i < len(distances):
                                memory['_similarity_score'] = 1 - distances[i]
                        processed_memories.append(memory)
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error for document {i}: {e}")
                        memory = {"content": doc, "type": "malformed_memory", "error": str(e)}
                        if i < len(metadatas):
                            memory['_metadata'] = metadatas[i]
                        if i < len(distances):
                            memory['_similarity_score'] = 1 - distances[i]
                        processed_memories.append(memory)
            
            # Filter by similarity threshold
            filtered_memories = [mem for mem in processed_memories if mem.get('_similarity_score', 0.0) >= similarity_threshold]
            
            print(f"Successfully retrieved {len(processed_memories)} memories, filtered to {len(filtered_memories)} for query: '{query}' (threshold: {similarity_threshold})")
            
            if filtered_memories:
                print(f"Sample memory (post-filter): {filtered_memories[0].get('title', 'No title')} - Score: {filtered_memories[0].get('_similarity_score', 'N/A')}")
                
            return filtered_memories
            
        except Exception as e:
            print(f"Error searching memories: {e}")
            import traceback
            traceback.print_exc()
            return []

    def search_relationships(self, query: str, n_results: int = 5, similarity_threshold: float = 0.2) -> List[Dict[str, Any]]:
        """Enhanced relationship search with similarity thresholding, proper error handling, and debugging."""
        if not self.relationships_collection:
            print("Warning: Relationships collection not initialized")
            return []
            
        try:
            count = self.relationships_collection.count()
            if count == 0:
                print("Warning: No relationships found in collection")
                return []
                
            print(f"Searching {count} relationships for query: '{query}' with n_results={n_results}")
            
            results = self.relationships_collection.query(
                query_texts=[query],
                n_results=min(n_results, count), # Ensure n_results does not exceed available documents
                include=['documents', 'metadatas', 'distances']
            )
            
            processed_relationships = []
            if results and 'documents' in results and results['documents']:
                documents = results['documents'][0] if results['documents'] else []
                metadatas = results['metadatas'][0] if results.get('metadatas') else []
                distances = results['distances'][0] if results.get('distances') else []
                
                print(f"Raw search returned {len(documents)} results")
                
                for i, doc in enumerate(documents):
                    try:
                        if doc.strip().startswith('{') or doc.strip().startswith('['):
                            relationship = json.loads(doc)
                            if i < len(metadatas):
                                relationship['_metadata'] = metadatas[i]
                            if i < len(distances):
                                relationship['_similarity_score'] = 1 - distances[i]
                        else:
                            relationship = {"content": doc, "type": "text_relationship"}
                            if i < len(metadatas):
                                relationship.update(metadatas[i])
                            if i < len(distances):
                                relationship['_similarity_score'] = 1 - distances[i]
                        processed_relationships.append(relationship)
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error for relationship {i}: {e}")
                        relationship = {"content": doc, "type": "malformed_relationship", "error": str(e)}
                        if i < len(metadatas):
                            relationship['_metadata'] = metadatas[i]
                        if i < len(distances):
                            relationship['_similarity_score'] = 1 - distances[i]
                        processed_relationships.append(relationship)

            # Filter by similarity threshold
            filtered_relationships = [rel for rel in processed_relationships if rel.get('_similarity_score', 0.0) >= similarity_threshold]
                        
            print(f"Successfully retrieved {len(processed_relationships)} relationships, filtered to {len(filtered_relationships)} for query: '{query}' (threshold: {similarity_threshold})")
            
            if filtered_relationships:
                print(f"Sample relationship (post-filter): {filtered_relationships[0].get('name', 'No name')} - Score: {filtered_relationships[0].get('_similarity_score', 'N/A')}")
                
            return filtered_relationships
            
        except Exception as e:
            print(f"Error searching relationships: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _create_searchable_content(self, memory: Dict[str, Any]) -> str:
        """Create searchable text content from memory object."""
        searchable_parts = []
        
        # Add title and description
        if memory.get("title"):
            searchable_parts.append(memory["title"])
        if memory.get("description"):
            searchable_parts.append(memory["description"])
            
        # Add emotions
        if memory.get("emotions"):
            searchable_parts.append(" ".join(memory["emotions"]))
            
        # Add associated people
        if memory.get("associated_people"):
            searchable_parts.append(" ".join(memory["associated_people"]))
            
        # Add significance
        if memory.get("significance"):
            searchable_parts.append(memory["significance"])
            
        return " ".join(searchable_parts)

    def _create_relationship_searchable_content(self, relationship: Dict[str, Any]) -> str:
        """Create searchable text content from relationship object."""
        searchable_parts = []
        
        # Add name and relationship type
        if relationship.get("name"):
            searchable_parts.append(relationship["name"])
        if relationship.get("relationship_type"):
            searchable_parts.append(relationship["relationship_type"])
            
        # Add emotional significance
        if relationship.get("emotional_significance"):
            searchable_parts.append(relationship["emotional_significance"])
            
        # Add key interactions
        if relationship.get("key_interactions"):
            searchable_parts.extend(relationship["key_interactions"])
            
        # Add unresolved elements
        if relationship.get("unresolved_elements"):
            searchable_parts.extend(relationship["unresolved_elements"])
            
        return " ".join(searchable_parts)

    def get_collection_stats(self) -> Dict[str, int]:
        """Get statistics about the collections."""
        stats = {
            "memories_count": 0,
            "relationships_count": 0
        }
        
        try:
            if self.memories_collection:
                stats["memories_count"] = self.memories_collection.count()
            if self.relationships_collection:
                stats["relationships_count"] = self.relationships_collection.count()
        except Exception as e:
            print(f"Error getting collection stats: {e}")
            
        return stats

    def test_search(self, test_queries: Optional[List[str]] = None) -> None:
        """Test search functionality with sample queries."""
        if not test_queries:
            test_queries = [
                "family",
                "school",
                "friends",
                "work",
                "childhood",
                "relationships",
                "emotions",
                "fear",
                "love",
                "memories"
            ]
        
        print(f"\n=== Testing RAG System for {self.agent_name} ===")
        stats = self.get_collection_stats()
        print(f"Collections: {stats['memories_count']} memories, {stats['relationships_count']} relationships")
        
        for query in test_queries:
            print(f"\n--- Testing query: '{query}' ---")
            
            # Test memory search (will use default threshold 0.2)
            memories = self.search_memories(query, n_results=3)
            print(f"Memories found (post-filter): {len(memories)}")
            if memories:
                for i, mem in enumerate(memories[:2]): 
                    title = mem.get('title', mem.get('content', 'No title'))[:50]
                    score = mem.get('_similarity_score', 'N/A')
                    print(f"  {i+1}. {title}... (Score: {score})")
            
            # Test relationship search (will use default threshold 0.2)
            relationships = self.search_relationships(query, n_results=3)
            print(f"Relationships found (post-filter): {len(relationships)}")
            if relationships:
                for i, rel in enumerate(relationships[:2]): 
                    name = rel.get('name', rel.get('content', 'No name'))
                    rel_type = rel.get('relationship_type', 'Unknown')
                    score = rel.get('_similarity_score', 'N/A')
                    print(f"  {i+1}. {name} ({rel_type}) (Score: {score})")