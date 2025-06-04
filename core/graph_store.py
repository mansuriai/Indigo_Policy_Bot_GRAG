# # core/graph_store.py
# import neo4j
# from typing import List, Dict, Optional, Set, Tuple, Any
# import numpy as np
# import logging
# from utils.config import config
# import json
# import hashlib

# class GraphStore:
#     """Graph-based vector store using Neo4j for GraphRAG implementation."""
    
#     def __init__(self):
#         self.logger = logging.getLogger(__name__)
        
#         # Initialize Neo4j connection
#         self.driver = neo4j.GraphDatabase.driver(
#             config.NEO4J_URI,
#             auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD)
#         )
        
#         # Create indexes and constraints for better performance
#         self._create_graph_schema()
        
#     def _create_graph_schema(self):
#         """Create necessary indexes and constraints in Neo4j."""
#         with self.driver.session() as session:
#             # Create constraints
#             session.run("""
#                 CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE
#             """)
#             session.run("""
#                 CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE
#             """)
#             session.run("""
#                 CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE
#             """)
#             session.run("""
#                 CREATE CONSTRAINT IF NOT EXISTS FOR (t:Topic) REQUIRE t.id IS UNIQUE
#             """)
            
#             # Create indexes for better query performance
#             session.run("""
#                 CREATE INDEX IF NOT EXISTS FOR (c:Chunk) ON (c.embedding)
#             """)
#             session.run("""
#                 CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.type)
#             """)
    
#     def add_documents_with_graph(self, documents: List[Dict], embeddings: List[List[float]], 
#                                 entities: List[Dict], relationships: List[Dict]):
#         """Add documents with their graph structure to Neo4j."""
#         with self.driver.session() as session:
#             for doc, embedding in zip(documents, embeddings):
#                 # Create document node
#                 doc_id = doc['metadata'].get('chunk_id', str(hash(doc['text'])))
                
#                 session.run("""
#                     MERGE (d:Document {id: $doc_id})
#                     SET d.text = $text,
#                         d.source = $source,
#                         d.embedding = $embedding,
#                         d.metadata = $metadata,
#                         d.content_type = $content_type,
#                         d.page_num = $page_num
#                 """, {
#                     'doc_id': doc_id,
#                     'text': doc['text'],
#                     'source': doc['metadata'].get('source', ''),
#                     'embedding': embedding,
#                     'metadata': json.dumps(doc['metadata']),
#                     'content_type': doc['metadata'].get('content_type', 'text'),
#                     'page_num': doc['metadata'].get('page_num', 0)
#                 })
                
#                 # Add entities for this document
#                 doc_entities = [e for e in entities if e.get('doc_id') == doc_id]
#                 for entity in doc_entities:
#                     session.run("""
#                         MERGE (e:Entity {id: $entity_id})
#                         SET e.name = $name,
#                             e.type = $type,
#                             e.description = $description,
#                             e.confidence = $confidence
                        
#                         MERGE (d:Document {id: $doc_id})
#                         MERGE (d)-[:CONTAINS_ENTITY]->(e)
#                     """, {
#                         'entity_id': entity['id'],
#                         'name': entity['name'],
#                         'type': entity['type'],
#                         'description': entity.get('description', ''),
#                         'confidence': entity.get('confidence', 1.0),
#                         'doc_id': doc_id
#                     })
                
#                 # Add relationships for this document
#                 doc_relationships = [r for r in relationships if r.get('doc_id') == doc_id]
#                 for rel in doc_relationships:
#                     session.run("""
#                         MATCH (e1:Entity {id: $entity1_id})
#                         MATCH (e2:Entity {id: $entity2_id})
#                         MERGE (e1)-[r:RELATED_TO {type: $rel_type}]->(e2)
#                         SET r.strength = $strength,
#                             r.description = $description
#                     """, {
#                         'entity1_id': rel['entity1_id'],
#                         'entity2_id': rel['entity2_id'],
#                         'rel_type': rel['type'],
#                         'strength': rel.get('strength', 1.0),
#                         'description': rel.get('description', '')
#                     })
    
#     def graph_search(self, query: str, embedding: List[float], k: int = 3, 
#                     entity_types: List[str] = None) -> List[Dict]:
#         """Enhanced search using graph structure and vector similarity."""
#         with self.driver.session() as session:
#             # First, get vector-similar documents
#             vector_results = session.run("""
#                 MATCH (d:Document)
#                 WITH d, gds.similarity.cosine(d.embedding, $query_embedding) AS similarity
#                 WHERE similarity > 0.5
#                 RETURN d, similarity
#                 ORDER BY similarity DESC
#                 LIMIT $k
#             """, {
#                 'query_embedding': embedding,
#                 'k': k * 2  # Get more for graph expansion
#             })
            
#             document_ids = [record['d']['id'] for record in vector_results]
            
#             if not document_ids:
#                 return []
            
#             # Expand search using graph relationships
#             graph_expansion_query = """
#                 MATCH (d:Document)
#                 WHERE d.id IN $doc_ids
                
#                 // Get related documents through entities
#                 OPTIONAL MATCH (d)-[:CONTAINS_ENTITY]->(e:Entity)-[:RELATED_TO]-(re:Entity)<-[:CONTAINS_ENTITY]-(rd:Document)
                
#                 // Get documents with similar entities
#                 OPTIONAL MATCH (d)-[:CONTAINS_ENTITY]->(e:Entity)<-[:CONTAINS_ENTITY]-(sd:Document)
#                 WHERE sd.id <> d.id
                
#                 WITH DISTINCT d, rd, sd, 
#                      gds.similarity.cosine(d.embedding, $query_embedding) AS base_similarity
                
#                 // Calculate graph-enhanced score
#                 WITH d, base_similarity,
#                      CASE WHEN rd IS NOT NULL THEN 0.3 ELSE 0 END as relation_boost,
#                      CASE WHEN sd IS NOT NULL THEN 0.2 ELSE 0 END as entity_boost
                
#                 WITH d, (base_similarity + relation_boost + entity_boost) as enhanced_score
                
#                 RETURN d.id as id, d.text as text, d.metadata as metadata, 
#                        d.source as source, enhanced_score
#                 ORDER BY enhanced_score DESC
#                 LIMIT $final_k
#             """
            
#             # Handle entity type filtering
#             if entity_types:
#                 graph_expansion_query = graph_expansion_query.replace(
#                     "OPTIONAL MATCH (d)-[:CONTAINS_ENTITY]->(e:Entity)",
#                     f"OPTIONAL MATCH (d)-[:CONTAINS_ENTITY]->(e:Entity) WHERE e.type IN {entity_types}"
#                 )
            
#             final_results = session.run(graph_expansion_query, {
#                 'doc_ids': document_ids,
#                 'query_embedding': embedding,
#                 'final_k': k
#             })
            
#             processed_results = []
#             for record in final_results:
#                 metadata = json.loads(record['metadata']) if record['metadata'] else {}
#                 processed_results.append({
#                     'text': record['text'],
#                     'metadata': metadata,
#                     'enhanced_score': record['enhanced_score'],
#                     'distance': 1 - record['enhanced_score']
#                 })
            
#             return processed_results
    
#     def get_entity_context(self, entity_names: List[str]) -> Dict[str, Any]:
#         """Get rich context for specific entities including their relationships."""
#         with self.driver.session() as session:
#             result = session.run("""
#                 MATCH (e:Entity)
#                 WHERE e.name IN $entity_names
                
#                 // Get related entities
#                 OPTIONAL MATCH (e)-[r:RELATED_TO]-(re:Entity)
                
#                 // Get documents containing these entities
#                 OPTIONAL MATCH (e)<-[:CONTAINS_ENTITY]-(d:Document)
                
#                 RETURN e.name as entity_name, e.type as entity_type, e.description as description,
#                        collect(DISTINCT {name: re.name, type: re.type, relation: r.type}) as related_entities,
#                        collect(DISTINCT {id: d.id, text: d.text[0..200]}) as documents
#             """, {'entity_names': entity_names})
            
#             context = {}
#             for record in result:
#                 context[record['entity_name']] = {
#                     'type': record['entity_type'],
#                     'description': record['description'],
#                     'related_entities': record['related_entities'],
#                     'documents': record['documents']
#                 }
            
#             return context
    
#     def get_graph_summary(self, topic: str = None) -> Dict[str, Any]:
#         """Get a summary of the knowledge graph for a specific topic or overall."""
#         with self.driver.session() as session:
#             if topic:
#                 # Topic-specific summary
#                 result = session.run("""
#                     MATCH (d:Document)
#                     WHERE d.text CONTAINS $topic OR d.source CONTAINS $topic
                    
#                     MATCH (d)-[:CONTAINS_ENTITY]->(e:Entity)
                    
#                     WITH e, count(d) as doc_count
                    
#                     MATCH (e)-[r:RELATED_TO]-(re:Entity)
                    
#                     RETURN e.name as entity, e.type as type, doc_count,
#                            count(r) as relationship_count,
#                            collect(DISTINCT re.name)[0..5] as top_related
#                     ORDER BY doc_count DESC, relationship_count DESC
#                     LIMIT 10
#                 """, {'topic': topic})
#             else:
#                 # Overall graph summary
#                 result = session.run("""
#                     MATCH (e:Entity)
#                     OPTIONAL MATCH (e)-[r:RELATED_TO]-(re:Entity)
#                     OPTIONAL MATCH (e)<-[:CONTAINS_ENTITY]-(d:Document)
                    
#                     WITH e, count(DISTINCT r) as rel_count, count(DISTINCT d) as doc_count
                    
#                     RETURN e.name as entity, e.type as type, rel_count, doc_count
#                     ORDER BY rel_count DESC, doc_count DESC
#                     LIMIT 20
#                 """)
            
#             summary = {
#                 'entities': [],
#                 'total_relationships': 0,
#                 'topic': topic
#             }
            
#             for record in result:
#                 summary['entities'].append({
#                     'name': record['entity'],
#                     'type': record['type'],
#                     'document_count': record['doc_count'],
#                     'relationship_count': record.get('relationship_count', record.get('rel_count', 0)),
#                     'related_entities': record.get('top_related', [])
#                 })
            
#             return summary
    
#     def close(self):
#         """Close the Neo4j driver."""
#         self.driver.close()






######################################

# core/graph_store.py - Fixed version without GDS dependency
import neo4j
from typing import List, Dict, Optional, Set, Tuple, Any
import numpy as np
import logging
from utils.config import config
import json
import hashlib

class GraphStore:
    """Graph-based vector store using Neo4j for GraphRAG implementation."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize Neo4j connection
        self.driver = neo4j.GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD)
        )
        
        # Create indexes and constraints for better performance
        self._create_graph_schema()
        
    def _create_graph_schema(self):
        """Create necessary indexes and constraints in Neo4j."""
        with self.driver.session() as session:
            # Create constraints
            session.run("""
                CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE
            """)
            session.run("""
                CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE
            """)
            session.run("""
                CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE
            """)
            session.run("""
                CREATE CONSTRAINT IF NOT EXISTS FOR (t:Topic) REQUIRE t.id IS UNIQUE
            """)
            
            # Create indexes for better query performance
            session.run("""
                CREATE INDEX IF NOT EXISTS FOR (d:Document) ON (d.source)
            """)
            session.run("""
                CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.type)
            """)
            session.run("""
                CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.name)
            """)

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            vec1 = np.array(vec1)
            vec2 = np.array(vec2)
            
            dot_product = np.dot(vec1, vec2)
            norm_vec1 = np.linalg.norm(vec1)
            norm_vec2 = np.linalg.norm(vec2)
            
            if norm_vec1 == 0 or norm_vec2 == 0:
                return 0.0
            
            return dot_product / (norm_vec1 * norm_vec2)
        except Exception as e:
            self.logger.warning(f"Error calculating cosine similarity: {e}")
            return 0.0
    
    def add_documents_with_graph(self, documents: List[Dict], embeddings: List[List[float]], 
                                entities: List[Dict], relationships: List[Dict]):
        """Add documents with their graph structure to Neo4j."""
        with self.driver.session() as session:
            for doc, embedding in zip(documents, embeddings):
                # Create document node
                doc_id = doc['metadata'].get('chunk_id', str(hash(doc['text'])))
                
                session.run("""
                    MERGE (d:Document {id: $doc_id})
                    SET d.text = $text,
                        d.source = $source,
                        d.embedding = $embedding,
                        d.metadata = $metadata,
                        d.content_type = $content_type,
                        d.page_num = $page_num
                """, {
                    'doc_id': doc_id,
                    'text': doc['text'],
                    'source': doc['metadata'].get('source', ''),
                    'embedding': embedding,
                    'metadata': json.dumps(doc['metadata']),
                    'content_type': doc['metadata'].get('content_type', 'text'),
                    'page_num': doc['metadata'].get('page_num', 0)
                })
                
                # Add entities for this document
                doc_entities = [e for e in entities if e.get('doc_id') == doc_id]
                for entity in doc_entities:
                    session.run("""
                        MERGE (e:Entity {id: $entity_id})
                        SET e.name = $name,
                            e.type = $type,
                            e.description = $description,
                            e.confidence = $confidence
                        
                        MERGE (d:Document {id: $doc_id})
                        MERGE (d)-[:CONTAINS_ENTITY]->(e)
                    """, {
                        'entity_id': entity['id'],
                        'name': entity['name'],
                        'type': entity['type'],
                        'description': entity.get('description', ''),
                        'confidence': entity.get('confidence', 1.0),
                        'doc_id': doc_id
                    })
                
                # Add relationships for this document
                doc_relationships = [r for r in relationships if r.get('doc_id') == doc_id]
                for rel in doc_relationships:
                    session.run("""
                        MATCH (e1:Entity {id: $entity1_id})
                        MATCH (e2:Entity {id: $entity2_id})
                        MERGE (e1)-[r:RELATED_TO {type: $rel_type}]->(e2)
                        SET r.strength = $strength,
                            r.description = $description
                    """, {
                        'entity1_id': rel['entity1_id'],
                        'entity2_id': rel['entity2_id'],
                        'rel_type': rel['type'],
                        'strength': rel.get('strength', 1.0),
                        'description': rel.get('description', '')
                    })
    
    def graph_search(self, query: str, embedding: List[float], k: int = 3, 
                    entity_types: List[str] = None) -> List[Dict]:
        """Enhanced search using graph structure and vector similarity."""
        with self.driver.session() as session:
            # First, get all documents with their embeddings
            all_docs_query = """
                MATCH (d:Document)
                WHERE d.embedding IS NOT NULL
                RETURN d.id as id, d.text as text, d.metadata as metadata, 
                       d.source as source, d.embedding as embedding
            """
            
            all_docs_result = session.run(all_docs_query)
            
            # Calculate similarities in Python
            doc_similarities = []
            for record in all_docs_result:
                try:
                    doc_embedding = record['embedding']
                    if doc_embedding:
                        similarity = self._cosine_similarity(embedding, doc_embedding)
                        if similarity > 0.1:  # Minimum threshold
                            doc_similarities.append({
                                'id': record['id'],
                                'text': record['text'],
                                'metadata': record['metadata'],
                                'source': record['source'],
                                'similarity': similarity
                            })
                except Exception as e:
                    self.logger.warning(f"Error processing document {record.get('id', 'unknown')}: {e}")
                    continue
            
            # Sort by similarity and get top candidates for graph expansion
            doc_similarities.sort(key=lambda x: x['similarity'], reverse=True)
            top_docs = doc_similarities[:k * 2]  # Get more for graph expansion
            
            if not top_docs:
                return []
            
            document_ids = [doc['id'] for doc in top_docs]
            
            # Build entity type filter for Cypher query
            entity_filter = ""
            if entity_types:
                entity_filter = "AND e.type IN $entity_types"
            
            # Expand search using graph relationships
            graph_expansion_query = f"""
                MATCH (d:Document)
                WHERE d.id IN $doc_ids
                
                // Get related documents through entities
                OPTIONAL MATCH (d)-[:CONTAINS_ENTITY]->(e:Entity)
                WHERE 1=1 {entity_filter}
                
                OPTIONAL MATCH (e)-[:RELATED_TO]-(re:Entity)<-[:CONTAINS_ENTITY]-(rd:Document)
                WHERE rd.id <> d.id
                
                // Get documents with similar entities
                OPTIONAL MATCH (e)<-[:CONTAINS_ENTITY]-(sd:Document)
                WHERE sd.id <> d.id
                
                WITH DISTINCT d, rd, sd, count(DISTINCT e) as entity_count,
                     count(DISTINCT re) as related_entity_count
                
                RETURN d.id as id, d.text as text, d.metadata as metadata, 
                       d.source as source, entity_count, related_entity_count,
                       CASE WHEN rd IS NOT NULL THEN 1 ELSE 0 END as has_related_docs,
                       CASE WHEN sd IS NOT NULL THEN 1 ELSE 0 END as has_similar_entities
            """
            
            query_params = {
                'doc_ids': document_ids
            }
            if entity_types:
                query_params['entity_types'] = entity_types
            
            graph_results = session.run(graph_expansion_query, query_params)
            
            # Process results and calculate enhanced scores
            processed_results = []
            doc_similarity_map = {doc['id']: doc['similarity'] for doc in top_docs}
            
            for record in graph_results:
                doc_id = record['id']
                base_similarity = doc_similarity_map.get(doc_id, 0.0)
                
                # Calculate graph enhancement boosts
                relation_boost = 0.3 if record['has_related_docs'] else 0.0
                entity_boost = 0.2 if record['has_similar_entities'] else 0.0
                entity_density_boost = min(0.1, record['entity_count'] * 0.05)
                
                enhanced_score = base_similarity + relation_boost + entity_boost + entity_density_boost
                
                try:
                    metadata = json.loads(record['metadata']) if record['metadata'] else {}
                except (json.JSONDecodeError, TypeError):
                    metadata = {}
                
                # Add graph metadata
                metadata['entity_count'] = record['entity_count']
                metadata['relationship_count'] = record['related_entity_count']
                
                processed_results.append({
                    'text': record['text'],
                    'metadata': metadata,
                    'enhanced_score': enhanced_score,
                    'distance': 1 - enhanced_score
                })
            
            # Sort by enhanced score and return top k
            processed_results.sort(key=lambda x: x['enhanced_score'], reverse=True)
            return processed_results[:k]
    
    def get_entity_context(self, entity_names: List[str]) -> Dict[str, Any]:
        """Get rich context for specific entities including their relationships."""
        if not entity_names:
            return {}
            
        with self.driver.session() as session:
            result = session.run("""
                MATCH (e:Entity)
                WHERE e.name IN $entity_names
                
                // Get related entities
                OPTIONAL MATCH (e)-[r:RELATED_TO]-(re:Entity)
                
                // Get documents containing these entities
                OPTIONAL MATCH (e)<-[:CONTAINS_ENTITY]-(d:Document)
                
                RETURN e.name as entity_name, e.type as entity_type, e.description as description,
                       collect(DISTINCT {name: re.name, type: re.type, relation: r.type, strength: r.strength}) as related_entities,
                       collect(DISTINCT {id: d.id, text: d.text[0..200], source: d.source}) as documents
            """, {'entity_names': entity_names})
            
            context = {}
            for record in result:
                # Filter out null related entities
                related_entities = [rel for rel in record['related_entities'] if rel['name'] is not None]
                documents = [doc for doc in record['documents'] if doc['id'] is not None]
                
                context[record['entity_name']] = {
                    'type': record['entity_type'],
                    'description': record['description'] or '',
                    'related_entities': related_entities,
                    'documents': documents
                }
            
            return context
    
    def get_graph_summary(self, topic: str = None) -> Dict[str, Any]:
        """Get a summary of the knowledge graph for a specific topic or overall."""
        with self.driver.session() as session:
            if topic:
                # Topic-specific summary
                result = session.run("""
                    MATCH (d:Document)
                    WHERE toLower(d.text) CONTAINS toLower($topic) OR toLower(d.source) CONTAINS toLower($topic)
                    
                    MATCH (d)-[:CONTAINS_ENTITY]->(e:Entity)
                    
                    WITH e, count(DISTINCT d) as doc_count
                    
                    OPTIONAL MATCH (e)-[r:RELATED_TO]-(re:Entity)
                    
                    WITH e, doc_count, count(DISTINCT r) as relationship_count,
                         collect(DISTINCT re.name)[0..5] as top_related
                    
                    RETURN e.name as entity, e.type as type, doc_count,
                           relationship_count, top_related
                    ORDER BY doc_count DESC, relationship_count DESC
                    LIMIT 10
                """, {'topic': topic})
            else:
                # Overall graph summary
                result = session.run("""
                    MATCH (e:Entity)
                    OPTIONAL MATCH (e)-[r:RELATED_TO]-(re:Entity)
                    OPTIONAL MATCH (e)<-[:CONTAINS_ENTITY]-(d:Document)
                    
                    WITH e, count(DISTINCT r) as rel_count, count(DISTINCT d) as doc_count
                    
                    RETURN e.name as entity, e.type as type, rel_count as relationship_count, doc_count
                    ORDER BY relationship_count DESC, doc_count DESC
                    LIMIT 20
                """)
            
            summary = {
                'entities': [],
                'total_relationships': 0,
                'topic': topic
            }
            
            total_relationships = 0
            for record in result:
                entity_data = {
                    'name': record['entity'],
                    'type': record['type'],
                    'document_count': record['doc_count'],
                    'relationship_count': record['relationship_count'],
                    'related_entities': record.get('top_related', [])
                }
                summary['entities'].append(entity_data)
                total_relationships += record['relationship_count']
            
            summary['total_relationships'] = total_relationships
            return summary
    
    def get_database_stats(self) -> Dict[str, int]:
        """Get basic statistics about the graph database."""
        with self.driver.session() as session:
            stats = {}
            
            # Count documents
            result = session.run("MATCH (d:Document) RETURN count(d) as count")
            stats['documents'] = result.single()['count']
            
            # Count entities
            result = session.run("MATCH (e:Entity) RETURN count(e) as count")
            stats['entities'] = result.single()['count']
            
            # Count relationships
            result = session.run("MATCH ()-[r:RELATED_TO]->() RETURN count(r) as count")
            stats['relationships'] = result.single()['count']
            
            # Count contains relationships
            result = session.run("MATCH ()-[r:CONTAINS_ENTITY]->() RETURN count(r) as count")
            stats['document_entity_links'] = result.single()['count']
            
            return stats
    
    def close(self):
        """Close the Neo4j driver."""
        if self.driver:
            self.driver.close()