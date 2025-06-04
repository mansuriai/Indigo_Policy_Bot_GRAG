import os
from neo4j import GraphDatabase
import logging
from typing import Dict, List, Any

class AuraDBConnection:
    """Connection class specifically for Neo4j AuraDB"""
    
    def __init__(self):
        # AuraDB connection details
        # self.uri = os.getenv("NEO4J_URI", "neo4j+s://graphrag_indigo.databases.neo4j.io")
        self.uri = os.getenv("NEO4J_URI", "neo4j+s://b8524829.databases.neo4j.io")
        self.username = os.getenv("NEO4J_USERNAME", "neo4j")  # Usually "neo4j" for AuraDB
        self.password = os.getenv("NEO4J_PASSWORD", "PFuMe99J8r3D9n1k9L0L6jHshDo66lNHRPgraBmpzDo")  # Your AuraDB password
        
        if not self.password:
            raise ValueError("NEO4J_PASSWORD environment variable is required for AuraDB")
        
        self.driver = None
        self.connect()
    
    def connect(self):
        """Connect to AuraDB with proper error handling"""
        try:
            # Create driver with AuraDB-specific settings
            # Note: neo4j+s:// scheme already handles encryption, so no need for encrypted=True
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password),
                # AuraDB-specific configurations
                max_connection_lifetime=30 * 60,  # 30 minutes
                max_connection_pool_size=50,
                connection_acquisition_timeout=2 * 60  # 2 minutes
                # No encryption settings needed with neo4j+s:// scheme
            )
            
            # Test the connection
            self.verify_connectivity()
            print(f"✅ Successfully connected to AuraDB: {self.uri}")
            
        except Exception as e:
            print(f"❌ Failed to connect to AuraDB: {e}")
            self._print_troubleshooting_guide()
            raise
    
    def verify_connectivity(self):
        """Verify that the connection works"""
        with self.driver.session() as session:
            result = session.run("RETURN 'Connection successful' as message, datetime() as timestamp")
            record = result.single()
            return record["message"], record["timestamp"]
    
    def _print_troubleshooting_guide(self):
        """Print troubleshooting steps for AuraDB connection issues"""
        print("\n🔧 AuraDB Connection Troubleshooting:")
        print("1. Check your AuraDB instance is running (not paused)")
        print("2. Verify the URI matches exactly from your AuraDB console")
        print("3. Ensure your password is correct (case-sensitive)")
        print("4. Check if your IP is whitelisted (if IP restrictions are enabled)")
        print("5. Verify your AuraDB instance hasn't expired (free tier)")
        print(f"6. Current URI: {self.uri}")
        print(f"7. Current Username: {self.username}")
    
    def close(self):
        """Close the connection"""
        if self.driver:
            self.driver.close()
            print("Connection closed")
    
    def execute_query(self, query: str, parameters: Dict = None) -> List[Dict]:
        """Execute a Cypher query"""
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]
    
    def create_constraints_and_indexes(self):
        """Create basic constraints and indexes for GraphRAG"""
        constraints_and_indexes = [
            # Create constraint for unique entity IDs
            "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            
            # Create indexes for better performance
            "CREATE INDEX entity_name_index IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX entity_type_index IF NOT EXISTS FOR (e:Entity) ON (e.type)",
            "CREATE INDEX document_source_index IF NOT EXISTS FOR (d:Document) ON (d.source)",
        ]
        
        for query in constraints_and_indexes:
            try:
                self.execute_query(query)
                print(f"✅ Executed: {query}")
            except Exception as e:
                print(f"⚠️  Constraint/Index may already exist: {e}")


class GraphRAGAuraDB:
    """GraphRAG implementation using Neo4j AuraDB"""
    
    def __init__(self):
        self.db = AuraDBConnection()
        self.setup_schema()
    
    def setup_schema(self):
        """Set up the graph schema for GraphRAG"""
        print("Setting up GraphRAG schema...")
        self.db.create_constraints_and_indexes()
    
    def add_entity(self, entity_id: str, name: str, entity_type: str = "Entity", 
                  description: str = "", source: str = "", **properties):
        """Add an entity to the graph"""
        query = """
        MERGE (e:Entity {id: $entity_id})
        SET e.name = $name,
            e.type = $entity_type,
            e.description = $description,
            e.source = $source,
            e += $properties,
            e.created_at = datetime(),
            e.updated_at = datetime()
        RETURN e
        """
        
        parameters = {
            'entity_id': entity_id,
            'name': name,
            'entity_type': entity_type,
            'description': description,
            'source': source,
            'properties': properties
        }
        
        return self.db.execute_query(query, parameters)
    
    def add_relationship(self, source_id: str, target_id: str, rel_type: str, 
                        context: str = "", confidence: float = 1.0, **properties):
        """Add a relationship between entities"""
        query = f"""
        MATCH (source:Entity {{id: $source_id}})
        MATCH (target:Entity {{id: $target_id}})
        MERGE (source)-[r:{rel_type}]->(target)
        SET r.context = $context,
            r.confidence = $confidence,
            r += $properties,
            r.created_at = datetime(),
            r.updated_at = datetime()
        RETURN r
        """
        
        parameters = {
            'source_id': source_id,
            'target_id': target_id,
            'context': context,
            'confidence': confidence,
            'properties': properties
        }
        
        return self.db.execute_query(query, parameters)
    
    def find_related_entities(self, entity_id: str, max_depth: int = 2, limit: int = 20):
        """Find entities related to a given entity"""
        query = """
        MATCH (start:Entity {id: $entity_id})
        CALL {
            WITH start
            MATCH path = (start)-[*1..%d]-(related:Entity)
            RETURN related, length(path) as depth
            ORDER BY depth
            LIMIT $limit
        }
        RETURN related.id as id, related.name as name, related.type as type, 
               related.description as description, depth
        ORDER BY depth, related.name
        """ % max_depth
        
        return self.db.execute_query(query, {'entity_id': entity_id, 'limit': limit})
    
    def search_entities(self, search_text: str, limit: int = 10):
        """Search entities by text (name and description)"""
        query = """
        MATCH (e:Entity)
        WHERE e.name CONTAINS $search_text 
           OR e.description CONTAINS $search_text
        RETURN e.id as id, e.name as name, e.type as type, 
               e.description as description, e.source as source
        ORDER BY 
            CASE 
                WHEN e.name CONTAINS $search_text THEN 1 
                ELSE 2 
            END,
            e.name
        LIMIT $limit
        """
        
        return self.db.execute_query(query, {'search_text': search_text, 'limit': limit})
    
    def get_graph_stats(self):
        """Get basic statistics about the graph"""
        queries = {
            'total_entities': "MATCH (e:Entity) RETURN count(e) as count",
            'total_relationships': "MATCH ()-[r]->() RETURN count(r) as count",
            'entity_types': """
                MATCH (e:Entity) 
                RETURN e.type as type, count(e) as count 
                ORDER BY count DESC
            """,
            'relationship_types': """
                MATCH ()-[r]->() 
                RETURN type(r) as type, count(r) as count 
                ORDER BY count DESC
            """
        }
        
        stats = {}
        for stat_name, query in queries.items():
            try:
                result = self.db.execute_query(query)
                stats[stat_name] = result
            except Exception as e:
                stats[stat_name] = f"Error: {e}"
        
        return stats
    
    def close(self):
        """Close the database connection"""
        self.db.close()


# Environment setup helper
def setup_environment():
    """Helper function to set up environment variables"""
    print("🔧 AuraDB Environment Setup")
    print("You need to set these environment variables:")
    print()
    print("export NEO4J_URI='neo4j+s://graphrag_indigo.databases.neo4j.io'")
    print("export NEO4J_USERNAME='neo4j'")
    print("export NEO4J_PASSWORD='your_actual_password'")
    print()
    print("Or create a .env file with:")
    print("NEO4J_URI=neo4j+s://graphrag_indigo.databases.neo4j.io")
    print("NEO4J_USERNAME=neo4j")
    print("NEO4J_PASSWORD=your_actual_password")


# Usage example
if __name__ == "__main__":
    try:
        # Initialize GraphRAG with AuraDB
        graph_rag = GraphRAGAuraDB()
        
        # Test adding some entities
        graph_rag.add_entity("apple_inc", "Apple Inc.", "Company", 
                           "Technology company", "test_source")
        
        graph_rag.add_entity("steve_jobs", "Steve Jobs", "Person", 
                           "Co-founder of Apple Inc.", "test_source")
        
        # Add relationship
        graph_rag.add_relationship("steve_jobs", "apple_inc", "FOUNDED", 
                                 "Steve Jobs co-founded Apple Inc.")
        
        # Test search
        results = graph_rag.search_entities("Apple")
        print("Search results:", results)
        
        # Get stats
        stats = graph_rag.get_graph_stats()
        print("Graph stats:", stats)
        
        # Close connection
        graph_rag.close()
        
    except Exception as e:
        print(f"Error: {e}")
        setup_environment()