# # utils/config.py
# import os

# from pathlib import Path
# from dotenv import load_dotenv
# from sentence_transformers import SentenceTransformer
# load_dotenv()

# class Config:
#     # Project structure
#     BASE_DIR = Path(__file__).parent.parent
#     DATA_DIR = BASE_DIR / "data"
#     DB_DIR = BASE_DIR / "storage" / "vectordb"
#     # MODEL_DIR = BASE_DIR / "models" / "all-mpnet-base-v2"   ####
    
#     # Create directories if they don't exist
#     DATA_DIR.mkdir(parents=True, exist_ok=True)
#     DB_DIR.mkdir(parents=True, exist_ok=True)
    
#     # SQL Database
#     DB_HOSTNAME = os.getenv("DB_HOSTNAME")
#     DB_NAME = os.getenv("DB_NAME")
#     DB_USERNAME = os.getenv("DB_USERNAME")
#     DB_PASSWORD = os.getenv("DB_PASSWORD")
#     DB_PORT = os.getenv("DB_PORT")

#     EMBEDDING_MODEL = "Snowflake/snowflake-arctic-embed-l-v2.0" 
#     # EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5" 
#     # model_name = "Snowflake/snowflake-arctic-embed-l-v2.0"
#     EMBEDDING_DIMENSION = 1024  # Adjust based on your specific embedding model
#     # EMBEDDING_MODEL = str(MODEL_DIR)
#     LLM_MODEL = "gpt-4.1-mini"
    
#     # Document processing
#     CHUNK_SIZE = 1000
#     CHUNK_OVERLAP = 200

#     # API Keys
#     OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
#     # Pinecone settings
#     PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
#     PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")
#     PINECONE_INDEX_NAME = "indigo-assistant2" #os.getenv("PINECONE_INDEX_NAME", "indigo-assistant")
    
#     # App settings
#     APP_TITLE = "GoAssist"
#     MAX_HISTORY_LENGTH = 8
    
#     # Vector DB settings
#     COLLECTION_NAME = "indigo-documents"
#     DISTANCE_METRIC = "cosine"
    
#     # Pinecone index settings
#     PINECONE_INDEX_SPEC = {
#         "cloud": "aws",
#         "region": "us-east-1",
#         "metric": "cosine"
#     }
    
#     # Web scraping settings
#     WEB_SCRAPING_DELAY = 1  # Delay between requests in seconds
    
# config = Config()


###################################

# utils/config.py - Updated for GraphRAG
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Project structure
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    DB_DIR = BASE_DIR / "storage" / "vectordb"
    
    # Create directories if they don't exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DB_DIR.mkdir(parents=True, exist_ok=True)
    
    # SQL Database
    DB_HOSTNAME = os.getenv("DB_HOSTNAME")
    DB_NAME = os.getenv("DB_NAME")
    DB_USERNAME = os.getenv("DB_USERNAME")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_PORT = os.getenv("DB_PORT")

    # Embedding settings
    EMBEDDING_MODEL = "Snowflake/snowflake-arctic-embed-l-v2.0"
    EMBEDDING_DIMENSION = 1024
    LLM_MODEL = "gpt-4.1"
    
    # Document processing
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200

    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Pinecone settings (kept for hybrid approach)
    # PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    # PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")
    # PINECONE_INDEX_NAME = "indigo-graphrag"
    
    # Neo4j Graph Database settings (NEW)
    # NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    # NEO4J_URI = os.getenv("NEO4J_URI", "neo4j+s://graphrag_indigo.databases.neo4j.io")
    NEO4J_URI = os.getenv("NEO4J_URI", "neo4j+s://b8524829.databases.neo4j.io")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
    NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
    
    # GraphRAG specific settings (NEW)
    ENTITY_EXTRACTION_MODEL = "en_core_web_sm"  # spaCy model
    MIN_ENTITY_CONFIDENCE = 0.7
    MAX_ENTITIES_PER_CHUNK = 20
    RELATIONSHIP_CONFIDENCE_THRESHOLD = 0.6
    GRAPH_TRAVERSAL_DEPTH = 2
    ENTITY_SIMILARITY_THRESHOLD = 0.85
    
    # Graph analysis settings (NEW)
    ENABLE_COMMUNITY_DETECTION = True
    COMMUNITY_ALGORITHM = "leiden"  # or "louvain"
    MIN_COMMUNITY_SIZE = 3
    GRAPH_EMBEDDING_DIMENSION = 128
    
    # Hybrid search settings (NEW)
    VECTOR_WEIGHT = 0.6  # Weight for vector similarity in hybrid search
    GRAPH_WEIGHT = 0.4   # Weight for graph-based similarity
    ENABLE_GRAPH_EXPANSION = True
    MAX_GRAPH_HOPS = 2
    
    # App settings
    APP_TITLE = "GoAssist GraphRAG"
    MAX_HISTORY_LENGTH = 8
    
    # Vector DB settings (kept for backward compatibility)
    COLLECTION_NAME = "indigo-documents"
    DISTANCE_METRIC = "cosine"
    
    # Pinecone index settings
    # PINECONE_INDEX_SPEC = {
    #     "cloud": "aws",
    #     "region": "us-east-1",
    #     "metric": "cosine"
    # }
    
    # Web scraping settings
    WEB_SCRAPING_DELAY = 1
    
    # GraphRAG processing settings (NEW)
    BATCH_SIZE_ENTITIES = 100
    BATCH_SIZE_RELATIONSHIPS = 50
    ENABLE_ENTITY_LINKING = True
    ENTITY_LINKING_THRESHOLD = 0.9
    
    # Performance optimization (NEW)
    ENABLE_CACHING = True
    CACHE_TTL = 3600  # 1 hour
    MAX_CACHE_SIZE = 1000
    ENABLE_PARALLEL_PROCESSING = True
    MAX_WORKERS = 4

config = Config()