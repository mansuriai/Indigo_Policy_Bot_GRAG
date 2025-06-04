# app/main.py - Updated for GraphRAG
import streamlit as st
from pathlib import Path
import time
from typing import List, Dict
import os, sys
from urllib.parse import urlencode

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.config import config

# Set page config as the first Streamlit command
st.set_page_config(
    page_title=config.APP_TITLE,
    layout="wide",
)

from core.embeddings import EmbeddingManager
from core.graph_store import GraphStore  # Updated import
from core.graph_llm import GraphLLMManager  # Updated import

def check_environment():
    """Check if all required environment variables are set."""
    missing_vars = []
    
    if not config.OPENAI_API_KEY:
        missing_vars.append("OPENAI_API_KEY")
    if not config.NEO4J_URI:
        missing_vars.append("NEO4J_URI")
    if not config.NEO4J_PASSWORD:
        missing_vars.append("NEO4J_PASSWORD")
    
    if missing_vars:
        error_msg = f"Missing required environment variables: {', '.join(missing_vars)}\n"
        error_msg += "Please ensure these variables are set in your .env file or environment."
        raise ValueError(error_msg)

def display_graph_insights(graph_summary: Dict):
    """Display graph insights in the sidebar."""
    if not graph_summary or not graph_summary.get('entities'):
        return
    
    with st.sidebar:
        st.subheader("🕸️ Knowledge Graph Insights")
        
        # Display top entities
        st.write("**Top Connected Entities:**")
        for i, entity in enumerate(graph_summary['entities'][:5], 1):
            st.write(f"{i}. **{entity['name']}** ({entity['type']})")
            st.caption(f"📄 {entity['document_count']} docs | 🔗 {entity['relationship_count']} connections")
        
        if graph_summary.get('topic'):
            st.write(f"**Topic Focus:** {graph_summary['topic']}")
        
        st.divider()

def display_sources(sources: List[Dict]):
    """Display sources with proper formatting and links."""
    if not sources:
        return
    
    with st.expander("📚 Source References & Graph Context", expanded=False):
        for i, source in enumerate(sources, 1):
            metadata = source.get('metadata', {})
            url = metadata.get('url', '')
            
            st.markdown(f"### Reference {i}")
            if url:
                st.markdown(f"[🔗 {metadata.get('source', 'Source')}]({url})")
            else:
                st.markdown(f"**{metadata.get('source', 'Source')}**")
            
            # Show enhanced score if available
            if 'enhanced_score' in source:
                st.caption(f"Relevance Score: {source['enhanced_score']:.3f}")
            
            # Show entity information if available
            entity_count = metadata.get('entity_count', 0)
            relationship_count = metadata.get('relationship_count', 0)
            if entity_count > 0:
                st.caption(f"🏷️ {entity_count} entities | 🔗 {relationship_count} relationships")
            
            # Show preview text
            preview_text = source['text'][:300] + "..." if len(source['text']) > 300 else source['text']
            st.caption(preview_text)
            st.divider()

@st.cache_resource
def initialize_components():
    try:
        # Check environment variables first
        check_environment()
        
        # Initialize components with better error handling
        try:
            embedding_manager = EmbeddingManager()
        except Exception as e:
            st.error(f"Embedding Manager Error: {str(e)}")
            return None
            
        try:
            graph_store = GraphStore()  # Updated to use GraphStore
        except Exception as e:
            st.error(f"Graph Store Error: {str(e)}")
            st.info("Make sure Neo4j is running and accessible at the configured URI.")
            return None
            
        try:
            llm_manager = GraphLLMManager()  # Updated to use GraphLLMManager
        except Exception as e:
            st.error(f"LLM Manager Error: {str(e)}")
            return None
        
        components = {
            'embedding_manager': embedding_manager,
            'graph_store': graph_store,
            'llm_manager': llm_manager
        }
        
        return components
    except Exception as e:
        st.error(f"Initialization Error: {str(e)}")
        st.info("Please check your .env file and ensure all required configuration is set correctly.")
        return None

components = initialize_components()

if components is None:
    st.stop()

embedding_manager = components['embedding_manager']
graph_store = components['graph_store']
llm_manager = components['llm_manager']

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_sources" not in st.session_state:
    st.session_state.current_sources = []
if "graph_summary" not in st.session_state:
    st.session_state.graph_summary = {}
if "context_window" not in st.session_state:
    st.session_state.context_window = 5
if "max_history" not in st.session_state:
    st.session_state.max_history = 10
if "show_sources" not in st.session_state:
    st.session_state.show_sources = False
if "use_graph_expansion" not in st.session_state:
    st.session_state.use_graph_expansion = True

st.title(config.APP_TITLE)

st.markdown("""
Get answers to all your IndiGo Airlines related queries with enhanced Graph-powered RAG, 
providing deeper insights through entity relationships and knowledge graph analysis.
""")


## commenting out the sidebar
# Sidebar controls for GraphRAG
# with st.sidebar:
#     st.header("🎛️ GraphRAG Controls")
    
#     st.session_state.context_window = st.slider(
#         "Context Window", 
#         min_value=3, 
#         max_value=15, 
#         value=st.session_state.context_window
#     )
    
#     st.session_state.use_graph_expansion = st.checkbox(
#         "Enable Graph Expansion", 
#         value=st.session_state.use_graph_expansion,
#         help="Use entity relationships to expand search results"
#     )
    
#     st.session_state.show_sources = st.checkbox(
#         "Show Source Details", 
#         value=st.session_state.show_sources
#     )
    
#     # Graph insights
#     if st.button("🔍 Analyze Knowledge Graph"):
#         with st.spinner("Analyzing knowledge graph..."):
#             st.session_state.graph_summary = graph_store.get_graph_summary()
    
#     # Display graph insights if available
#     display_graph_insights(st.session_state.graph_summary)

# Chat interface
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Display sources if enabled
if st.session_state.show_sources and st.session_state.current_sources:
    display_sources(st.session_state.current_sources)

# User input
user_input = st.chat_input("Ask me anything about IndiGo Airlines...")

# Enhanced query processing with GraphRAG
if user_input:
    # Add user message to chat history
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input
    })
    
    # Display user message
    with st.chat_message("user"):
        st.write(user_input)
    
    # Create a placeholder for the streaming response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        try:
            with st.spinner("Processing with GraphRAG..."):
                # Step 1: Extract entities from the query
                query_entities = llm_manager.extract_query_entities(user_input)
                
                # Step 2: Generate embedding for query
                query_embedding = embedding_manager.generate_embeddings([user_input])[0]
                
                # Step 3: Enhanced graph search
                if st.session_state.use_graph_expansion:
                    relevant_docs = graph_store.graph_search(
                        user_input,
                        query_embedding,
                        k=st.session_state.context_window,
                        entity_types=[e['type'] for e in query_entities] if query_entities else None
                    )
                else:
                    # Fallback to basic vector search
                    relevant_docs = graph_store.graph_search(
                        user_input,
                        query_embedding,
                        k=st.session_state.context_window
                    )
                
                # Step 4: Get entity context for query entities
                entity_context = {}
                if query_entities:
                    entity_names = [e['name'] for e in query_entities]
                    entity_context = graph_store.get_entity_context(entity_names)
                
                # Step 5: Get graph summary if not available
                if not st.session_state.graph_summary:
                    topic = None
                    # Try to infer topic from entities
                    if query_entities:
                        topic = query_entities[0]['name'] if query_entities else None
                    st.session_state.graph_summary = graph_store.get_graph_summary(topic)
            
            # Save the current sources for potential display
            st.session_state.current_sources = relevant_docs

            # Step 6: Generate enhanced response using graph context
            if entity_context and any(entity_context.values()):
                response = llm_manager.generate_graph_enhanced_response(
                    user_input,
                    relevant_docs,
                    entity_context,
                    relationships=[],  # Could be populated from graph_store if needed
                    chat_history=st.session_state.chat_history[-st.session_state.max_history:],
                    streaming_container=response_placeholder
                )
            else:
                # Use graph summary for broader context
                response = llm_manager.generate_response_with_graph_summary(
                    user_input,
                    relevant_docs,
                    st.session_state.graph_summary,
                    chat_history=st.session_state.chat_history[-st.session_state.max_history:],
                    streaming_container=response_placeholder
                )
            
            # Display the response
            response_placeholder.markdown(response)
            
            # Display sources separately
            if st.session_state.show_sources:
                display_sources(relevant_docs)

            # Update chat history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response
            })
            
        except Exception as e:
            st.error(f"An error occurred during GraphRAG processing: {str(e)}")
            st.error("Full error details:")
            st.exception(e)

# New conversation button
if st.button("🔄 New Conversation"):
    st.session_state.chat_history = []
    st.session_state.current_sources = []
    st.rerun()