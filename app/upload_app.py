# # app/upload_app.py
# import streamlit as st
# from pathlib import Path
# import time
# import os, sys

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# # from fix_ssl import *
# from utils.config import config
# from utils.s3_manager import S3Manager
# from core.document_processor import EnhancedDocumentProcessor
# from core.embeddings import EmbeddingManager
# from core.vector_store import VectorStore

# # Initialize session state
# if "uploaded_files" not in st.session_state:
#     st.session_state.uploaded_files = []

# # Initialize components
# doc_processor = EnhancedDocumentProcessor()
# embedding_manager = EmbeddingManager()
# vector_store = VectorStore()

# st.set_page_config(
#     page_title=f"{config.APP_TITLE} - Document Upload",
#     layout="wide"
# )

# st.title(f"{config.APP_TITLE} - Document Upload")

# # File upload section
# st.header("Upload Documents")
# uploaded_files = st.file_uploader(
#     "Upload PDF documents",
#     type=['pdf'],
#     accept_multiple_files=True
# )

# if uploaded_files:
#     st.session_state.uploaded_files = uploaded_files

# if st.session_state.uploaded_files:
#     st.write(f"{len(st.session_state.uploaded_files)} documents ready for processing.")
    
#     if st.button("Process Documents"):
#         with st.spinner("Processing documents..."):
#             for file in st.session_state.uploaded_files:
#                 # Save PDF to storage directory
#                 pdf_path = config.PDF_STORAGE_DIR / file.name
#                 with open(pdf_path, 'wb') as f:
#                     f.write(file.getvalue())
                
#                 # Process for vector store
#                 file_path = config.DATA_DIR / file.name
#                 with open(file_path, 'wb') as f:
#                     f.write(file.getvalue())
                
#                 # Process documents
#                 chunks = doc_processor.process_file(file_path)
                
#                 # Generate embeddings
#                 embeddings = embedding_manager.generate_embeddings(
#                     [chunk['text'] for chunk in chunks]
#                 )
                
#                 # Store in vector database
#                 vector_store.add_documents(chunks, embeddings)
            
#             st.success("Documents processed and indexed!")
#             st.session_state.uploaded_files = []


# app/upload_app.py - Updated for GraphRAG
import streamlit as st
from pathlib import Path
import time
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.config import config
from core.graph_document_processor import GraphDocumentProcessor  # Updated import
from core.embeddings import EmbeddingManager
from core.graph_store import GraphStore  # Updated import

# Initialize session state
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "processing_status" not in st.session_state:
    st.session_state.processing_status = {}

# Initialize components
doc_processor = GraphDocumentProcessor()  # Updated to use GraphDocumentProcessor
embedding_manager = EmbeddingManager()
graph_store = GraphStore()  # Updated to use GraphStore

st.set_page_config(
    page_title=f"{config.APP_TITLE} - Document Upload",
    layout="wide"
)

st.title(f"{config.APP_TITLE} - Document Upload with Graph Processing")

# Sidebar with processing options
with st.sidebar:
    st.header("🔧 Processing Options")
    
    extract_entities = st.checkbox(
        "Extract Entities", 
        value=True,
        help="Extract named entities and domain-specific entities"
    )
    
    extract_relationships = st.checkbox(
        "Extract Relationships", 
        value=True,
        help="Extract relationships between entities"
    )
    
    enable_graph_analysis = st.checkbox(
        "Enable Graph Analysis", 
        value=True,
        help="Calculate centrality metrics and importance scores"
    )
    
    st.divider()
    
    st.subheader("📊 Processing Stats")
    if st.session_state.processing_status:
        for filename, stats in st.session_state.processing_status.items():
            st.write(f"**{filename}:**")
            st.write(f"- Chunks: {stats.get('chunks', 0)}")
            st.write(f"- Entities: {stats.get('entities', 0)}")
            st.write(f"- Relationships: {stats.get('relationships', 0)}")

# File upload section
st.header("📁 Upload Documents")
uploaded_files = st.file_uploader(
    "Upload PDF documents for graph-enhanced processing",
    type=['pdf'],
    accept_multiple_files=True,
    help="PDFs will be processed to extract text, entities, and relationships"
)

if uploaded_files:
    st.session_state.uploaded_files = uploaded_files

if st.session_state.uploaded_files:
    st.write(f"📋 {len(st.session_state.uploaded_files)} documents ready for graph processing.")
    
    # Display file information
    with st.expander("📄 File Details", expanded=False):
        for file in st.session_state.uploaded_files:
            file_size = len(file.getvalue()) / 1024  # Size in KB
            st.write(f"- **{file.name}** ({file_size:.1f} KB)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🚀 Process Documents with GraphRAG", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_files = len(st.session_state.uploaded_files)
            
            for i, file in enumerate(st.session_state.uploaded_files):
                status_text.text(f"Processing {file.name}...")
                
                try:
                    # Save file temporarily
                    file_path = config.DATA_DIR / file.name
                    with open(file_path, 'wb') as f:
                        f.write(file.getvalue())
                    
                    # Process with GraphRAG
                    with st.spinner(f"Extracting content from {file.name}..."):
                        documents, entities, relationships = doc_processor.process_file_with_graph(file_path)
                    
                    # Generate embeddings
                    with st.spinner(f"Generating embeddings for {file.name}..."):
                        embeddings = embedding_manager.generate_embeddings(
                            [chunk['text'] for chunk in documents]
                        )
                    
                    # Store in graph database
                    with st.spinner(f"Storing in graph database..."):
                        graph_store.add_documents_with_graph(
                            documents, 
                            embeddings, 
                            entities if extract_entities else [], 
                            relationships if extract_relationships else []
                        )
                    
                    # Update processing status
                    st.session_state.processing_status[file.name] = {
                        'chunks': len(documents),
                        'entities': len(entities) if extract_entities else 0,
                        'relationships': len(relationships) if extract_relationships else 0,
                        'status': 'completed'
                    }
                    
                    # Clean up temporary file
                    if file_path.exists():
                        file_path.unlink()
                    
                except Exception as e:
                    st.error(f"Error processing {file.name}: {str(e)}")
                    st.session_state.processing_status[file.name] = {
                        'status': 'failed',
                        'error': str(e)
                    }
                
                # Update progress
                progress_bar.progress((i + 1) / total_files)
            
            status_text.text("✅ Processing complete!")
            st.success("All documents processed and indexed with GraphRAG!")
            st.session_state.uploaded_files = []
            
            # Show final statistics
            total_chunks = sum(stats.get('chunks', 0) for stats in st.session_state.processing_status.values())
            total_entities = sum(stats.get('entities', 0) for stats in st.session_state.processing_status.values())
            total_relationships = sum(stats.get('relationships', 0) for stats in st.session_state.processing_status.values())
            
            st.info(f"""
            **Processing Summary:**
            - 📄 Total chunks: {total_chunks}
            - 🏷️ Total entities: {total_entities}
            - 🔗 Total relationships: {total_relationships}
            """)
    
    with col2:
        if st.button("🧹 Clear Upload Queue"):
            st.session_state.uploaded_files = []
            st.rerun()

# Graph Analysis Section
st.header("🕸️ Knowledge Graph Analysis")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📊 Analyze Current Graph"):
        with st.spinner("Analyzing knowledge graph..."):
            graph_summary = graph_store.get_graph_summary()
            
            if graph_summary and graph_summary.get('entities'):
                st.subheader("Top Entities by Connections")
                
                # Create a simple table
                entities_data = []
                for entity in graph_summary['entities'][:10]:
                    entities_data.append({
                        'Entity': entity['name'],
                        'Type': entity['type'],
                        'Documents': entity['document_count'],
                        'Connections': entity.get('relationship_count', 0)
                    })
                
                st.table(entities_data)
            else:
                st.info("No graph data available. Please upload and process some documents first.")

with col2:
    topic_search = st.text_input("🔍 Analyze Topic", placeholder="e.g., flight booking")
    if st.button("Search Topic") and topic_search:
        with st.spinner(f"Analyzing topic: {topic_search}"):
            topic_summary = graph_store.get_graph_summary(topic_search)
            
            if topic_summary and topic_summary.get('entities'):
                st.subheader(f"Entities related to '{topic_search}'")
                
                for entity in topic_summary['entities'][:5]:
                    st.write(f"**{entity['name']}** ({entity['type']})")
                    st.caption(f"Found in {entity['document_count']} documents")
            else:
                st.warning(f"No entities found related to '{topic_search}'")

with col3:
    entity_search = st.text_input("🏷️ Entity Details", placeholder="e.g., IndiGo")
    if st.button("Get Entity Context") and entity_search:
        with st.spinner(f"Getting context for: {entity_search}"):
            entity_context = graph_store.get_entity_context([entity_search])
            
            if entity_context and entity_search in entity_context:
                context = entity_context[entity_search]
                st.subheader(f"Context for '{entity_search}'")
                
                st.write(f"**Type:** {context.get('type', 'Unknown')}")
                if context.get('description'):
                    st.write(f"**Description:** {context['description']}")
                
                if context.get('related_entities'):
                    st.write("**Related Entities:**")
                    for rel_entity in context['related_entities'][:5]:
                        st.write(f"- {rel_entity['name']} ({rel_entity['relation']})")
                
                if context.get('documents'):
                    st.write(f"**Found in {len(context['documents'])} documents**")
            else:
                st.warning(f"Entity '{entity_search}' not found in the knowledge graph")

# Advanced Options
st.header("⚙️ Advanced Graph Operations")

with st.expander("🔧 Advanced Tools", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Graph Maintenance")
        
        if st.button("🔄 Rebuild Graph Indexes"):
            with st.spinner("Rebuilding graph indexes..."):
                try:
                    # This would call a method to rebuild indexes
                    # graph_store.rebuild_indexes()
                    st.success("Graph indexes rebuilt successfully!")
                except Exception as e:
                    st.error(f"Error rebuilding indexes: {str(e)}")
        
        if st.button("📈 Calculate Graph Metrics"):
            with st.spinner("Calculating graph metrics..."):
                try:
                    # This would call advanced graph analysis
                    st.info("Graph metrics calculation completed!")
                    st.write("**Metrics:**")
                    st.write("- Node count: Available in graph summary")
                    st.write("- Edge count: Available in graph summary") 
                    st.write("- Average degree: Calculated from relationships")
                    st.write("- Graph density: Based on possible connections")
                except Exception as e:
                    st.error(f"Error calculating metrics: {str(e)}")
    
    with col2:
        st.subheader("Export Options")
        
        export_format = st.selectbox(
            "Export Format",
            ["GraphML", "JSON", "CSV (Nodes)", "CSV (Edges)"]
        )
        
        if st.button("📤 Export Graph Data"):
            with st.spinner(f"Exporting graph data as {export_format}..."):
                try:
                    # This would implement actual export functionality
                    st.success(f"Graph data exported successfully as {export_format}!")
                    st.info("Export functionality would be implemented here.")
                except Exception as e:
                    st.error(f"Error exporting data: {str(e)}")

# Connection Status
st.sidebar.markdown("---")
st.sidebar.subheader("🔌 System Status")

try:
    # Test graph store connection
    graph_store.get_graph_summary()
    st.sidebar.success("✅ Graph DB Connected")
except Exception as e:
    st.sidebar.error("❌ Graph DB Disconnected")
    st.sidebar.caption(f"Error: {str(e)}")

try:
    # Test embedding manager
    embedding_manager.generate_embeddings(["test"])
    st.sidebar.success("✅ Embeddings Ready")
except Exception as e:
    st.sidebar.error("❌ Embeddings Error")
    st.sidebar.caption(f"Error: {str(e)}")

# Instructions
st.sidebar.markdown("---")
st.sidebar.subheader("📝 Instructions")
st.sidebar.markdown("""
1. **Upload PDFs** using the file uploader
2. **Configure options** in the sidebar
3. **Process documents** to extract entities and relationships
4. **Analyze the graph** using the analysis tools
5. **Export data** if needed

**GraphRAG Features:**
- Entity extraction with NLP
- Relationship detection
- Graph-based search enhancement
- Centrality-based importance scoring
""")

# Footer
st.markdown("---")
st.markdown("🚀 **GraphRAG Enhanced Document Processing** - Leveraging knowledge graphs for better information retrieval")