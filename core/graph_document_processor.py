# core/graph_document_processor.py
from typing import List, Dict, Union, Optional, Tuple
from pathlib import Path
import PyPDF2
import pdfplumber
import pandas as pd
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
import spacy
import networkx as nx
from collections import defaultdict
import re
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.config import config
from utils.helpers import generate_document_id
from langchain_huggingface import HuggingFaceEmbeddings

class GraphDocumentProcessor:
    """Enhanced document processor that extracts entities and relationships for GraphRAG."""
    
    def __init__(self):
        # Load spaCy model for NER
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise RuntimeError("Please install the spaCy English model: python -m spacy download en_core_web_sm")
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
        )
        
        # Define domain-specific entity types for IndiGo Airlines
        self.domain_entities = {
            'FLIGHT_SERVICE': ['flight', 'booking', 'check-in', 'baggage', 'seat selection'],
            'FARE_TYPE': ['6E', 'student discount', 'armed forces', 'business'],
            'LOCATION': ['airport', 'terminal', 'gate', 'destination'],
            'POLICY': ['cancellation', 'refund', 'change fee', 'policy'],
            'PRODUCT': ['IndiGo', '6E Connect', 'Fast Forward', 'seat'],
            'OFFER': ['discount', 'cashback', 'promotion', 'deal', 'offer']
        }
    
    def _extract_entities_with_nlp(self, text: str, doc_id: str) -> List[Dict]:
        """Extract entities using spaCy NLP and domain-specific rules."""
        doc = self.nlp(text)
        entities = []
        
        # Extract named entities
        for ent in doc.ents:
            if len(ent.text.strip()) > 2:  # Filter out very short entities
                entities.append({
                    'id': f"{doc_id}_{ent.start}_{ent.end}",
                    'name': ent.text.strip(),
                    'type': ent.label_,
                    'start': ent.start,
                    'end': ent.end,
                    'confidence': 1.0,
                    'doc_id': doc_id,
                    'description': f"{ent.label_} entity found in document"
                })
        
        # Extract domain-specific entities
        text_lower = text.lower()
        for entity_type, keywords in self.domain_entities.items():
            for keyword in keywords:
                if keyword in text_lower:
                    # Find all occurrences
                    for match in re.finditer(re.escape(keyword), text_lower):
                        entities.append({
                            'id': f"{doc_id}_{entity_type}_{match.start()}",
                            'name': keyword,
                            'type': entity_type,
                            'start': match.start(),
                            'end': match.end(),
                            'confidence': 0.8,
                            'doc_id': doc_id,
                            'description': f"Domain-specific {entity_type} entity"
                        })
        
        return entities
    
    def _extract_relationships(self, text: str, entities: List[Dict], doc_id: str) -> List[Dict]:
        """Extract relationships between entities using dependency parsing and rules."""
        doc = self.nlp(text)
        relationships = []
        
        # Create entity position mapping
        entity_positions = {}
        for entity in entities:
            for i in range(entity['start'], entity['end']):
                entity_positions[i] = entity
        
        # Extract relationships using dependency parsing
        for token in doc:
            if token.pos_ in ['VERB', 'AUX'] and not token.is_stop:
                # Find entities connected by this verb
                connected_entities = []
                
                # Check subject
                for child in token.children:
                    if child.dep_ in ['nsubj', 'nsubjpass']:
                        for i in range(child.idx, child.idx + len(child.text)):
                            if i in entity_positions:
                                connected_entities.append(entity_positions[i])
                                break
                
                # Check objects
                for child in token.children:
                    if child.dep_ in ['dobj', 'pobj', 'attr']:
                        for i in range(child.idx, child.idx + len(child.text)):
                            if i in entity_positions:
                                connected_entities.append(entity_positions[i])
                                break
                
                # Create relationships between connected entities
                if len(connected_entities) >= 2:
                    for i, ent1 in enumerate(connected_entities):
                        for ent2 in connected_entities[i+1:]:
                            if ent1['id'] != ent2['id']:
                                relationships.append({
                                    'entity1_id': ent1['id'],
                                    'entity2_id': ent2['id'],
                                    'type': 'CONNECTED_BY_' + token.lemma_.upper(),
                                    'strength': 0.7,
                                    'description': f"Connected by verb: {token.text}",
                                    'doc_id': doc_id
                                })
        
        # Add domain-specific relationship rules
        relationships.extend(self._extract_domain_relationships(entities, doc_id))
        
        return relationships
    
    def _extract_domain_relationships(self, entities: List[Dict], doc_id: str) -> List[Dict]:
        """Extract domain-specific relationships for airline content."""
        relationships = []
        
        # Group entities by type
        entities_by_type = defaultdict(list)
        for entity in entities:
            entities_by_type[entity['type']].append(entity)
        
        # Define relationship rules
        relationship_rules = [
            ('FLIGHT_SERVICE', 'FARE_TYPE', 'APPLIES_TO'),
            ('OFFER', 'FARE_TYPE', 'AVAILABLE_FOR'),
            ('POLICY', 'FLIGHT_SERVICE', 'GOVERNS'),
            ('LOCATION', 'FLIGHT_SERVICE', 'LOCATION_OF'),
            ('PRODUCT', 'FLIGHT_SERVICE', 'PROVIDES')
        ]
        
        # Apply rules
        for type1, type2, rel_type in relationship_rules:
            for ent1 in entities_by_type[type1]:
                for ent2 in entities_by_type[type2]:
                    if ent1['id'] != ent2['id']:
                        relationships.append({
                            'entity1_id': ent1['id'],
                            'entity2_id': ent2['id'],
                            'type': rel_type,
                            'strength': 0.6,
                            'description': f"Domain relationship: {type1} {rel_type} {type2}",
                            'doc_id': doc_id
                        })
        
        return relationships
    
    def _create_knowledge_graph(self, entities: List[Dict], relationships: List[Dict]) -> nx.Graph:
        """Create a NetworkX graph from entities and relationships."""
        G = nx.Graph()
        
        # Add entity nodes
        for entity in entities:
            G.add_node(entity['id'], **entity)
        
        # Add relationship edges
        for rel in relationships:
            G.add_edge(
                rel['entity1_id'], 
                rel['entity2_id'], 
                type=rel['type'],
                strength=rel['strength'],
                description=rel['description']
            )
        
        return G
    
    def _enhance_entities_with_graph_metrics(self, entities: List[Dict], 
                                           relationships: List[Dict]) -> List[Dict]:
        """Enhance entities with graph-based importance scores."""
        G = self._create_knowledge_graph(entities, relationships)
        
        # Calculate centrality measures
        if len(G.nodes()) > 0:
            try:
                centrality = nx.degree_centrality(G)
                betweenness = nx.betweenness_centrality(G)
                pagerank = nx.pagerank(G, max_iter=100)
                
                # Enhance entities with graph metrics
                for entity in entities:
                    entity_id = entity['id']
                    if entity_id in centrality:
                        entity['degree_centrality'] = centrality[entity_id]
                        entity['betweenness_centrality'] = betweenness[entity_id]
                        entity['pagerank'] = pagerank[entity_id]
                        
                        # Calculate composite importance score
                        entity['importance_score'] = (
                            0.4 * centrality[entity_id] +
                            0.3 * betweenness[entity_id] +
                            0.3 * pagerank[entity_id]
                        )
                    else:
                        entity['importance_score'] = 0.1
            except:
                # Fallback if graph analysis fails
                for entity in entities:
                    entity['importance_score'] = 0.5
        
        return entities
    
    def process_file_with_graph(self, file_path: Path) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Process a file and extract documents, entities, and relationships.
        Returns: (documents, entities, relationships)
        """
        if file_path.suffix.lower() != '.pdf':
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

        # Extract content (reuse existing logic)
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text_contents = self._extract_text_by_page(pdf_reader)
            tables_data = self._extract_tables(file_path)

        # Process and chunk content
        processed_chunks = []
        all_entities = []
        all_relationships = []
        
        # Process text content
        for text_content in text_contents:
            chunks = self.text_splitter.split_text(text_content.content)
            for i, chunk in enumerate(chunks):
                doc_id = generate_document_id(f"{file_path.name}_{text_content.page_num}_{i}")
                
                # Extract entities and relationships for this chunk
                entities = self._extract_entities_with_nlp(chunk, doc_id)
                relationships = self._extract_relationships(chunk, entities, doc_id)
                
                processed_chunks.append({
                    'text': chunk,
                    'metadata': {
                        'source': file_path.name,
                        'chunk_id': doc_id,
                        'page_num': text_content.page_num,
                        'content_type': 'text',
                        'total_pages': len(pdf_reader.pages),
                        'entity_count': len(entities),
                        'relationship_count': len(relationships)
                    }
                })
                
                all_entities.extend(entities)
                all_relationships.extend(relationships)

        # Process tables
        for table_data in tables_data:
            table_text = table_data['table'].to_string()
            chunks = self.text_splitter.split_text(table_text)
            
            for i, chunk in enumerate(chunks):
                doc_id = generate_document_id(f"{file_path.name}_table_{table_data['page_num']}_{i}")
                
                entities = self._extract_entities_with_nlp(chunk, doc_id)
                relationships = self._extract_relationships(chunk, entities, doc_id)
                
                processed_chunks.append({
                    'text': chunk,
                    'metadata': {
                        'source': file_path.name,
                        'chunk_id': doc_id,
                        'page_num': table_data['page_num'],
                        'content_type': 'table',
                        'table_num': table_data['table_num'],
                        'total_pages': len(pdf_reader.pages),
                        'entity_count': len(entities),
                        'relationship_count': len(relationships)
                    }
                })
                
                all_entities.extend(entities)
                all_relationships.extend(relationships)
        
        # Enhance entities with graph metrics
        all_entities = self._enhance_entities_with_graph_metrics(all_entities, all_relationships)
        
        return processed_chunks, all_entities, all_relationships
    
    def _extract_text_by_page(self, pdf_reader: PyPDF2.PdfReader):
        """Extract text content by page (reused from original)."""
        from core.document_processor import DocumentContent
        contents = []
        for page_num, page in enumerate(pdf_reader.pages, 1):
            text = page.extract_text()
            if text.strip():
                contents.append(DocumentContent(
                    content=text,
                    content_type='text',
                    page_num=page_num,
                    metadata={'type': 'main_text'}
                ))
        return contents
    
    def _extract_tables(self, pdf_path: Path):
        """Extract tables from PDF (reused from original)."""
        tables_data = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                tables = page.extract_tables()
                for table_num, table in enumerate(tables, 1):
                    if table and any(any(cell for cell in row) for row in table):
                        df = pd.DataFrame(table[1:], columns=table[0])
                        tables_data.append({
                            'table': df,
                            'page_num': page_num,
                            'table_num': table_num,
                            'row_count': len(df),
                            'col_count': len(df.columns)
                        })
        return tables_data