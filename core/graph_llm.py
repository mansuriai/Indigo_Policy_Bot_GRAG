# core/graph_llm.py
from typing import List, Dict, Optional, Tuple, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.callbacks.base import BaseCallbackHandler
from utils.config import config
from utils.helpers import format_chat_history
import json
import re

class GraphLLMManager:
    """Enhanced LLM Manager that leverages graph context and entity relationships."""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=config.LLM_MODEL,
            temperature=0.7,
            api_key=config.OPENAI_API_KEY,
            streaming=True
        )
        
        self.system_prompt = """You are an AI travel assistant for IndiGo Airlines with access to a rich knowledge graph of entities and relationships.

        IMPORTANT GUIDELINES:
        1. Use the provided ENTITY CONTEXT and GRAPH RELATIONSHIPS to provide comprehensive, detailed answers.
        2. When entities are mentioned in your response, reference their relationships to provide deeper insights.
        3. Connect related concepts using the relationship information to give holistic answers.
        4. If entity relationships suggest additional relevant information, include it proactively.
        5. Use the graph structure to identify and mention related services, policies, or offers.
        6. Prioritize information from highly connected entities (those with more relationships).
        7. Structure your response to show how different concepts connect to each other.
        8. Make your response conversational while being comprehensive and accurate.

        CONTEXT DOCUMENTS:
        {context}

        ENTITY CONTEXT:
        {entity_context}

        GRAPH RELATIONSHIPS:
        {relationships}

        CONVERSATION HISTORY:
        {chat_history}

        Remember: Use the graph structure to provide richer, more connected answers that show how different concepts relate to each other.
        """
        
        self.human_prompt = "{question}"
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", self.human_prompt)
        ])
        
        # Non-streaming LLM for entity extraction
        self.analysis_llm = ChatOpenAI(
            model=config.LLM_MODEL,
            temperature=0.1,
            api_key=config.OPENAI_API_KEY,
            streaming=False
        )
        
        self.entity_extraction_prompt = """Extract key entities from the user's question that might be relevant for querying a knowledge graph about IndiGo Airlines.

        User Question: {question}

        Extract entities in the following categories:
        - FLIGHT_SERVICE: flight operations, booking, check-in, baggage, etc.
        - FARE_TYPE: types of fares, discounts, pricing categories
        - LOCATION: airports, cities, terminals, gates
        - POLICY: rules, regulations, procedures
        - PRODUCT: IndiGo services, features, add-ons
        - OFFER: promotions, discounts, deals

        Return a JSON list of entities with their types:
        [{"name": "entity_name", "type": "ENTITY_TYPE"}]
        """
        
        self.entity_prompt = ChatPromptTemplate.from_messages([
            ("system", self.entity_extraction_prompt),
            ("human", "")
        ])
        
        self.entity_chain = (
            self.entity_prompt
            | self.analysis_llm
            | StrOutputParser()
        )
    
    def extract_query_entities(self, question: str) -> List[Dict[str, str]]:
        """Extract entities from the user's question for graph querying."""
        try:
            response = self.entity_chain.invoke({"question": question})
            
            # Clean up the response to extract JSON
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:]
            if response.endswith('```'):
                response = response[:-3]
            
            entities = json.loads(response)
            return entities if isinstance(entities, list) else []
        except Exception as e:
            print(f"Error extracting entities: {str(e)}")
            # Fallback: simple keyword extraction
            return self._simple_entity_extraction(question)
    
    def _simple_entity_extraction(self, question: str) -> List[Dict[str, str]]:
        """Fallback entity extraction using keywords."""
        entities = []
        question_lower = question.lower()
        
        # Define keyword mappings
        keyword_mappings = {
            'FLIGHT_SERVICE': ['flight', 'booking', 'check-in', 'baggage', 'seat', 'boarding'],
            'FARE_TYPE': ['fare', '6e', 'student', 'discount', 'business', 'economy'],
            'LOCATION': ['airport', 'delhi', 'mumbai', 'bangalore', 'terminal'],
            'POLICY': ['policy', 'cancellation', 'refund', 'change', 'rules'],
            'PRODUCT': ['indigo', '6e connect', 'fast forward', 'priority'],
            'OFFER': ['offer', 'deal', 'promotion', 'cashback', 'savings']
        }
        
        for entity_type, keywords in keyword_mappings.items():
            for keyword in keywords:
                if keyword in question_lower:
                    entities.append({
                        'name': keyword,
                        'type': entity_type
                    })
        
        return entities
    
    def _format_entity_context(self, entity_context: Dict[str, Any]) -> str:
        """Format entity context for the prompt."""
        if not entity_context:
            return "No specific entity context available."
        
        formatted_context = []
        for entity_name, context in entity_context.items():
            context_str = f"Entity: {entity_name} (Type: {context.get('type', 'Unknown')})\n"
            
            if context.get('description'):
                context_str += f"Description: {context['description']}\n"
            
            if context.get('related_entities'):
                related = [f"{rel['name']} ({rel['relation']})" for rel in context['related_entities'][:5]]
                context_str += f"Related to: {', '.join(related)}\n"
            
            if context.get('documents'):
                doc_count = len(context['documents'])
                context_str += f"Found in {doc_count} document(s)\n"
            
            formatted_context.append(context_str)
        
        return "\n".join(formatted_context)
    
    def _format_relationships(self, relationships: List[Dict]) -> str:
        """Format relationship information for the prompt."""
        if not relationships:
            return "No specific relationships found."
        
        formatted_rels = []
        for rel in relationships[:10]:  # Limit to top 10 relationships
            rel_str = f"- {rel.get('entity1', 'Entity1')} --[{rel.get('type', 'RELATED')}]--> {rel.get('entity2', 'Entity2')}"
            if rel.get('description'):
                rel_str += f" ({rel['description']})"
            formatted_rels.append(rel_str)
        
        return "\n".join(formatted_rels)
    
    def generate_graph_enhanced_response(
        self,
        question: str,
        context: List[Dict],
        entity_context: Dict[str, Any],
        relationships: List[Dict] = None,
        chat_history: Optional[List[Dict]] = None,
        streaming_container = None
    ) -> str:
        """Generate a response enhanced with graph context and relationships."""
        
        # Format all the context information
        formatted_context = "\n\n".join([
            f"DOCUMENT {i+1}:\n{doc['text']}\n" 
            for i, doc in enumerate(context)
        ])
        
        formatted_entity_context = self._format_entity_context(entity_context)
        formatted_relationships = self._format_relationships(relationships or [])
        formatted_history = format_chat_history(chat_history) if chat_history else ""
        
        # If streaming is requested, use a streaming handler
        if streaming_container:
            from core.llm import StreamHandler
            stream_handler = StreamHandler(streaming_container)
            
            # Create a streaming LLM
            streaming_llm = ChatOpenAI(
                model=config.LLM_MODEL,
                temperature=0.7,
                streaming=True,
                callbacks=[stream_handler]
            )
            
            # Create a chain with the streaming LLM
            streaming_chain = (
                self.prompt 
                | streaming_llm 
                | StrOutputParser()
            )
            
            # Run the chain
            response = streaming_chain.invoke({
                "context": formatted_context,
                "entity_context": formatted_entity_context,
                "relationships": formatted_relationships,
                "chat_history": formatted_history,
                "question": question
            })
            
            # Use the accumulated text from the stream handler
            response = stream_handler.text
        else:
            # No streaming, use the normal chain
            chain = (
                self.prompt 
                | self.llm 
                | StrOutputParser()
            )
            
            response = chain.invoke({
                "context": formatted_context,
                "entity_context": formatted_entity_context,
                "relationships": formatted_relationships,
                "chat_history": formatted_history,
                "question": question
            })
        
        return response
    
    def _extract_graph_insights(self, entity_context: Dict[str, Any], 
                               relationships: List[Dict]) -> str:
        """Extract insights from the graph structure to enhance responses."""
        insights = []
        
        # Find highly connected entities (important topics)
        entity_connections = {}
        for rel in relationships:
            entity1 = rel.get('entity1', '')
            entity2 = rel.get('entity2', '')
            
            entity_connections[entity1] = entity_connections.get(entity1, 0) + 1
            entity_connections[entity2] = entity_connections.get(entity2, 0) + 1
        
        # Get top connected entities
        top_entities = sorted(entity_connections.items(), key=lambda x: x[1], reverse=True)[:3]
        
        if top_entities:
            insights.append(f"Key topics in this domain: {', '.join([ent[0] for ent in top_entities])}")
        
        # Find relationship patterns
        relationship_types = {}
        for rel in relationships:
            rel_type = rel.get('type', 'UNKNOWN')
            relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1
        
        if relationship_types:
            common_relations = sorted(relationship_types.items(), key=lambda x: x[1], reverse=True)[:2]
            insights.append(f"Common relationships: {', '.join([rel[0] for rel in common_relations])}")
        
        return "; ".join(insights) if insights else ""
    
    def generate_response_with_graph_summary(
        self,
        question: str,
        context: List[Dict],
        graph_summary: Dict[str, Any],
        chat_history: Optional[List[Dict]] = None,
        streaming_container = None
    ) -> str:
        """Generate response using graph summary for broader context."""
        
        summary_context = ""
        if graph_summary and graph_summary.get('entities'):
            summary_context = "KNOWLEDGE GRAPH OVERVIEW:\n"
            for entity in graph_summary['entities'][:5]:
                summary_context += f"- {entity['name']} ({entity['type']}): {entity['document_count']} docs, {entity['relationship_count']} connections\n"
        
        enhanced_system_prompt = f"""You are an AI travel assistant for IndiGo Airlines with comprehensive knowledge graph access.

        {summary_context}

        Use this graph overview to understand the broader context and connections in our knowledge base.
        Provide comprehensive answers that leverage these connections.

        CONTEXT DOCUMENTS:
        {{context}}

        CONVERSATION HISTORY:
        {{chat_history}}
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", enhanced_system_prompt),
            ("human", "{question}")
        ])
        
        formatted_context = "\n\n".join([doc['text'] for doc in context])
        formatted_history = format_chat_history(chat_history) if chat_history else ""
        
        if streaming_container:
            from core.llm import StreamHandler
            stream_handler = StreamHandler(streaming_container)
            
            streaming_llm = ChatOpenAI(
                model=config.LLM_MODEL,
                temperature=0.7,
                streaming=True,
                callbacks=[stream_handler]
            )
            
            streaming_chain = prompt | streaming_llm | StrOutputParser()
            
            response = streaming_chain.invoke({
                "context": formatted_context,
                "chat_history": formatted_history,
                "question": question
            })
            
            response = stream_handler.text
        else:
            chain = prompt | self.llm | StrOutputParser()
            
            response = chain.invoke({
                "context": formatted_context,
                "chat_history": formatted_history,
                "question": question
            })
        
        return response
        