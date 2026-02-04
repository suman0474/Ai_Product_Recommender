# agentic/standards_chat_agent.py
# Specialized Chat Agent for Standards Documentation Q&A

import json
import logging
import re
from typing import Dict, Any, List, Optional
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel, field_validator

import os
# Import existing infrastructure
from agentic.rag.vector_store import get_vector_store
from services.llm.fallback import create_llm_with_fallback
from prompts_library import load_prompt_sections

# Import consolidated JSON utilities
from utils.json_utils import (
    extract_json_from_response,
    sanitize_json_string,
    ensure_string,
    ensure_float
)
logger = logging.getLogger(__name__)


# ============================================================================
# RESPONSE VALIDATION MODELS (Pydantic)
# ============================================================================

class CitationModel(BaseModel):
    """Citation from a standards document."""
    source: str
    content: str
    relevance: float

    @field_validator('relevance')
    @classmethod
    def validate_relevance(cls, v):
        """Ensure relevance is between 0.0 and 1.0."""
        if not isinstance(v, (int, float)):
            raise ValueError('relevance must be numeric')
        if not (0.0 <= v <= 1.0):
            raise ValueError(f'relevance must be between 0.0 and 1.0, got {v}')
        return float(v)


class StandardsRAGResponse(BaseModel):
    """Standard format for Standards RAG responses."""
    answer: str
    citations: List[CitationModel] = []
    confidence: float
    sources_used: List[str] = []

    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v):
        """Ensure confidence is between 0.0 and 1.0."""
        if not isinstance(v, (int, float)):
            raise ValueError('confidence must be numeric')
        if not (0.0 <= v <= 1.0):
            raise ValueError(f'confidence must be between 0.0 and 1.0, got {v}')
        return float(v)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
# Note: extract_json_from_response and sanitize_json_string are now imported
# from utils.json_utils module for consistency across all RAG agents.

# Local helper kept for backward compatibility (uses imported function internally)
def _sanitize_json_string(text: str) -> str:
    """Wrapper for backward compatibility - delegates to utils.json_utils."""
    return sanitize_json_string(text)


def normalize_response_dict(response_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    FIX #2: Normalize LLM response dictionary to match StandardsRAGResponse schema.

    Handles:
    - answer field being a dict instead of string → converts to string
    - Missing required fields → adds defaults
    - Invalid data types → coerces to correct types
    - Malformed citations → reconstructs valid list

    Args:
        response_dict: Raw parsed response from LLM

    Returns:
        Normalized dictionary matching StandardsRAGResponse schema
    """
    logger.info("[FIX2] Normalizing response dictionary for schema compliance")

    # Use consolidated utilities for type coercion
    answer = ensure_string(
        response_dict.get('answer', ''),
        default="Unable to generate answer."
    )
    if not answer:
        answer = "Unable to generate answer."

    # Ensure confidence is a float between 0-1
    confidence = ensure_float(
        response_dict.get('confidence', 0.5),
        default=0.5,
        min_val=0.0,
        max_val=1.0
    )

    # Handle citations
    citations = response_dict.get('citations', [])
    if not isinstance(citations, list):
        citations = []

    # Validate and reconstruct citations
    valid_citations = []
    for cite in citations:
        if not isinstance(cite, dict):
            continue
        try:
            valid_cite = {
                'source': ensure_string(cite.get('source', 'unknown'), default='unknown'),
                'content': ensure_string(cite.get('content', ''), default=''),
                'relevance': ensure_float(cite.get('relevance', 0.5), default=0.5, min_val=0.0, max_val=1.0)
            }
            valid_citations.append(valid_cite)
        except (ValueError, TypeError):
            continue

    # Handle sources_used
    sources_used = response_dict.get('sources_used', [])
    if not isinstance(sources_used, list):
        sources_used = []
    sources_used = [str(s) for s in sources_used if s]

    normalized = {
        'answer': answer,
        'citations': valid_citations,
        'confidence': confidence,
        'sources_used': sources_used
    }

    logger.info("[FIX2] Response normalized successfully")
    return normalized


# ============================================================================
# PROMPT TEMPLATE - Loaded from consolidated prompts_library file
# ============================================================================

_RAG_PROMPTS = load_prompt_sections("rag_prompts")
STANDARDS_CHAT_PROMPT = _RAG_PROMPTS["STANDARDS_CHAT"]


# ============================================================================
# STANDARDS CHAT AGENT
# ============================================================================

class StandardsChatAgent:
    """
    Specialized agent for answering questions using standards documents from Pinecone.

    This agent retrieves relevant standards documents from the vector store,
    constructs context, and generates grounded answers using Google Generative AI.
    Includes production-ready retry logic and error handling.
    """

    def __init__(self, llm=None, temperature: float = 0.1):
        """
        Initialize the Standards Chat Agent.

        Args:
            llm: Language model instance (if None, creates default Gemini)
            temperature: Temperature for generation (default 0.1 for factual responses)
        """
        # Initialize LLM
        if llm is None:
            self.llm = create_llm_with_fallback(
                model="gemini-2.5-flash",
                temperature=temperature,
                max_tokens=2000,
                timeout=120  # Add timeout protection to prevent hanging requests
            )
        else:
            self.llm = llm

        # Get vector store
        self.vector_store = get_vector_store()

        # Create prompt template
        self.prompt = ChatPromptTemplate.from_template(STANDARDS_CHAT_PROMPT)

        # Create output parser
        self.parser = JsonOutputParser()

        # Create chain
        self.chain = self.prompt | self.llm | self.parser

        logger.info("StandardsChatAgent initialized with Google Generative AI")

    def retrieve_documents(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Retrieve relevant standards documents from vector store.

        Args:
            question: User's question
            top_k: Number of top documents to retrieve

        Returns:
            Dictionary with retrieval results
        """
        try:
            # Search vector store
            search_results = self.vector_store.search(
                collection_type="standards",
                query=question,
                top_k=top_k
            )

            logger.info(f"Retrieved {search_results.get('result_count', 0)} documents for question")

            return search_results

        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return {
                "results": [],
                "result_count": 0,
                "error": str(e)
            }

    def build_context(self, search_results: Dict[str, Any]) -> str:
        """
        Build formatted context from search results.

        Args:
            search_results: Results from vector store search

        Returns:
            Formatted context string
        """
        results = search_results.get('results', [])

        if not results:
            return "No relevant standards documents found."

        context_parts = []

        for i, result in enumerate(results, 1):
            # Extract metadata
            metadata = result.get('metadata', {})
            source = metadata.get('filename', 'unknown')
            standard_type = metadata.get('standard_type', 'general')
            standards_refs = metadata.get('standards_references', [])

            # Get content and relevance
            content = result.get('content', '')
            relevance = result.get('relevance_score', 0.0)

            # Build document section
            doc_section = f"[Document {i}: {source}]\n"
            doc_section += f"Type: {standard_type}\n"
            if standards_refs:
                doc_section += f"Referenced Standards: {', '.join(standards_refs[:5])}\n"
            doc_section += f"Relevance: {relevance:.3f}\n"
            doc_section += f"\nContent:\n{content}"

            context_parts.append(doc_section)

        return "\n\n" + "="*80 + "\n\n".join(context_parts)

    def extract_citations(self, search_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract citation information from search results.

        Args:
            search_results: Results from vector store search

        Returns:
            List of citations
        """
        citations = []

        for result in search_results.get('results', []):
            metadata = result.get('metadata', {})
            citation = {
                'source': metadata.get('filename', 'unknown'),
                'content': result.get('content', '')[:200] + "...",  # First 200 chars
                'relevance': result.get('relevance_score', 0.0),
                'standard_type': metadata.get('standard_type', 'general'),
                'standards_references': metadata.get('standards_references', [])
            }
            citations.append(citation)

        return citations

    def run(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Answer question using standards documents.

        Args:
            question: User's question
            top_k: Number of documents to retrieve

        Returns:
            Dictionary with answer, citations, confidence, and sources
        """
        search_results = None
        context = ""
        
        try:
            logger.info(f"Processing question: {question[:100]}...")

            # Step 1: Retrieve documents
            search_results = self.retrieve_documents(question, top_k)

            if search_results.get('result_count', 0) == 0:
                # ROOT CAUSE FIX: Provide actionable error message based on mock reason
                mock_reason = search_results.get('mock_reason', None)

                if mock_reason == 'MISSING_API_KEY':
                    error_msg = (
                        "Vector store is not properly configured: PINECONE_API_KEY environment variable is missing. "
                        "Please set the PINECONE_API_KEY environment variable to enable document retrieval."
                    )
                    logger.error(f"[StandardsChatAgent] {error_msg}")
                elif mock_reason and 'INIT_ERROR' in mock_reason:
                    error_msg = (
                        f"Vector store initialization failed ({mock_reason}). "
                        "Please verify PINECONE_API_KEY is valid and the Pinecone index exists."
                    )
                    logger.error(f"[StandardsChatAgent] {error_msg}")
                else:
                    error_msg = "I don't have any relevant standards documents to answer this question."

                return {
                    'answer': error_msg,
                    'citations': [],
                    'confidence': 0.0,
                    'sources_used': [],
                    'error': 'No documents found' if not mock_reason else f'Vector store unavailable: {mock_reason}'
                }

            # Step 2: Build context
            context = self.build_context(search_results)

            # Step 3: Generate answer with strict JSON parsing
            logger.info("Generating answer with LLM...")

            response = None
            raw_text = None

            try:
                # Invoke LLM directly (without the parser in the chain)
                raw_chain = self.prompt | self.llm
                raw_response = raw_chain.invoke({
                    'question': question,
                    'retrieved_context': context
                })

                # Extract text content from response
                if hasattr(raw_response, 'content'):
                    raw_text = raw_response.content
                elif hasattr(raw_response, 'text'):
                    raw_text = raw_response.text
                else:
                    raw_text = str(raw_response)

                logger.debug(f"Raw LLM response: {raw_text[:200]}...")

                # Parse JSON with strict validation
                try:
                    # First, try direct JSON parsing
                    response_dict = json.loads(raw_text)
                    logger.debug("JSON parsed successfully (direct)")
                except json.JSONDecodeError as json_error:
                    logger.warning(
                        f"Direct JSON parse failed: {json_error}. "
                        f"Attempting extraction from response: {raw_text[:100]}..."
                    )
                    # Try extracting JSON from markdown or other formats
                    response_dict = extract_json_from_response(raw_text)

                    if response_dict is None:
                        logger.error(
                            f"Failed to extract JSON from response. Raw: {raw_text[:300]}"
                        )
                        raise ValueError(
                            "LLM returned invalid JSON format. "
                            "Response must be valid JSON only."
                        )
                    logger.info("JSON extracted successfully (from response)")

                # FIX #2: Normalize response dictionary for schema compliance
                logger.debug("Normalizing response for schema compliance...")
                response_dict = normalize_response_dict(response_dict)

                # Validate against StandardsRAGResponse schema
                try:
                    response = StandardsRAGResponse(**response_dict)
                    logger.debug(f"Response validated: confidence={response.confidence}")
                except Exception as validation_error:
                    logger.error(f"Response validation failed: {validation_error}")
                    logger.error(f"Response dict: {response_dict}")
                    raise ValueError(
                        f"LLM response structure invalid: {validation_error}"
                    )

            except Exception as e:
                logger.error(f"Error generating answer: {e}", exc_info=True)
                # Create fallback response with retrieved sources
                sources_used = list(set(
                    r.get('metadata', {}).get('filename', 'unknown')
                    for r in search_results.get('results', [])
                ))
                response = StandardsRAGResponse(
                    answer=f"I found relevant standards documents ({', '.join(sources_used[:3])}) but the AI service is temporarily unavailable. Please try again in a few moments or check your API quota status.",
                    citations=self.extract_citations(search_results),
                    confidence=0.3,
                    sources_used=sources_used
                )

            # Convert Pydantic model to dict for consistency with rest of code
            if isinstance(response, StandardsRAGResponse):
                response = response.model_dump()
            elif response is None:
                logger.error("Response is None after all attempts")
                response = {}

            # Step 4: Add metadata
            result = {
                'answer': response.get('answer', ''),
                'citations': response.get('citations', []),
                'confidence': response.get('confidence', 0.0),
                'sources_used': response.get('sources_used', [])
            }

            # Calculate average retrieval relevance as fallback confidence
            if search_results and 'results' in search_results and search_results['results']:
                avg_relevance = sum(r.get('relevance_score', 0) for r in search_results['results']) / len(search_results['results'])
                # Use the higher of LLM confidence or retrieval relevance
                result['confidence'] = max(result['confidence'], avg_relevance)
                
                # If sources_used is empty but we have results, populate from search results
                if not result['sources_used']:
                    result['sources_used'] = list(set(
                        r.get('metadata', {}).get('filename', 'unknown')
                        for r in search_results['results']
                    ))

            logger.info(f"Answer generated with confidence: {result['confidence']:.2f}")
            logger.info(f"Sources used: {result['sources_used']}")

            return result

        except Exception as e:
            logger.error(f"Error in StandardsChatAgent.run: {e}", exc_info=True)
            
            # Even on error, try to return useful info if we have search results
            sources_used = []
            if search_results and 'results' in search_results:
                sources_used = list(set(
                    r.get('metadata', {}).get('filename', 'unknown')
                    for r in search_results['results']
                ))
            
            return {
                'answer': f"An error occurred while processing your question: {str(e)}",
                'citations': [],
                'confidence': 0.0,
                'sources_used': sources_used,
                'error': str(e)
            }


# ============================================================================
# SINGLETON INSTANCE (Performance Optimization)
# ============================================================================

_standards_chat_agent_instance = None
_standards_chat_agent_lock = None

def _get_agent_lock():
    """Get thread lock for singleton (lazy initialization)."""
    global _standards_chat_agent_lock
    if _standards_chat_agent_lock is None:
        import threading
        _standards_chat_agent_lock = threading.Lock()
    return _standards_chat_agent_lock


def get_standards_chat_agent(temperature: float = 0.1) -> StandardsChatAgent:
    """
    Get or create the singleton StandardsChatAgent instance.
    
    This avoids expensive re-initialization of LLM and vector store connections
    on every call, significantly improving performance for batch operations.
    
    Args:
        temperature: Temperature for generation (only used on first call)
        
    Returns:
        Cached StandardsChatAgent instance
    """
    global _standards_chat_agent_instance
    
    if _standards_chat_agent_instance is None:
        with _get_agent_lock():
            # Double-check after acquiring lock
            if _standards_chat_agent_instance is None:
                logger.info("[StandardsChatAgent] Creating singleton instance (first call)")
                _standards_chat_agent_instance = StandardsChatAgent(temperature=temperature)
    
    return _standards_chat_agent_instance


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_standards_chat_agent(temperature: float = 0.1) -> StandardsChatAgent:
    """
    Create a StandardsChatAgent instance with default settings.
    
    NOTE: For better performance, use get_standards_chat_agent() instead,
    which returns a cached singleton instance.

    Args:
        temperature: Temperature for generation

    Returns:
        StandardsChatAgent instance (uses singleton for efficiency)
    """
    # Use singleton for better performance
    return get_standards_chat_agent(temperature=temperature)


def ask_standards_question(question: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Quick function to ask a question to the standards system.

    Args:
        question: Question to ask
        top_k: Number of documents to retrieve

    Returns:
        Answer dictionary
    """
    agent = create_standards_chat_agent()
    return agent.run(question, top_k)


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    # Test the agent
    logger.info("="*80)
    logger.info("TESTING STANDARDS CHAT AGENT")
    logger.info("="*80)

    test_questions = [
        "What are the SIL2 requirements for pressure transmitters?",
        "What calibration standards apply to temperature sensors?",
        "What are the ATEX requirements for flow measurement devices?"
    ]

    agent = create_standards_chat_agent()

    for i, question in enumerate(test_questions, 1):
        logger.info(f"\n[Test {i}] Question: {question}")
        logger.info("-"*80)

        result = agent.run(question, top_k=3)

        logger.info(f"\nAnswer: {result['answer'][:300]}...")
        logger.info(f"\nConfidence: {result['confidence']:.2f}")
        logger.info(f"Sources: {', '.join(result['sources_used'])}")

        if result.get('citations'):
            logger.info(f"\nCitations:")
            for j, cite in enumerate(result['citations'][:2], 1):
                logger.info(f"  [{j}] {cite['source']} (relevance: {cite.get('relevance', 0):.2f})")

        logger.info("="*80)
