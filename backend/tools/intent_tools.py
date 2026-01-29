# tools/intent_tools.py
# Intent Classification and Requirements Extraction Tools

import json
import logging
from typing import Dict, Any, Optional
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
import os
from llm_fallback import create_llm_with_fallback, invoke_with_retry_fallback
from prompts_library import load_prompt_sections
logger = logging.getLogger(__name__)


# ============================================================================
# TOOL INPUT SCHEMAS
# ============================================================================

class ClassifyIntentInput(BaseModel):
    """Input for intent classification"""
    user_input: str = Field(description="User's input message to classify")
    current_step: Optional[str] = Field(default=None, description="Current workflow step")
    context: Optional[str] = Field(default=None, description="Conversation context")


class ExtractRequirementsInput(BaseModel):
    """Input for requirements extraction"""
    user_input: str = Field(description="User's input containing requirements")


# ============================================================================
# PROMPTS - Loaded from consolidated prompts_library file
# ============================================================================

_INTENT_PROMPTS = load_prompt_sections("intent_prompts")

INTENT_CLASSIFICATION_PROMPT = _INTENT_PROMPTS["CLASSIFICATION"]
REQUIREMENTS_EXTRACTION_PROMPT = _INTENT_PROMPTS["REQUIREMENTS_EXTRACTION"]


# ============================================================================
# RULE-BASED CLASSIFICATION (OPTIMIZATION - avoid LLM for obvious intents)
# ============================================================================
# These patterns are checked BEFORE calling the LLM to save API calls.
# All callers of classify_intent_tool benefit automatically.

_GREETING_PHRASES = {'hi', 'hello', 'hey', 'hi there', 'hello there',
                     'good morning', 'good afternoon', 'good evening'}

_CONFIRM_PHRASES = {'yes', 'y', 'yep', 'yeah', 'sure', 'ok', 'okay',
                    'proceed', 'continue', 'go ahead', 'confirm', 'approved'}

_REJECT_PHRASES = {'no', 'n', 'nope', 'cancel', 'stop', 'reject', 'decline',
                   'never mind', 'nevermind', 'forget it'}

_KNOWLEDGE_STARTERS = [
    'what is ', 'what are ', 'what does ', "what's ",
    'how does ', 'how do ', 'how to ', 'how can ',
    'why is ', 'why does ', 'why do ', 'why are ',
    'tell me about ', 'explain ', 'define ', 'describe ',
    'difference between ', 'compare ', 'meaning of ', 'definition of '
]

_PRODUCT_REQUEST_STARTERS = [
    'i need a ', 'i need an ', 'looking for a ', 'looking for an ',
    'i want a ', 'i want an ', 'need a ', 'need an ',
    'find me a ', 'find a ', 'get me a ', 'show me ',
    'recommend a ', 'suggest a '
]


def _classify_rule_based(user_input: str) -> Optional[Dict[str, Any]]:
    """
    Attempt rule-based intent classification without calling the LLM.

    Returns a result dict if a rule matches, or None if LLM is needed.
    This saves 1 LLM API call for ~40-60% of common queries.
    """
    query = user_input.lower().strip()

    # Pure greeting
    if query in _GREETING_PHRASES:
        logger.info(f"[INTENT_RULE] Greeting detected: '{query}'")
        return {
            "success": True,
            "intent": "greeting",
            "confidence": 1.0,
            "next_step": "greeting",
            "extracted_info": {"rule_based": True},
            "is_solution": False,
            "solution_indicators": []
        }

    # Confirm/Reject
    if query in _CONFIRM_PHRASES:
        logger.info(f"[INTENT_RULE] Confirm detected: '{query}'")
        return {
            "success": True,
            "intent": "confirm",
            "confidence": 1.0,
            "next_step": None,
            "extracted_info": {"rule_based": True},
            "is_solution": False,
            "solution_indicators": []
        }

    if query in _REJECT_PHRASES:
        logger.info(f"[INTENT_RULE] Reject detected: '{query}'")
        return {
            "success": True,
            "intent": "reject",
            "confidence": 1.0,
            "next_step": None,
            "extracted_info": {"rule_based": True},
            "is_solution": False,
            "solution_indicators": []
        }

    # Knowledge questions
    if any(query.startswith(p) for p in _KNOWLEDGE_STARTERS):
        logger.info(f"[INTENT_RULE] Knowledge question detected: '{query[:50]}'")
        return {
            "success": True,
            "intent": "question",
            "confidence": 0.95,
            "next_step": None,
            "extracted_info": {"rule_based": True},
            "is_solution": False,
            "solution_indicators": []
        }

    # Product requests - need to distinguish complex systems from simple requests (PHASE 3 FIX)
    if any(query.startswith(p) for p in _PRODUCT_REQUEST_STARTERS):
        # PHASE 3 FIX: Knowledge question indicators - these override solution detection
        # If query is asking ABOUT systems (knowledge), not building them
        knowledge_indicators = [
            'what is', 'what are', 'how does', 'how do', 'explain',
            'tell me about', 'describe', 'difference between', 'compare',
            'meaning of', 'definition of', 'what\'s the', 'can you explain'
        ]
        is_knowledge_query = any(kw in query for kw in knowledge_indicators)
        
        # PHASE 3 FIX: Stricter solution phrases - require ACTION verbs for building/designing
        # These must indicate the user is BUILDING something, not asking about it
        solution_action_phrases = [
            'i\'m designing', 'i\'m building', 'i\'m implementing',
            'i\'m creating', 'i\'m planning', 'i\'m developing',
            'design a system', 'build a system', 'create a system',
            'implement a solution', 'develop a solution',
            'need to design', 'need to build', 'need to implement',
            'planning to build', 'planning to design', 'planning to create'
        ]
        is_solution_action = any(phrase in query for phrase in solution_action_phrases)
        
        # PHASE 3 FIX: Complex system phrases (multi-instrument requirement)
        complex_system_phrases = [
            'multiple instruments', 'multiple transmitters', 'multiple sensors',
            'instrumentation package', 'full instrumentation', 'complete instrumentation',
            'monitoring system for', 'control system for', 'measurement system for',
            'plant instrumentation', 'process instrumentation', 'skid package'
        ]
        is_complex_system = any(phrase in query for phrase in complex_system_phrases)
        
        # PHASE 3 FIX: Semantic validation - use engenie_chat classifier
        # ONLY used to detect knowledge queries that should NOT go to solution
        # NOT used to override simple requirements to question
        is_rag_knowledge_query = False
        if is_knowledge_query:
            # Already a knowledge query based on keywords
            is_rag_knowledge_query = True
        elif is_solution_action or is_complex_system:
            # User wants to BUILD something, but might actually be asking about it
            # Use semantic classifier to double-check
            try:
                from agentic.engenie_chat.engenie_chat_intent_agent import classify_query, DataSource
                data_source, confidence, _ = classify_query(query, use_semantic_llm=False)
                # Standards and Strategy RAG indicate knowledge/info queries
                rag_knowledge_sources = {DataSource.STANDARDS_RAG, DataSource.STRATEGY_RAG}
                if data_source in rag_knowledge_sources and confidence >= 0.8:
                    is_rag_knowledge_query = True
                    logger.info(f"[INTENT_RULE] Semantic check: Knowledge RAG detected ({data_source.value}, conf={confidence:.2f})")
            except ImportError:
                logger.debug("[INTENT_RULE] Semantic classifier not available, using rule-based only")
            except Exception as e:
                logger.debug(f"[INTENT_RULE] Semantic check failed: {e}")
        
        # PHASE 3 FIX: Decision logic with semantic awareness
        # 1. Knowledge queries → EnGenie Chat (question)
        if is_knowledge_query or is_rag_knowledge_query:
            logger.info(f"[INTENT_RULE] Knowledge query detected: '{query[:50]}' (knowledge={is_knowledge_query}, rag_knowledge={is_rag_knowledge_query})")
            return {
                "success": True,
                "intent": "question",
                "confidence": 0.9,
                "next_step": None,
                "extracted_info": {
                    "rule_based": True,
                    "is_knowledge_query": is_knowledge_query,
                    "is_rag_knowledge_query": is_rag_knowledge_query
                },
                "is_solution": False,
                "solution_indicators": []
            }
        
        # Solution action phrase OR complex system → Solution workflow
        if is_solution_action or is_complex_system:
            matched_indicators = []
            if is_solution_action:
                matched_indicators.append("solution_action_phrase")
            if is_complex_system:
                matched_indicators.append("complex_system_phrase")
            
            logger.info(
                f"[INTENT_RULE] Solution request detected: '{query[:50]}' "
                f"(action={is_solution_action}, complex={is_complex_system})"
            )
            return {
                "success": True,
                "intent": "solution",
                "confidence": 0.85,
                "next_step": None,
                "extracted_info": {
                    "rule_based": True,
                    "is_solution_action": is_solution_action,
                    "is_complex_system": is_complex_system
                },
                "is_solution": True,
                "solution_indicators": matched_indicators
            }
        
        # Default: Simple product request → Instrument Identifier
        logger.info(f"[INTENT_RULE] Simple product request detected: '{query[:50]}'")
        return {
            "success": True,
            "intent": "requirements",
            "confidence": 0.9,
            "next_step": None,
            "extracted_info": {"rule_based": True},
            "is_solution": False,
            "solution_indicators": []
        }

    # PHASE 3 FIX: Solution design phrases that don't start with product request starters
    # These are explicit "I'm building/designing" statements
    solution_design_starters = [
        "i'm designing", "i'm building", "i'm implementing",
        "i'm creating", "i'm planning", "i'm developing",
        "we're designing", "we're building", "we're implementing",
        "we are designing", "we are building", "we are implementing"
    ]
    if any(query.startswith(s) for s in solution_design_starters):
        logger.info(f"[INTENT_RULE] Solution design phrase detected: '{query[:50]}'")
        return {
            "success": True,
            "intent": "solution",
            "confidence": 0.9,
            "next_step": None,
            "extracted_info": {"rule_based": True, "is_design_statement": True},
            "is_solution": True,
            "solution_indicators": ["design_statement"]
        }

    # No rule matched - need LLM
    return None


# ============================================================================
# TOOLS
# ============================================================================

@tool("classify_intent", args_schema=ClassifyIntentInput)
def classify_intent_tool(
    user_input: str,
    current_step: Optional[str] = None,
    context: Optional[str] = None
) -> Dict[str, Any]:
    """
    Classify user intent for routing in the procurement workflow.
    Returns intent type, confidence, and suggested next step.

    OPTIMIZATION: Tries rule-based classification first to avoid LLM calls
    for obvious intents (greetings, confirms, knowledge questions, simple requests).
    Falls back to LLM with invoke_with_retry_fallback for ambiguous queries.
    """
    # =========================================================================
    # STEP 1: Try rule-based classification (no LLM call)
    # =========================================================================
    rule_result = _classify_rule_based(user_input)
    if rule_result is not None:
        return rule_result

    # =========================================================================
    # STEP 2: Fall back to LLM classification
    # =========================================================================
    logger.info(f"[INTENT_LLM] Rule-based didn't match, calling LLM for: '{user_input[:60]}'")

    try:
        llm = create_llm_with_fallback(
            model="gemini-2.5-flash",
            temperature=0.1,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        prompt = ChatPromptTemplate.from_template(INTENT_CLASSIFICATION_PROMPT)
        parser = JsonOutputParser()

        chain = prompt | llm | parser

        # Use retry wrapper with automatic key rotation and OpenAI fallback
        result = invoke_with_retry_fallback(
            chain,
            {
                "user_input": user_input,
                "current_step": current_step or "start",
                "context": context or "New conversation"
            },
            max_retries=3,
            fallback_to_openai=True,
            model="gemini-2.5-flash",
            temperature=0.1
        )

        return {
            "success": True,
            "intent": result.get("intent", "unrelated"),
            "confidence": result.get("confidence", 0.5),
            "next_step": result.get("next_step"),
            "extracted_info": result.get("extracted_info", {}),
            "is_solution": result.get("is_solution", False),
            "solution_indicators": result.get("solution_indicators", [])
        }

    except Exception as e:
        logger.error(f"Intent classification failed: {e}")
        return {
            "success": False,
            "intent": "unrelated",
            "confidence": 0.0,
            "next_step": None,
            "error": str(e)
        }


@tool("extract_requirements", args_schema=ExtractRequirementsInput)
def extract_requirements_tool(user_input: str) -> Dict[str, Any]:
    """
    Extract structured technical requirements from user input.
    Identifies product type, specifications, and infers missing common specs.

    Uses invoke_with_retry_fallback for automatic retry, key rotation,
    and OpenAI fallback on RESOURCE_EXHAUSTED errors.
    """
    try:
        llm = create_llm_with_fallback(
            model="gemini-2.5-flash",
            temperature=0.1,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        prompt = ChatPromptTemplate.from_template(REQUIREMENTS_EXTRACTION_PROMPT)
        parser = JsonOutputParser()

        chain = prompt | llm | parser

        # Use retry wrapper with automatic key rotation and OpenAI fallback
        result = invoke_with_retry_fallback(
            chain,
            {"user_input": user_input},
            max_retries=3,
            fallback_to_openai=True,
            model="gemini-2.5-flash",
            temperature=0.1
        )

        return {
            "success": True,
            "product_type": result.get("product_type"),
            "specifications": result.get("specifications", {}),
            "inferred_specs": result.get("inferred_specs", {}),
            "raw_requirements_text": result.get("raw_requirements_text", user_input)
        }

    except Exception as e:
        logger.error(f"Requirements extraction failed: {e}")
        return {
            "success": False,
            "product_type": None,
            "specifications": {},
            "error": str(e)
        }
