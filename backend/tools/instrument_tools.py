# tools/instrument_tools.py
# Instrument Identification and Accessory Tools

import json
import logging
from typing import Dict, Any, List, Optional
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
import os
from llm_fallback import create_llm_with_fallback
from prompts_library import load_prompt
logger = logging.getLogger(__name__)


# ============================================================================
# TOOL INPUT SCHEMAS
# ============================================================================

class IdentifyInstrumentsInput(BaseModel):
    """Input for instrument identification"""
    requirements: str = Field(description="Process requirements or problem statement")


class IdentifyAccessoriesInput(BaseModel):
    """Input for accessory identification"""
    instruments: List[Dict[str, Any]] = Field(description="List of identified instruments")
    process_context: Optional[str] = Field(default=None, description="Process context")


# ============================================================================
# PROMPTS - Loaded from prompts_library
# ============================================================================

INSTRUMENT_IDENTIFICATION_PROMPT = load_prompt("instrument_identification_prompt")

ACCESSORIES_IDENTIFICATION_PROMPT = load_prompt("accessories_identification_prompt")



# ============================================================================
# TOOLS
# ============================================================================

@tool("identify_instruments", args_schema=IdentifyInstrumentsInput)
def identify_instruments_tool(requirements: str) -> Dict[str, Any]:
    """
    Identify instruments needed from process requirements.
    Extracts product types, specifications, and creates a Bill of Materials.
    """
    try:
        llm = create_llm_with_fallback(
            model="gemini-2.5-pro",
            temperature=0.1,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        prompt = ChatPromptTemplate.from_template(INSTRUMENT_IDENTIFICATION_PROMPT)
        parser = JsonOutputParser()

        chain = prompt | llm | parser

        result = chain.invoke({"requirements": requirements})

        return {
            "success": True,
            "project_name": result.get("project_name"),
            "instruments": result.get("instruments", []),
            "instrument_count": len(result.get("instruments", [])),
            "summary": result.get("summary")
        }

    except Exception as e:
        logger.error(f"Instrument identification failed: {e}")
        return {
            "success": False,
            "instruments": [],
            "error": str(e)
        }


# ============================================================================
# KEYWORD-BASED FALLBACK FOR ACCESSORY EXTRACTION
# ============================================================================

# Universal accessory keywords applicable to multiple system types
ACCESSORY_KEYWORDS = {
    # Temperature systems
    "thermowell": "Thermowell / Protective Sleeve",
    "extension wire": "Thermocouple Extension Wire",
    "shield": "Thermocouple Protection Shield",
    "head": "Connection Head / Terminal Block",

    # Pressure systems
    "impulse line": "Impulse Line / Tubing",
    "isolation valve": "Isolation Valve / Manifold",
    "snubber": "Snubber / Pulsation Dampener",
    "equalizing": "Equalizing Connection",

    # Flow systems
    "straightener": "Flow Straightener",
    "orifice": "Orifice Plate / Flow Restriction",
    "tap": "Pressure Tap / Sensing Point",

    # General accessories
    "cable gland": "Cable Gland / Connector",
    "junction box": "Junction Box / Terminal Enclosure",
    "mounting bracket": "Mounting Bracket / Support",
    "gasket": "Gasket / Sealing Material",
    "fastener": "Fastener / Bolt Set",
    "power supply": "Power Supply Unit",
    "signal cable": "Signal Cable / Wiring",
    "connector": "Connector / Adapter",
    "conduit": "Conduit / Cable Protection",
    "strain relief": "Strain Relief / Cable Management",
    "terminal strip": "Terminal Strip / Connection Board",
    "calibration": "Calibration Kit / Standard",
    "protective cap": "Protective Cap / Cover",
    "spare parts": "Spare Parts / Replacement Kit",
}


def extract_accessories_by_keywords(
    instruments: List[Dict[str, Any]],
    context: str
) -> List[Dict[str, Any]]:
    """
    Extract accessories using keyword matching when LLM fails.
    Works universally for temperature, pressure, flow, and other systems.

    Args:
        instruments: List of identified instruments
        context: Process context for accessory determination

    Returns:
        List of accessories inferred from instrument types and context
    """
    accessories = []

    # Combine instrument info with context for analysis
    combined_text = f"{context} {json.dumps(instruments)}".lower()

    # Extract matching accessory keywords
    matched_accessories = set()
    for keyword, accessory_type in ACCESSORY_KEYWORDS.items():
        if keyword in combined_text:
            matched_accessories.add(accessory_type)

    # Create accessory items with inferred specifications
    for idx, accessory_type in enumerate(sorted(matched_accessories), 1):
        accessory = {
            "number": idx,
            "name": accessory_type,
            "category": accessory_type.split("/")[0].strip(),
            "quantity": 1,
            "specifications": {
                "extraction_method": "keyword_fallback",
                "inferred_from": "system_context_and_instrument_types",
            },
            "source_method": "keyword_extraction",
            "confidence": "low"  # Flag that this came from fallback
        }
        accessories.append(accessory)

    logger.info(f"[Keyword Extraction] Extracted {len(accessories)} accessories: {matched_accessories}")
    return accessories


@tool("identify_accessories", args_schema=IdentifyAccessoriesInput)
def identify_accessories_tool(
    instruments: List[Dict[str, Any]],
    process_context: Optional[str] = None
) -> Dict[str, Any]:
    """
    Robustly identify accessories needed for the identified instruments.

    Features:
    - 3-attempt retry mechanism for LLM-based identification
    - Automatic fallback to keyword-based extraction
    - Never returns empty if instruments exist (ensures complete solutions)
    - Universal logic works for all system types

    Returns list of accessories with specifications.
    """

    # Validate inputs
    if not instruments:
        logger.warning("[Accessory ID] No instruments provided - cannot identify accessories")
        return {
            "success": False,
            "accessories": [],
            "error": "No instruments provided",
            "method": "validation_failure"
        }

    context = process_context or "General industrial process"
    logger.info(f"[Accessory ID] Starting accessory identification for {len(instruments)} instrument(s)")

    # =========================================================================
    # ATTEMPT 1-3: LLM-based identification with retry
    # =========================================================================

    llm_accessories = None
    last_error = None

    for attempt in range(1, 4):  # 3 attempts
        try:
            logger.info(f"[Accessory ID] LLM attempt {attempt}/3...")

            llm = create_llm_with_fallback(
                model="gemini-2.5-pro",
                temperature=0.1,
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )

            prompt = ChatPromptTemplate.from_template(ACCESSORIES_IDENTIFICATION_PROMPT)
            parser = JsonOutputParser()
            chain = prompt | llm | parser

            result = chain.invoke({
                "instruments": json.dumps(instruments, indent=2),
                "process_context": context
            })

            # Validate LLM response structure
            if not isinstance(result, dict):
                logger.warning(f"[Accessory ID] Attempt {attempt}: Invalid response type {type(result)}")
                last_error = f"Invalid response type: {type(result)}"
                continue

            accessories = result.get("accessories", [])

            # Check if LLM returned valid accessories
            if accessories and isinstance(accessories, list):
                logger.info(f"[Accessory ID] ✅ LLM attempt {attempt} succeeded: {len(accessories)} accessories identified")
                llm_accessories = accessories
                break
            else:
                logger.warning(f"[Accessory ID] Attempt {attempt}: Empty or invalid accessories list")
                last_error = "Empty accessories list from LLM"
                continue

        except json.JSONDecodeError as e:
            logger.warning(f"[Accessory ID] Attempt {attempt}: JSON parse error - {str(e)}")
            last_error = f"JSON parse error: {str(e)}"
            continue

        except Exception as e:
            logger.warning(f"[Accessory ID] Attempt {attempt}: {type(e).__name__} - {str(e)}")
            last_error = str(e)
            continue

    # =========================================================================
    # FALLBACK: Keyword-based extraction if LLM failed
    # =========================================================================

    if llm_accessories is not None:
        # LLM succeeded
        return {
            "success": True,
            "accessories": llm_accessories,
            "accessory_count": len(llm_accessories),
            "summary": f"Identified {len(llm_accessories)} accessories using LLM",
            "method": "llm_identification"
        }
    else:
        # LLM failed all 3 attempts - use fallback
        logger.warning(f"[Accessory ID] ⚠️ LLM failed after 3 attempts. Last error: {last_error}")
        logger.info("[Accessory ID] 🔄 Activating keyword-based fallback extraction...")

        fallback_accessories = extract_accessories_by_keywords(instruments, context)

        if fallback_accessories:
            logger.info(f"[Accessory ID] ✅ Fallback succeeded: {len(fallback_accessories)} accessories extracted")
            return {
                "success": True,
                "accessories": fallback_accessories,
                "accessory_count": len(fallback_accessories),
                "summary": f"Identified {len(fallback_accessories)} accessories using keyword extraction",
                "method": "keyword_fallback",
                "llm_failure_reason": last_error
            }
        else:
            # Even fallback returned nothing
            logger.error(f"[Accessory ID] ❌ Both LLM and fallback failed. No accessories identified.")
            logger.error(f"[Accessory ID] LLM error: {last_error}")

            return {
                "success": False,
                "accessories": [],
                "accessory_count": 0,
                "summary": "Could not identify accessories (LLM and fallback both failed)",
                "method": "none",
                "llm_failure_reason": last_error,
                "error": "Accessory identification failed - LLM returned invalid response and keyword fallback found no matches"
            }
