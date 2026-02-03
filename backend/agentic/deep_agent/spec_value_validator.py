# agentic/deep_agent/spec_value_validator.py
# =============================================================================
# SPECIFICATION VALUE VALIDATOR
# =============================================================================
#
# Validates that specification values contain actual technical data
# instead of generic descriptive text, procedural requirements, or lists.
#
# Based on implementation plan: Fixing Pressure Relief Valve Specification Values
# =============================================================================

import logging
import re
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# VALUE TYPE CLASSIFICATION
# =============================================================================

class SpecValueType(Enum):
    """Classification of specification value types."""
    TECHNICAL_VALUE = "technical_value"  # Actual measurable value (VALID)
    PROCEDURAL_TEXT = "procedural_text"  # Procedure/requirement (INVALID)
    EQUIPMENT_LIST = "equipment_list"    # List of equipment types (INVALID)
    FREQUENCY_TEXT = "frequency_text"    # Schedule/frequency text (BORDERLINE)
    GENERIC_TEXT = "generic_text"        # Generic description (INVALID)
    SELECTION_CRITERIA = "selection_criteria"  # Selection requirements (INVALID)
    EMPTY_VALUE = "empty_value"          # Empty/N/A (INVALID)


class SourceTag(Enum):
    """Source tags for specifications."""
    DATASHEET = "DATASHEET"      # From equipment datasheet
    NAMEPLATE = "NAMEPLATE"      # From physical nameplate
    CALIBRATION = "CALIBRATION"  # From calibration certificate
    CONFIGURATION = "CONFIGURATION"  # From DCS/control system
    REQUIREMENT = "REQUIREMENT"  # From standards document
    USER_INPUT = "USER_INPUT"    # Manually entered
    STANDARDS = "STANDARDS"      # Legacy standards tag


@dataclass
class ValidationResult:
    """Result of specification value validation."""
    is_valid: bool
    value_type: SpecValueType
    reason: str
    suggested_source_tag: Optional[SourceTag] = None
    suggested_action: Optional[str] = None


# =============================================================================
# VALIDATION PATTERNS
# =============================================================================

# Patterns that indicate procedural/requirement text (INVALID)
PROCEDURAL_PATTERNS = [
    # Safety procedures
    r'\b(lockout|tagout|LOTO|permit|vent|depressurize)\b',
    r'\b(appropriate|suitable)\s+PPE\b',
    r'\b(work\s+permit|gas\s+testing)\b',
    
    # Maintenance procedures
    r'\b(inspect|clean|replace|lubricate|maintain)\b',
    r'\b(routine\s+inspection|visual\s+check)\b',
    r'\b(remove\s+from\s+service|tag|repair)\b',
    
    # Requirements/standards language
    r'\b(must|should|shall|ensure|require)\b',
    r'\b(comply|accordance|per|as\s+per)\b',
    r'\b(refer\s+to|consult|see)\b',
]

# Patterns that indicate equipment/resource lists (INVALID for specs)
EQUIPMENT_LIST_PATTERNS = [
    r'(?:^|\s)((?:[A-Z][a-z]+\s+)*(?:tester|pump|calibrator|barrier|isolator|device))s?\s*,',
    r'(?:HART|Modbus|Profibus|Foundation Fieldbus).*,.*(?:HART|Modbus|Profibus)',
    r'(?:^|\s)((?:[A-Z][a-z]+\s+)+)(?:and|or)\s+(?:[A-Z][a-z]+\s+)+',
]

# Patterns for generic descriptive text (INVALID)
GENERIC_TEXT_PATTERNS = [
    r'\b(typically|usually|generally|commonly|often)\b',
    r'\b(depends?\s+on|varies?\s+with|application[- ](?:specific|dependent))\b',
    r'\b(various|multiple|several|different)\s+(?:types|models|options)\b',
    r'\b(may|might|can)\s+(?:be|include|vary)\b',
]

# Patterns for selection criteria (INVALID for actual specs)
SELECTION_CRITERIA_PATTERNS = [
    r'\b(selection\s+criteria|choose|select|consider)\b',
    r'\b(gas\s+group|temperature\s+class|matching\s+with)\b',
    r'\b(trained|qualified|certified)\s+technicians?\b',
    r'\b(topics|subjects|areas)\s*:',
]

# Patterns for technical values (VALID)
TECHNICAL_VALUE_PATTERNS = [
    # Numerical values with units
    r'\d+(?:\.\d+)?\s*(?:mm|cm|m|inch|in|ft)',  # Length
    r'\d+(?:\.\d+)?\s*(?:bar|psi|kPa|MPa|Pa)',  # Pressure
    r'[±]?\d+(?:\.\d+)?\s*(?:°C|°F|K|R)',  # Temperature
    r'\d+-?\d*\s*(?:mA|A|V|W|VA)',  # Electrical
    r'[±]\s*\d+(?:\.\d+)?\s*%',  # Percentage/Accuracy
    r'\d+(?:\.\d+)?\s*(?:mm|inch|in)(?:\s+Class\s+\d+)?',  # Size with class
    
    # Specific ratings/standards
    r'\b(?:IP|NEMA)\s*\d+[A-Z]?\b',  # Protection ratings
    r'\bSIL\s*[1-4]\b',  # SIL ratings
    r'\b(?:ATEX|IECEx|CSA|UL|FM)\b',  # Certifications
    r'\b(?:316L?|304|Hastelloy|Inconel)\s*(?:SS|Stainless\s+Steel)?\b',  # Materials
    
    # Protocols (specific, not lists)
    r'^\s*(?:HART|Modbus\s+RTU|Foundation\s+Fieldbus\s+H1|Profibus\s+PA)\s*$',
]

# Field names that indicate procedural/requirement content (not technical specs)
PROCEDURAL_FIELD_NAMES = [
    'safety_procedure', 'work_requirements', 'ppe_requirement',
    'corrective_action', 'maintenance_action', 'training_topics',
    'documentation_requirements', 'technician_qualification',
]

# Field names that indicate lists (should have specific values, not types)
LIST_FIELD_NAMES = [
    'equipment_types', 'spare_parts_inventory', 'protocol_support',
    'cable_specification', 'network_architecture',
]


# =============================================================================
# VALUE VALIDATION FUNCTIONS
# =============================================================================

def validate_spec_value(
    key: str,
    value: Any,
    product_type: Optional[str] = None
) -> ValidationResult:
    """
    Validate if a specification value is a technical value or generic text.
    
    Args:
        key: The specification field name
        value: The specification value
        product_type: Optional product type for context
        
    Returns:
        ValidationResult with classification and recommendations
    """
    if value is None or str(value).strip() == "":
        return ValidationResult(
            is_valid=False,
            value_type=SpecValueType.EMPTY_VALUE,
            reason="Empty or null value",
            suggested_action="Remove this specification or request actual value"
        )
    
    value_str = str(value).strip()
    value_lower = value_str.lower()
    key_lower = key.lower()
    
    # Check for N/A or placeholder values
    if re.match(r'^(n/?a|none|null|not\s+applicable|-)$', value_lower):
        return ValidationResult(
            is_valid=False,
            value_type=SpecValueType.EMPTY_VALUE,
            reason="Placeholder value (N/A, None, etc.)",
            suggested_action="Remove this specification or provide actual value"
        )
    
    # Check if field name indicates procedural content
    if any(proc_field in key_lower for proc_field in PROCEDURAL_FIELD_NAMES):
        return ValidationResult(
            is_valid=False,
            value_type=SpecValueType.PROCEDURAL_TEXT,
            reason=f"Field name '{key}' indicates procedural content",
            suggested_source_tag=SourceTag.REQUIREMENT,
            suggested_action="Move to separate 'Requirements' or 'Procedures' section"
        )
    
    # Check for procedural text patterns
    for pattern in PROCEDURAL_PATTERNS:
        if re.search(pattern, value_str, re.IGNORECASE):
            return ValidationResult(
                is_valid=False,
                value_type=SpecValueType.PROCEDURAL_TEXT,
                reason=f"Contains procedural language: '{pattern}'",
                suggested_source_tag=SourceTag.REQUIREMENT,
                suggested_action="Move to 'Maintenance Procedures' or 'Safety Requirements'"
            )
    
    # Check for equipment lists
    for pattern in EQUIPMENT_LIST_PATTERNS:
        if re.search(pattern, value_str):
            return ValidationResult(
                is_valid=False,
                value_type=SpecValueType.EQUIPMENT_LIST,
                reason="Contains equipment/resource list instead of specific value",
                suggested_action="Replace with specific model number or part number"
            )
    
    # Check for generic text
    for pattern in GENERIC_TEXT_PATTERNS:
        if re.search(pattern, value_str, re.IGNORECASE):
            return ValidationResult(
                is_valid=False,
                value_type=SpecValueType.GENERIC_TEXT,
                reason=f"Generic descriptive text: '{pattern}'",
                suggested_action="Replace with actual measured or specified value"
            )
    
    # Check for selection criteria
    for pattern in SELECTION_CRITERIA_PATTERNS:
        if re.search(pattern, value_str, re.IGNORECASE):
            return ValidationResult(
                is_valid=False,
                value_type=SpecValueType.SELECTION_CRITERIA,
                reason="Contains selection criteria instead of actual value",
                suggested_source_tag=SourceTag.REQUIREMENT,
                suggested_action="Replace with chosen/specified value"
            )
    
    # Check if value is too long (> 200 chars likely descriptive text)
    if len(value_str) > 200:
        return ValidationResult(
            is_valid=False,
            value_type=SpecValueType.GENERIC_TEXT,
            reason=f"Value too long ({len(value_str)} chars) - likely description not spec",
            suggested_action="Extract actual technical value from description"
        )
    
    # Check for technical value patterns (VALID)
    has_technical_pattern = False
    for pattern in TECHNICAL_VALUE_PATTERNS:
        if re.search(pattern, value_str, re.IGNORECASE):
            has_technical_pattern = True
            break
    
    # Frequency/schedule values (borderline - acceptable if specific)
    frequency_patterns = [
        r'\b(?:\d+\s+)?(?:annually|yearly|monthly|weekly|daily)\b',
        r'\b(?:\d+\s+)?(?:years?|months?|weeks?|days?|hours?)\b',
        r'\bevery\s+\d+\s+(?:years?|months?)\b',
    ]
    
    is_frequency = any(re.search(p, value_lower) for p in frequency_patterns)
    
    if is_frequency:
        # Frequency is acceptable if specific
        if re.search(r'\d+', value_str):
            return ValidationResult(
                is_valid=True,
                value_type=SpecValueType.FREQUENCY_TEXT,
                reason="Specific frequency/schedule value",
                suggested_source_tag=SourceTag.REQUIREMENT
            )
        else:
            return ValidationResult(
                is_valid=False,
                value_type=SpecValueType.FREQUENCY_TEXT,
                reason="Generic frequency text without specific value",
                suggested_action="Specify exact interval (e.g., '12 months', '1 year')"
            )
    
    # If value contains numbers/units, it's likely valid
    has_numbers = bool(re.search(r'\d', value_str))
    has_units = bool(re.search(r'(?:mm|inch|°C|°F|psi|bar|mA|V|%|Class)', value_str, re.IGNORECASE))
    
    if has_technical_pattern or (has_numbers and has_units):
        # Determine likely source
        if re.search(r'\bcalibrat', key_lower):
            suggested_source = SourceTag.CALIBRATION
        elif re.search(r'(?:set|configured|programmed)', value_lower):
            suggested_source = SourceTag.CONFIGURATION
        elif re.search(r'(?:model|serial|manufacturer)', key_lower):
            suggested_source = SourceTag.NAMEPLATE
        else:
            suggested_source = SourceTag.DATASHEET
            
        return ValidationResult(
            is_valid=True,
            value_type=SpecValueType.TECHNICAL_VALUE,
            reason="Contains technical value with units/numbers",
            suggested_source_tag=suggested_source
        )
    
    # Default: Likely generic text if no technical patterns found
    return ValidationResult(
        is_valid=False,
        value_type=SpecValueType.GENERIC_TEXT,
        reason="No technical value patterns detected",
        suggested_action="Verify this is a technical specification and not a description"
    )


def validate_specifications(
    specs: Dict[str, Any],
    product_type: Optional[str] = None
) -> Dict[str, ValidationResult]:
    """
    Validate all specifications in a dictionary.
    
    Args:
        specs: Dictionary of specifications
        product_type: Optional product type for context
        
    Returns:
        Dictionary mapping spec keys to validation results
    """
    results = {}
    
    for key, value in specs.items():
        # Skip metadata fields
        if key.startswith('_'):
            continue
            
        # Handle nested specs
        if isinstance(value, dict) and 'value' in value:
            actual_value = value['value']
        else:
            actual_value = value
            
        results[key] = validate_spec_value(key, actual_value, product_type)
    
    return results


def get_validation_summary(
    validation_results: Dict[str, ValidationResult]
) -> Dict[str, Any]:
    """
    Generate a summary report of validation results.
    
    Args:
        validation_results: Dictionary of validation results
        
    Returns:
        Summary statistics
    """
    total = len(validation_results)
    valid = sum(1 for r in validation_results.values() if r.is_valid)
    invalid = total - valid
    
    by_type = {}
    for result in validation_results.values():
        type_name = result.value_type.value
        by_type[type_name] = by_type.get(type_name, 0) + 1
    
    invalid_specs = {
        key: result 
        for key, result in validation_results.items() 
        if not result.is_valid
    }
    
    return {
        "total_specs": total,
        "valid_specs": valid,
        "invalid_specs": invalid,
        "validity_rate": valid / total if total > 0 else 0,
        "by_type": by_type,
        "invalid_spec_details": invalid_specs
    }


def categorize_specifications(
    specs: Dict[str, Any]
) -> Dict[str, List[str]]:
    """
    Categorize specifications by recommended section.
    
    Args:
        specs: Dictionary of specifications
        
    Returns:
        Dictionary mapping section names to lists of spec keys
    """
    validation_results = validate_specifications(specs)
    
    categories = {
        "technical_specifications": [],
        "maintenance_procedures": [],
        "safety_requirements": [],
        "selection_criteria": [],
        "requires_clarification": []
    }
    
    for key, result in validation_results.items():
        if result.is_valid:
            categories["technical_specifications"].append(key)
        elif result.value_type == SpecValueType.PROCEDURAL_TEXT:
            if 'safety' in key.lower() or 'ppe' in key.lower():
                categories["safety_requirements"].append(key)
            else:
                categories["maintenance_procedures"].append(key)
        elif result.value_type == SpecValueType.SELECTION_CRITERIA:
            categories["selection_criteria"].append(key)
        else:
            categories["requires_clarification"].append(key)
    
    return categories


def print_validation_report(
    specs: Dict[str, Any],
    product_type: Optional[str] = None
):
    """
    Print a detailed validation report to console.
    
    Args:
        specs: Dictionary of specifications
        product_type: Optional product type for context
    """
    validation_results = validate_specifications(specs, product_type)
    summary = get_validation_summary(validation_results)
    categories = categorize_specifications(specs)
    
    print("\n" + "="*80)
    print("SPECIFICATION VALIDATION REPORT")
    if product_type:
        print(f"Product Type: {product_type}")
    print("="*80)
    
    print(f"\n📊 SUMMARY:")
    print(f"  Total Specifications: {summary['total_specs']}")
    print(f"  ✅ Valid: {summary['valid_specs']} ({summary['validity_rate']:.1%})")
    print(f"  ❌ Invalid: {summary['invalid_specs']}")
    
    print(f"\n📈 BY TYPE:")
    for type_name, count in summary['by_type'].items():
        print(f"  {type_name}: {count}")
    
    print(f"\n📁 RECOMMENDED CATEGORIZATION:")
    for section, keys in categories.items():
        if keys:
            print(f"  {section}: {len(keys)} items")
    
    if summary['invalid_specs'] > 0:
        print(f"\n❌ INVALID SPECIFICATIONS ({summary['invalid_specs']}):")
        for key, result in summary['invalid_spec_details'].items():
            value = specs.get(key, "")
            if isinstance(value, dict):
                value = value.get('value', value)
            value_preview = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
            
            print(f"\n  🔴 {key}")
            print(f"     Value: {value_preview}")
            print(f"     Type: {result.value_type.value}")
            print(f"     Reason: {result.reason}")
            if result.suggested_action:
                print(f"     💡 Action: {result.suggested_action}")
            if result.suggested_source_tag:
                print(f"     🏷️  Suggested Tag: {result.suggested_source_tag.value}")
    
    print("\n" + "="*80 + "\n")


# =============================================================================
# FILTERING FUNCTIONS
# =============================================================================

def filter_valid_specs(
    specs: Dict[str, Any],
    product_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Filter specifications to return only valid technical values.
    
    Args:
        specs: Dictionary of specifications
        product_type: Optional product type for context
        
    Returns:
        Dictionary containing only valid specifications
    """
    validation_results = validate_specifications(specs, product_type)
    
    filtered = {}
    for key, value in specs.items():
        if key.startswith('_'):
            filtered[key] = value
            continue
            
        result = validation_results.get(key)
        if result and result.is_valid:
            filtered[key] = value
        else:
            logger.debug(f"Filtered out invalid spec: {key} - {result.reason if result else 'unknown'}")
    
    return filtered


def separate_specs_by_source(
    specs: Dict[str, Any],
    product_type: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Separate specifications into different dictionaries by source type.
    
    Args:
        specs: Dictionary of specifications
        product_type: Optional product type for context
        
    Returns:
        Dictionary with keys: technical_specs, procedures, requirements
    """
    validation_results = validate_specifications(specs, product_type)
    
    separated = {
        "technical_specs": {},
        "procedures": {},
        "requirements": {},
        "invalid": {}
    }
    
    for key, value in specs.items():
        if key.startswith('_'):
            continue
            
        result = validation_results.get(key)
        if not result:
            continue
            
        if result.is_valid:
            separated["technical_specs"][key] = value
        elif result.value_type in (SpecValueType.PROCEDURAL_TEXT, SpecValueType.FREQUENCY_TEXT):
            separated["procedures"][key] = value
        elif result.value_type in (SpecValueType.SELECTION_CRITERIA, SpecValueType.EQUIPMENT_LIST):
            separated["requirements"][key] = value
        else:
            separated["invalid"][key] = value
    
    return separated


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "SpecValueType",
    "SourceTag",
    "ValidationResult",
    "validate_spec_value",
    "validate_specifications",
    "get_validation_summary",
    "categorize_specifications",
    "print_validation_report",
    "filter_valid_specs",
    "separate_specs_by_source",
]
