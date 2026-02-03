
import logging
import sys
import os
from datetime import datetime

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_labeling_fix")

def test_labeling_fix():
    """Test that 'Not specified' values are filtered out."""
    from agentic.deep_agent.optimized_parallel_agent import deduplicate_and_merge_specifications
    
    user_specs = {"Voltage": "24V"}
    
    standards_specs = {
        "Pressure": "100 psi",
        "Material": "not specified",  # Should be filtered
        "Grade": "not_specified",     # Should be filtered
        "Coating": "Not Specified",   # Should be filtered (case insensitive)
        "ValidSpec": "A516"
    }
    
    llm_specs = {
        "Temperature": "50C",
        "Length": "Unknown"           # Should be filtered
    }
    
    logger.info("Running deduplication...")
    merged = deduplicate_and_merge_specifications(user_specs, llm_specs, standards_specs)
    
    # Assertions
    assert "Voltage" in merged
    assert "Pressure" in merged
    assert "ValidSpec" in merged
    assert "Temperature" in merged
    
    # Verify filtering
    assert "Material" not in merged, "Material 'not specified' was not filtered"
    assert "Grade" not in merged, "Grade 'not_specified' was not filtered"
    assert "Coating" not in merged, "Coating 'Not Specified' was not filtered"
    assert "Length" not in merged, "Length 'Unknown' was not filtered"
    
    # Verify value content
    assert merged["Voltage"]["source"] == "user_specified"
    assert merged["Pressure"]["source"] == "standards"
    assert merged["Temperature"]["source"] == "llm_generated"
    
    logger.info("Labeling fix verification PASSED!")

if __name__ == "__main__":
    test_labeling_fix()
