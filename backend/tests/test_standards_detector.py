# tests/test_standards_detector.py
# =============================================================================
# TESTS FOR STANDARDS DETECTION UTILITY
# =============================================================================

import pytest
from agentic.standards_detector import (
    detect_standards_indicators,
    should_run_standards_enrichment,
    StandardsDetectionResult
)


class TestStandardsDetection:
    """Test suite for standards detection functionality"""

    def test_detect_sil_standards(self):
        """Test detection of SIL standards in text"""
        user_input = "I need a SIL 2 certified transmitter"
        result = detect_standards_indicators(user_input)

        assert result["detected"] == True
        assert result["confidence"] > 0.5
        assert "safety_standards" in result["indicators"]

    def test_detect_atex_standards(self):
        """Test detection of ATEX standards"""
        user_input = "I need an ATEX Zone 1 temperature sensor for a hazardous area"
        result = detect_standards_indicators(user_input)

        assert result["detected"] == True
        assert result["confidence"] > 0.5
        assert any("atex" in ind.lower() or "hazard" in ind.lower()
                  for ind in result["indicators"])

    def test_detect_api_standards(self):
        """Test detection of API standards"""
        user_input = "We need pressure relief valves per API 520 standard"
        result = detect_standards_indicators(user_input)

        assert result["detected"] == True
        assert result["confidence"] > 0.5

    def test_detect_domain_keywords_oil_gas(self):
        """Test detection of Oil & Gas domain"""
        user_input = "I'm designing a crude oil distillation unit"
        result = detect_standards_indicators(user_input)

        assert result["detected"] == True
        assert "domain_keywords" in result["indicators"]

    def test_detect_domain_keywords_pharma(self):
        """Test detection of Pharma domain"""
        user_input = "We need sensors for pharmaceutical manufacturing with GMP compliance"
        result = detect_standards_indicators(user_input)

        assert result["detected"] == True

    def test_detect_hazardous_area(self):
        """Test detection of hazardous area keyword"""
        user_input = "We need explosion-proof instruments for our facility"
        result = detect_standards_indicators(user_input)

        assert result["detected"] == True
        assert "domain_keywords" in result["indicators"]

    def test_no_standards_simple_request(self):
        """Test that simple requests are NOT detected as standards"""
        user_input = "I need a temperature sensor for my plant"
        result = detect_standards_indicators(user_input)

        assert result["detected"] == False
        assert result["confidence"] == 0.0

    def test_no_standards_basic_transmitter(self):
        """Test basic transmitter request without standards"""
        user_input = "I need a pressure transmitter"
        result = detect_standards_indicators(user_input)

        assert result["detected"] == False

    def test_detect_from_requirements_dict(self):
        """Test detection from provided_requirements dict"""
        provided_requirements = {
            "sil_level": "SIL 2",
            "domain": "Oil & Gas"
        }
        result = detect_standards_indicators(
            user_input="I need a transmitter",
            provided_requirements=provided_requirements
        )

        assert result["detected"] == True
        assert "sil_level_requirement" in result["indicators"]

    def test_detect_from_hazardous_area_requirement(self):
        """Test detection from hazardous_area in requirements"""
        provided_requirements = {
            "hazardous_area": True,
            "atex_zone": "Zone 1"
        }
        result = detect_standards_indicators(
            user_input="I need a sensor",
            provided_requirements=provided_requirements
        )

        assert result["detected"] == True

    def test_confidence_levels(self):
        """Test that confidence increases with more indicators"""
        # Single indicator
        result1 = detect_standards_indicators("I need SIL certified equipment")
        conf1 = result1["confidence"]

        # Multiple indicators
        result2 = detect_standards_indicators(
            "I need SIL 2 certified ATEX equipment for hazardous oil & gas facility"
        )
        conf2 = result2["confidence"]

        assert conf2 > conf1

    def test_should_run_standards_enrichment_true(self):
        """Test convenience function returns True for standards"""
        should_run = should_run_standards_enrichment(
            "I need SIL 2 ATEX certified transmitter"
        )
        assert should_run == True

    def test_should_run_standards_enrichment_false(self):
        """Test convenience function returns False for non-standards"""
        should_run = should_run_standards_enrichment(
            "I need a temperature sensor"
        )
        assert should_run == False

    def test_matched_keywords_populated(self):
        """Test that matched keywords are captured"""
        user_input = "SIL 2 ATEX Zone 1 equipment"
        result = detect_standards_indicators(user_input)

        assert len(result["matched_keywords"]) > 0
        assert any("sil" in kw.lower() for kw in result["matched_keywords"])

    def test_empty_input(self):
        """Test handling of empty input"""
        result = detect_standards_indicators("")

        assert result["detected"] == False
        assert result["confidence"] == 0.0

    def test_none_requirements(self):
        """Test handling of None provided_requirements"""
        result = detect_standards_indicators(
            "I need a sensor",
            provided_requirements=None
        )

        assert result["detected"] == False

    def test_empty_requirements(self):
        """Test handling of empty requirements dict"""
        result = detect_standards_indicators(
            "I need a sensor",
            provided_requirements={}
        )

        assert result["detected"] == False

    def test_case_insensitive_detection(self):
        """Test that detection is case-insensitive"""
        result1 = detect_standards_indicators("I need SIL 2 equipment")
        result2 = detect_standards_indicators("I need sil 2 equipment")
        result3 = detect_standards_indicators("I need Sil 2 equipment")

        assert result1["detected"] == result2["detected"] == result3["detected"]

    def test_multiple_domains(self):
        """Test detection with multiple domain keywords"""
        user_input = (
            "Oil & gas facility with pharmaceutical manufacturing area "
            "and chemical processing"
        )
        result = detect_standards_indicators(user_input)

        assert result["detected"] == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
