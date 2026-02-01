"""
Test suite for Deep Agent Enrichment Fixes
Tests validation, spec limits, deduplication, and hallucination prevention
"""

import pytest
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentic.deep_agent.optimized_parallel_agent import (
    _is_valid_spec_key,
    _clean_and_flatten_specs,
    _normalize_spec_key,
    MAX_SPECS_PER_ITEM,
    deduplicate_and_merge_specifications,
)


class TestHallucinationDetection:
    """Test that hallucinated keys are rejected."""

    def test_reject_loss_of_ambition(self):
        """Should reject 'loss_of_ambition' pattern."""
        assert not _is_valid_spec_key("Cable_Tray_Cable_Protection_Against_Loss_Of_Ambition")

    def test_reject_loss_of_appetite(self):
        """Should reject 'loss_of_appetite' pattern."""
        assert not _is_valid_spec_key("Cable_Tray_Cable_Protection_Against_Loss_Of_Appetite")

    def test_reject_self_self_pattern(self):
        """Should reject recursive 'self-self-' pattern."""
        assert not _is_valid_spec_key("Cable_Tray_Cable_Protection_Against_Loss_Of_Self-self-self-acceptance")

    def test_accept_valid_keys(self):
        """Should accept valid technical specification keys."""
        assert _is_valid_spec_key("Operating_Temperature_Range")
        assert _is_valid_spec_key("Material_Of_Construction")
        assert _is_valid_spec_key("Pressure_Rating")


class TestSpecCountLimit:
    """Test that specification count stays under 60."""

    def test_max_specs_constant(self):
        """Verify MAX_SPECS_PER_ITEM is set to 60."""
        assert MAX_SPECS_PER_ITEM == 60

    def test_clean_and_flatten_respects_validation(self):
        """Test that _clean_and_flatten_specs filters invalid keys."""
        raw_specs = {
            "Operating_Temperature": {"value": "0-50°C", "confidence": 0.9},
            "Loss_Of_Ambition": {"value": "N/A", "confidence": 0.5},
            "Pressure_Rating": {"value": "10 bar", "confidence": 0.95},
        }

        cleaned = _clean_and_flatten_specs(raw_specs)

        # Should only keep valid specs
        assert "Operating_Temperature" in cleaned
        assert "Pressure_Rating" in cleaned
        assert "Loss_Of_Ambition" not in cleaned
        assert len(cleaned) == 2

    def test_filters_not_specified_values(self):
        """Test that 'Not specified' values are filtered."""
        raw_specs = {
            "key1": {"value": "Not specified (STANDARDS)", "confidence": 0.5},
            "key2": {"value": "10-50°C", "confidence": 0.8},
        }

        cleaned = _clean_and_flatten_specs(raw_specs)

        assert len(cleaned) == 1
        assert "key2" in cleaned


class TestSpecKeyNormalization:
    """Test specification key normalization for deduplication."""

    def test_normalize_consistency(self):
        """Different formats of same key should normalize identically."""
        key1 = "Operating Temperature Range"
        key2 = "operating-temperature-range"
        key3 = "OPERATING_TEMPERATURE_RANGE"

        assert _normalize_spec_key(key1) == _normalize_spec_key(key2) == _normalize_spec_key(key3)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
