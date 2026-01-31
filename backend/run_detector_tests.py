#!/usr/bin/env python
"""
Direct test of standards_detector.py without Flask dependencies
"""
import sys
sys.path.insert(0, '.')

# Import just the detector module
import importlib.util
spec = importlib.util.spec_from_file_location("standards_detector", "agentic/standards_detector.py")
detector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(detector)

print("=" * 70)
print("STANDARDS DETECTION TEST SUITE")
print("=" * 70)

test_count = 0
pass_count = 0
fail_count = 0

def run_test(name, test_func):
    global test_count, pass_count, fail_count
    test_count += 1
    try:
        test_func()
        print(f"[PASS] {name}")
        pass_count += 1
        return True
    except AssertionError as e:
        print(f"[FAIL] {name}: {e}")
        fail_count += 1
        return False
    except Exception as e:
        print(f"[ERROR] {name}: {e}")
        fail_count += 1
        return False

# Test 1: SIL detection
def test_sil_detection():
    result = detector.detect_standards_indicators("I need a SIL 2 certified transmitter")
    assert result["detected"] == True, "Should detect SIL"
    assert result["confidence"] >= 0.5, "Should have confidence >= 0.5"
    assert "safety_standards" in result["indicators"], "Should include safety_standards indicator"

run_test("Test 1: SIL Standard Detection", test_sil_detection)

# Test 2: ATEX detection
def test_atex_detection():
    result = detector.detect_standards_indicators("I need ATEX Zone 1 equipment for hazardous area")
    assert result["detected"] == True, "Should detect ATEX"
    assert result["confidence"] >= 0.5, "Should have confidence >= 0.5"
    assert result["indicators"], "Should have indicators"
    assert any(ind in result["indicators"] for ind in ["safety_standards", "domain_keywords"]), \
           f"Should detect standards or domain keywords, got: {result['indicators']}"

run_test("Test 2: ATEX Standard Detection", test_atex_detection)

# Test 3: API standards
def test_api_standards():
    result = detector.detect_standards_indicators("pressure relief valves per API 520 standard")
    assert result["detected"] == True, f"Should detect API, got: detected={result['detected']}, indicators={result['indicators']}"
    assert result["confidence"] > 0.0, "Should have confidence > 0"
    assert "process_standards" in result["indicators"], "Should include process_standards indicator"

run_test("Test 3: API Standard Detection", test_api_standards)

# Test 4: Oil & Gas domain
def test_oil_gas_domain():
    result = detector.detect_standards_indicators("crude oil distillation unit design")
    assert result["detected"] == True, "Should detect Oil & Gas domain"
    assert "domain_keywords" in result["indicators"], "Should include domain_keywords"

run_test("Test 4: Oil & Gas Domain Detection", test_oil_gas_domain)

# Test 5: Pharma domain
def test_pharma_domain():
    result = detector.detect_standards_indicators("pharmaceutical manufacturing GMP compliance")
    assert result["detected"] == True, "Should detect Pharma domain"

run_test("Test 5: Pharma Domain Detection", test_pharma_domain)

# Test 6: Hazardous area keyword
def test_hazardous_area():
    result = detector.detect_standards_indicators("explosion-proof equipment")
    assert result["detected"] == True, "Should detect hazardous area"
    assert "domain_keywords" in result["indicators"], "Should include domain_keywords"

run_test("Test 6: Hazardous Area Detection", test_hazardous_area)

# Test 7: Simple request (NO standards)
def test_simple_request():
    result = detector.detect_standards_indicators("I need a temperature sensor for my plant")
    assert result["detected"] == False, "Should NOT detect standards"
    assert result["confidence"] == 0.0, "Should have 0.0 confidence"
    assert len(result["indicators"]) == 0, "Should have no indicators"

run_test("Test 7: Simple Request (No Standards)", test_simple_request)

# Test 8: Basic transmitter (NO standards)
def test_basic_transmitter():
    result = detector.detect_standards_indicators("I need a pressure transmitter")
    assert result["detected"] == False, "Should NOT detect standards"

run_test("Test 8: Basic Transmitter (No Standards)", test_basic_transmitter)

# Test 9: Requirements dict with SIL
def test_requirements_sil():
    provided = {"sil_level": "SIL 2", "domain": "Oil & Gas"}
    result = detector.detect_standards_indicators("I need a transmitter", provided)
    assert result["detected"] == True, "Should detect from requirements"
    assert "sil_level_requirement" in result["indicators"], "Should include sil_level_requirement"

run_test("Test 9: Detection from Requirements (SIL)", test_requirements_sil)

# Test 10: Requirements dict with hazardous_area
def test_requirements_hazard():
    provided = {"hazardous_area": True, "atex_zone": "Zone 1"}
    result = detector.detect_standards_indicators("I need a sensor", provided)
    assert result["detected"] == True, "Should detect from requirements"

run_test("Test 10: Detection from Requirements (Hazard)", test_requirements_hazard)

# Test 11: Confidence levels increase with indicators
def test_confidence_progression():
    r1 = detector.detect_standards_indicators("SIL")
    r2 = detector.detect_standards_indicators("SIL ATEX Zone 1")
    r3 = detector.detect_standards_indicators("SIL ATEX Zone 1 Oil & Gas hazardous")

    assert r1["confidence"] <= r2["confidence"], "2 indicators should have >= confidence than 1"
    assert r2["confidence"] <= r3["confidence"], "3+ indicators should have >= confidence than 2"

run_test("Test 11: Confidence Progression", test_confidence_progression)

# Test 12: Convenience function - True case
def test_convenience_true():
    result = detector.should_run_standards_enrichment("I need SIL 2 ATEX equipment")
    assert result == True, "Should return True"

run_test("Test 12: Convenience Function (True)", test_convenience_true)

# Test 13: Convenience function - False case
def test_convenience_false():
    result = detector.should_run_standards_enrichment("I need a temperature sensor")
    assert result == False, "Should return False"

run_test("Test 13: Convenience Function (False)", test_convenience_false)

# Test 14: Matched keywords populated
def test_matched_keywords():
    result = detector.detect_standards_indicators("SIL 2 ATEX Zone 1")
    assert len(result["matched_keywords"]) > 0, "Should have matched keywords"
    assert any("sil" in kw.lower() for kw in result["matched_keywords"]), "Should include sil keyword"

run_test("Test 14: Matched Keywords Populated", test_matched_keywords)

# Test 15: Empty input
def test_empty_input():
    result = detector.detect_standards_indicators("")
    assert result["detected"] == False, "Should NOT detect empty input"
    assert result["confidence"] == 0.0, "Should have 0.0 confidence"

run_test("Test 15: Empty Input Handling", test_empty_input)

# Test 16: None requirements
def test_none_requirements():
    result = detector.detect_standards_indicators("I need a sensor", None)
    assert result["detected"] == False, "Should handle None requirements"

run_test("Test 16: None Requirements Handling", test_none_requirements)

# Test 17: Empty requirements dict
def test_empty_requirements():
    result = detector.detect_standards_indicators("I need a sensor", {})
    assert result["detected"] == False, "Should handle empty requirements"

run_test("Test 17: Empty Requirements Dict", test_empty_requirements)

# Test 18: Case insensitive matching
def test_case_insensitive():
    r1 = detector.detect_standards_indicators("SIL 2")
    r2 = detector.detect_standards_indicators("sil 2")
    r3 = detector.detect_standards_indicators("Sil 2")

    assert r1["detected"] == r2["detected"] == r3["detected"], "Should be case insensitive"

run_test("Test 18: Case Insensitive Matching", test_case_insensitive)

# Test 19: Multiple domain keywords
def test_multiple_domains():
    text = "Oil & gas facility with pharmaceutical manufacturing and chemical processing"
    result = detector.detect_standards_indicators(text)
    assert result["detected"] == True, "Should detect multiple domains"
    assert result["confidence"] > 0.0, "Should have positive confidence"

run_test("Test 19: Multiple Domain Keywords", test_multiple_domains)

# Test 20: Detection source tracking
def test_detection_source():
    r1 = detector.detect_standards_indicators("SIL 2")
    assert r1["detection_source"] != "", "Should track detection source"

    r2 = detector.detect_standards_indicators("I need a sensor", {"sil_level": "SIL 2"})
    assert r2["detection_source"] != "", "Should track detection source from requirements"

run_test("Test 20: Detection Source Tracking", test_detection_source)

# Print summary
print()
print("=" * 70)
print(f"TEST RESULTS: {pass_count}/{test_count} PASSED")
print("=" * 70)

if fail_count > 0:
    print(f"FAILED: {fail_count} test(s)")
    sys.exit(1)
else:
    print("ALL TESTS PASSED!")
    sys.exit(0)
