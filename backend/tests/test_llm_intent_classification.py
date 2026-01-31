# tests/test_llm_intent_classification.py
"""
Test script for LLM-based intent classification.
Verifies that the new _classify_llm_fast() function correctly classifies
greetings, confirms, rejects, and exits using temperature 0.0.

Run from backend directory:
    python -m tests.test_llm_intent_classification
"""

import os
import sys
import time
import logging
from dotenv import load_dotenv

# Load environment variables (fix for missing GOOGLE_API_KEY)
load_dotenv()

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test cases for LLM classification
# QUICK_CLASSIFICATION handles: greeting, confirm, reject, exit, unknown
# CLASSIFICATION (full) handles: invalid_input, chat, search, solution
TEST_CASES = {
    # ===== QUICK_CLASSIFICATION intents (control flow) =====
    "greeting": [
        "hi",
        "hello",
        "hey",
        "good morning",
        "Hello there",
        "Hi!",
    ],
    "confirm": [
        "yes",
        "ok",
        "okay",
        "proceed",
        "continue",
        "go ahead",
        "sure",
        "Yeah",
    ],
    "reject": [
        "no",
        "nope",
        "cancel",
        "stop",
        "never mind",
        "forget it",
    ],
    "exit": [
        "start over",
        "reset",
        "new search",
        "begin again",
        "exit",
        "quit",
    ],
    # ===== NEW 4-INTENT ARCHITECTURE (via unknown -> full classification) =====
    "unknown": [
        # These should return None from QUICK_CLASSIFICATION and go to full LLM
        "I need a pressure transmitter",            # -> search
        "What is ATEX certification?",              # -> chat
        "Looking for a flow meter 0-100 GPM",       # -> search
        "Design a temperature monitoring system",   # -> solution
        "Hi, I need a sensor",                      # Should be unknown (mixed)
        "What's the weather today?",                # -> invalid_input
    ],
}


def test_classify_llm_fast():
    """Test the _classify_llm_fast function."""
    from tools.intent_tools import _classify_llm_fast
    
    print("\n" + "=" * 70)
    print("TESTING LLM-BASED FAST CLASSIFICATION (temperature=0.0)")
    print("=" * 70)
    
    results = {"passed": 0, "failed": 0, "errors": 0}
    detailed_results = []
    
    for expected_intent, test_inputs in TEST_CASES.items():
        print(f"\n--- Testing {expected_intent.upper()} intent ---")
        
        for user_input in test_inputs:
            try:
                start_time = time.time()
                result = _classify_llm_fast(user_input)
                elapsed_ms = (time.time() - start_time) * 1000
                
                # For 'unknown', we expect None (defers to full classification)
                if expected_intent == "unknown":
                    if result is None:
                        status = "PASS"
                        results["passed"] += 1
                        actual_intent = "unknown (None returned)"
                    else:
                        actual_intent = result.get("intent", "ERROR")
                        status = "FAIL"
                        results["failed"] += 1
                else:
                    if result is None:
                        status = "FAIL"
                        actual_intent = "None (expected {})".format(expected_intent)
                        results["failed"] += 1
                    else:
                        actual_intent = result.get("intent", "ERROR")
                        confidence = result.get("confidence", 0)
                        
                        if actual_intent == expected_intent:
                            status = "PASS"
                            results["passed"] += 1
                        else:
                            status = "FAIL"
                            results["failed"] += 1
                
                detailed_results.append({
                    "input": user_input,
                    "expected": expected_intent,
                    "actual": actual_intent,
                    "status": status,
                    "time_ms": elapsed_ms
                })
                
                print(f"  [{status}] '{user_input}' -> {actual_intent} ({elapsed_ms:.0f}ms)")
                
            except Exception as e:
                results["errors"] += 1
                detailed_results.append({
                    "input": user_input,
                    "expected": expected_intent,
                    "actual": f"ERROR: {e}",
                    "status": "ERROR",
                    "time_ms": 0
                })
                print(f"  [ERROR] '{user_input}' -> {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Passed: {results['passed']}")
    print(f"  Failed: {results['failed']}")
    print(f"  Errors: {results['errors']}")
    print(f"  Total:  {sum(results.values())}")
    
    # Calculate average latency
    latencies = [r["time_ms"] for r in detailed_results if r["time_ms"] > 0]
    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        print(f"\n  Average latency: {avg_latency:.0f}ms")
        print(f"  Min latency: {min(latencies):.0f}ms")
        print(f"  Max latency: {max(latencies):.0f}ms")
    
    return results


def test_consistency():
    """Test that temperature 0.0 produces consistent results."""
    from tools.intent_tools import _classify_llm_fast
    
    print("\n" + "=" * 70)
    print("TESTING CONSISTENCY (same input 5 times)")
    print("=" * 70)
    
    test_inputs = ["hello", "yes", "start over"]
    
    for user_input in test_inputs:
        print(f"\n  Testing '{user_input}' 5 times...")
        intents = []
        
        for i in range(5):
            result = _classify_llm_fast(user_input)
            if result:
                intents.append(result.get("intent", "unknown"))
            else:
                intents.append("unknown")
            time.sleep(0.5)  # Small delay to avoid rate limiting
        
        unique_intents = set(intents)
        if len(unique_intents) == 1:
            print(f"    [PASS] All 5 responses: '{intents[0]}'")
        else:
            print(f"    [FAIL] Inconsistent: {intents}")


def test_exit_detection():
    """Test the updated should_exit_workflow function."""
    print("\n" + "=" * 70)
    print("TESTING EXIT DETECTION (LLM-based)")
    print("=" * 70)
    
    try:
        from agentic.intent_classification_routing_agent import should_exit_workflow
    except ImportError:
        # Handle relative import from tests directory
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from agentic.intent_classification_routing_agent import should_exit_workflow
    
    exit_cases = [
        ("start over", True),
        ("reset", True),
        ("hello", True),  # Greeting also triggers exit (new conversation)
        ("hi", True),
        ("I need a pressure transmitter", False),
        ("What is ATEX?", False),
    ]
    
    for user_input, expected_exit in exit_cases:
        result = should_exit_workflow(user_input)
        status = "PASS" if result == expected_exit else "FAIL"
        print(f"  [{status}] '{user_input}' -> exit={result} (expected={expected_exit})")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("LLM-BASED INTENT CLASSIFICATION TEST SUITE")
    print("=" * 70)
    
    # Check for API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("\nERROR: GOOGLE_API_KEY environment variable not set!")
        print("Please set the API key and run again.")
        sys.exit(1)
    
    # Run tests
    try:
        # Test 1: Basic classification
        results = test_classify_llm_fast()
        
        # Test 2: Consistency
        test_consistency()
        
        # Test 3: Exit detection
        test_exit_detection()
        
        # Final status
        print("\n" + "=" * 70)
        print("TEST COMPLETE")
        print("=" * 70)
        
        if results["failed"] == 0 and results["errors"] == 0:
            print("\nAll tests PASSED!")
            sys.exit(0)
        else:
            print(f"\nSome tests FAILED: {results['failed']} failures, {results['errors']} errors")
            sys.exit(1)
            
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
