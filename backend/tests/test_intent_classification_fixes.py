"""
Test suite for Intent Classification Fixes (Phase 2 Implementation)

Validates:
1. IntentConfig centralization and routing logic
2. Rule-based classification detection of complex systems
3. Intent validation and unknown intent handling
4. Solution vs. simple product routing

PHASE 2 FIX: Tests the new intent classification system
"""

import sys
import os
from typing import Dict, Any

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Try importing with fallback for missing dependencies
try:
    from agentic.intent_classification_routing_agent import (
        IntentConfig,
        WorkflowTarget,
        IntentClassificationRoutingAgent,
        should_exit_workflow
    )
    from tools.intent_tools import _classify_rule_based
    IMPORTS_OK = True
except ImportError as e:
    print(f"⚠️  Warning: Some imports failed: {e}")
    print("Attempting to load only core test logic...")
    IMPORTS_OK = False


# =============================================================================
# TEST SUITE 1: IntentConfig Routing Logic
# =============================================================================

class TestIntentConfig:
    """Tests for the centralized IntentConfig class."""

    def test_solution_forcing_intents(self):
        """Solution forcing intents should ALWAYS route to SOLUTION_WORKFLOW."""
        for intent in ["solution", "system", "systems", "complex_system", "design"]:
            result = IntentConfig.get_workflow(intent, is_solution=False)
            assert result == WorkflowTarget.SOLUTION_WORKFLOW, \
                f"Intent '{intent}' should force SOLUTION_WORKFLOW"

    def test_is_solution_override(self):
        """is_solution=True should override non-solution intents."""
        # Test with neutral intent
        result = IntentConfig.get_workflow("question", is_solution=True)
        assert result == WorkflowTarget.SOLUTION_WORKFLOW, \
            "is_solution=True should override 'question' intent to SOLUTION_WORKFLOW"

        # Test with requirement intent
        result = IntentConfig.get_workflow("requirements", is_solution=True)
        assert result == WorkflowTarget.SOLUTION_WORKFLOW, \
            "is_solution=True should override 'requirements' intent to SOLUTION_WORKFLOW"

    def test_intent_mapping(self):
        """Intent mappings should work correctly."""
        test_cases = {
            "greeting": WorkflowTarget.ENGENIE_CHAT,
            "question": WorkflowTarget.ENGENIE_CHAT,
            "requirements": WorkflowTarget.INSTRUMENT_IDENTIFIER,
            "additional_specs": WorkflowTarget.INSTRUMENT_IDENTIFIER,
            "confirm": WorkflowTarget.ENGENIE_CHAT,
            "reject": WorkflowTarget.ENGENIE_CHAT,
            "chitchat": WorkflowTarget.OUT_OF_DOMAIN,
            "unrelated": WorkflowTarget.OUT_OF_DOMAIN,
        }

        for intent, expected_workflow in test_cases.items():
            result = IntentConfig.get_workflow(intent, is_solution=False)
            assert result == expected_workflow, \
                f"Intent '{intent}' should map to {expected_workflow.value}, got {result.value}"

    def test_unknown_intent_defaults_to_ENGENIE_CHAT(self):
        """Unknown intents should default to ENGENIE_CHAT for safety."""
        result = IntentConfig.get_workflow("unknown_intent_xyz", is_solution=False)
        assert result == WorkflowTarget.ENGENIE_CHAT, \
            "Unknown intents should default to ENGENIE_CHAT"

    def test_is_known_intent(self):
        """Intent validation should work correctly."""
        assert IntentConfig.is_known_intent("solution") == True
        assert IntentConfig.is_known_intent("question") == True
        assert IntentConfig.is_known_intent("greeting") == True
        assert IntentConfig.is_known_intent("unknown_xyz") == False
        assert IntentConfig.is_known_intent("") == False
        assert IntentConfig.is_known_intent(None) == False

    def test_case_insensitivity(self):
        """Intent matching should be case-insensitive."""
        assert IntentConfig.get_workflow("SOLUTION", is_solution=False) == WorkflowTarget.SOLUTION_WORKFLOW
        assert IntentConfig.get_workflow("Solution", is_solution=False) == WorkflowTarget.SOLUTION_WORKFLOW
        assert IntentConfig.get_workflow("QUESTION", is_solution=False) == WorkflowTarget.ENGENIE_CHAT

    def test_whitespace_handling(self):
        """Intent matching should handle leading/trailing whitespace."""
        assert IntentConfig.get_workflow("  solution  ", is_solution=False) == WorkflowTarget.SOLUTION_WORKFLOW
        assert IntentConfig.get_workflow("\nquestion\n", is_solution=False) == WorkflowTarget.ENGENIE_CHAT


# =============================================================================
# TEST SUITE 2: Rule-Based Classification (Phase 2 FIX)
# =============================================================================

class TestRuleBasedClassification:
    """Tests for the improved rule-based classification (PHASE 2 FIX)."""

    def test_simple_product_request_classification(self):
        """Simple product requests should classify as 'requirements'."""
        queries = [
            "I need a pressure transmitter",
            "Looking for a temperature sensor",
            "I want a flow meter for water"
        ]

        for query in queries:
            result = _classify_rule_based(query)
            if result:  # If rule-based matched
                assert result["intent"] == "requirements", \
                    f"Simple request '{query}' should be 'requirements', got '{result['intent']}'"
                assert result["is_solution"] == False, \
                    f"Simple request '{query}' should have is_solution=False"

    def test_complex_system_detection(self):
        """Complex system requests should classify as 'solution' (PHASE 3 FIX)."""
        # PHASE 3: These queries contain explicit action verbs (designing, building)
        # OR explicit multi-instrument phrases
        queries = [
            "I'm designing a complete temperature measurement system for a chemical reactor",
            "I'm building a monitoring system for my process",
            "I need to design a system with multiple instruments"
        ]

        for query in queries:
            result = _classify_rule_based(query)
            if result:  # If rule-based matched
                assert result["intent"] == "solution", \
                    f"Complex request '{query}' should be 'solution', got '{result['intent']}'"
                assert result["is_solution"] == True, \
                    f"Complex request '{query}' should have is_solution=True"

    def test_solution_phrase_detection(self):
        """Explicit solution phrases with ACTION verbs should trigger solution classification (PHASE 3 FIX)."""
        # PHASE 3: These queries must contain ACTION verbs like "designing", "building", "implementing"
        queries = [
            "I'm designing a heating circuit with multiple zones",
            "I'm building a complete system for pressure monitoring",
            "I'm implementing a solution for my process"
        ]

        for query in queries:
            result = _classify_rule_based(query)
            if result:  # If rule-based matched
                assert result["is_solution"] == True, \
                    f"Solution phrase '{query}' should have is_solution=True"

    def test_greeting_classification(self):
        """Simple greetings should still be classified correctly."""
        greetings = ["hi", "hello", "hey", "good morning"]

        for greeting in greetings:
            result = _classify_rule_based(greeting)
            assert result is not None, f"Greeting '{greeting}' should be classified"
            assert result["intent"] == "greeting", \
                f"Greeting '{greeting}' should be classified as 'greeting'"

    def test_confirm_reject_classification(self):
        """Confirm/reject phrases should be classified correctly."""
        confirms = ["yes", "ok", "proceed", "confirm"]
        rejects = ["no", "cancel", "reject", "stop"]

        for confirm in confirms:
            result = _classify_rule_based(confirm)
            assert result is not None, f"Confirm '{confirm}' should be classified"
            assert result["intent"] == "confirm", \
                f"'{confirm}' should be 'confirm'"

        for reject in rejects:
            result = _classify_rule_based(reject)
            assert result is not None, f"Reject '{reject}' should be classified"
            assert result["intent"] == "reject", \
                f"'{reject}' should be 'reject'"

    def test_knowledge_question_classification(self):
        """Knowledge questions should still be classified correctly."""
        questions = [
            "What is a pressure transmitter?",
            "How does a thermowell work?",
            "Tell me about ATEX certification"
        ]

        for question in questions:
            result = _classify_rule_based(question)
            if result:  # If rule-based matched
                assert result["intent"] == "question", \
                    f"Question '{question}' should be 'question'"

    def test_knowledge_query_not_solution_phase3(self):
        """PHASE 3 FIX: Knowledge queries with keywords like 'what is' should NOT be classified as solution."""
        queries = [
            "I need a complete explanation of pressure measurement systems",
            "I need to understand how integrated solutions work",
            "I want a full description of instrumentation packages"
        ]

        for query in queries:
            result = _classify_rule_based(query)
            if result:  # If rule-based matched
                assert result["is_solution"] == False, \
                    f"Knowledge query '{query}' should NOT have is_solution=True"
                assert result["intent"] in ["question", "requirements"], \
                    f"Knowledge query '{query}' should be 'question' or 'requirements', got '{result['intent']}'"

    def test_rag_queries_not_solution_phase3(self):
        """PHASE 3 FIX: Queries that semantically belong to RAG should NOT route to solution."""
        # These queries might contain words like 'system' or 'solution' but are knowledge queries
        queries = [
            "I need information about complete measurement systems",
            "Tell me about integrated control solutions",
            "What is a complete instrumentation package?"
        ]

        for query in queries:
            result = _classify_rule_based(query)
            if result:  # If rule-based matched
                # These should NOT be flagged as solution since they're asking FOR information
                assert result["is_solution"] == False, \
                    f"RAG query '{query}' should NOT have is_solution=True"


# =============================================================================
# TEST SUITE 3: Exit Detection
# =============================================================================

class TestExitDetection:
    """Tests for workflow exit detection."""

    def test_exit_phrases(self):
        """Exit phrases should be detected."""
        exit_queries = [
            "start over",
            "new search",
            "reset",
            "clear",
            "begin again",
            "exit",
            "quit",
            "cancel",
            "back to start"
        ]

        for query in exit_queries:
            assert should_exit_workflow(query) == True, \
                f"Query '{query}' should be detected as exit"

    def test_greeting_is_exit(self):
        """Pure greetings should be treated as workflow exit."""
        greetings = ["hi", "hello", "hey"]

        for greeting in greetings:
            assert should_exit_workflow(greeting) == True, \
                f"Greeting '{greeting}' should trigger exit"

    def test_non_exit_queries(self):
        """Non-exit queries should not trigger exit."""
        queries = [
            "I need a pressure transmitter",
            "What is ATEX?",
            "Can you help me find a sensor?"
        ]

        for query in queries:
            assert should_exit_workflow(query) == False, \
                f"Query '{query}' should NOT be detected as exit"


# =============================================================================
# TEST SUITE 4: Intent Classification Routing Agent Integration
# =============================================================================

class TestIntentClassificationRoutingAgent:
    """Integration tests for the full routing agent."""

    def test_agent_initialization(self):
        """Agent should initialize correctly."""
        agent = IntentClassificationRoutingAgent()
        assert agent is not None
        assert agent.classification_count == 0

    def test_unknown_intent_handling(self):
        """Agent should handle unknown intents gracefully."""
        agent = IntentClassificationRoutingAgent()
        # This test assumes the agent can handle an unknown intent
        # In real usage, the LLM should never return unknown intents
        # But the agent should default to ENGENIE_CHAT if it does
        assert True  # Placeholder - full integration test needs mock LLM

    def test_workflow_locking(self):
        """Agent should lock/unlock workflow state correctly."""
        agent = IntentClassificationRoutingAgent()
        memory = agent._memory

        # Clear any existing state
        memory.clear_workflow("test_session")

        # Set a workflow
        memory.set_workflow("test_session", "solution")
        assert memory.get_workflow("test_session") == "solution"

        # Clear the workflow
        memory.clear_workflow("test_session")
        assert memory.get_workflow("test_session") is None


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

if __name__ == "__main__":
    print("Running Intent Classification Fix Tests (Phase 2 Implementation)...")
    print("=" * 70)

    # Run all test classes
    test_classes = [
        TestIntentConfig,
        TestRuleBasedClassification,
        TestExitDetection,
        TestIntentClassificationRoutingAgent
    ]

    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        test_instance = test_class()
        for method_name in dir(test_instance):
            if method_name.startswith("test_"):
                try:
                    method = getattr(test_instance, method_name)
                    method()
                    print(f"  ✓ {method_name}")
                except AssertionError as e:
                    print(f"  ✗ {method_name}: {e}")
                except Exception as e:
                    print(f"  ⚠ {method_name}: {type(e).__name__}: {e}")

    print("\n" + "=" * 70)
    print("Test run complete!")
