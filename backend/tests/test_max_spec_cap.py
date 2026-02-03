
import unittest
import sys
import os
import logging

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), 'backend'))

from agentic.deep_agent.optimized_parallel_agent import deduplicate_and_merge_specifications, MAX_SPECS_PER_ITEM

class TestMaxSpecCap(unittest.TestCase):
    def test_max_spec_cap_enforcement(self):
        print(f"\nTesting MAX_SPECS_PER_ITEM cap: {MAX_SPECS_PER_ITEM}")
        
        # 1. Create 50 user specs
        user_specs = {f"user_spec_{i}": f"value_{i}" for i in range(50)}
        
        # 2. Create 50 standards specs
        standards_specs = {f"std_spec_{i}": {"value": f"val_{i}", "confidence": 0.9} for i in range(50)}
        
        # 3. Create 50 LLM specs (should only accept a few before hitting 100)
        llm_specs = {f"llm_spec_{i}": {"value": f"val_{i}", "confidence": 0.7} for i in range(50)}
        
        # Merge
        result = deduplicate_and_merge_specifications(user_specs, llm_specs, standards_specs)
        
        print(f"Total merged specs: {len(result)}")
        
        # Assertions
        self.assertLessEqual(len(result), MAX_SPECS_PER_ITEM, "Exceeded max spec cap!")
        self.assertEqual(len(result), 100, "Should have reached exactly 100")
        
        # Verify content
        # Should have all 50 user specs
        for i in range(50):
            self.assertIn(f"user_spec_{i}", result)
            
        # Should have all 50 standards specs (50+50 = 100)
        for i in range(50):
            self.assertIn(f"std_spec_{i}", result)
            
        # Should have 0 LLM specs (since 100 reached)
        llm_count = sum(1 for k in result if k.startswith("llm_spec_"))
        self.assertEqual(llm_count, 0, "Should not have accepted LLM specs after cap reached")

if __name__ == "__main__":
    unittest.main()
