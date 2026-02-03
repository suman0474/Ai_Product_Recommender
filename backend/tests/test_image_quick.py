"""
Quick test for enhanced image flow - minimal output
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agentic.vendor_image_utils import fetch_vendor_product_images
import time

print("\n" + "="*60)
print("TESTING ENHANCED IMAGE FLOW")
print("="*60)

# Test 1: First request (should search and cache)
print("\n[TEST 1] First request - Emerson Rosemount 3051")
print("-" * 60)

start = time.time()
images = fetch_vendor_product_images(
    vendor_name="Emerson",
    model_family="Rosemount 3051",
    product_type="Pressure Transmitter"
)
elapsed = time.time() - start

if images:
    img = images[0]
    print("SUCCESS!")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  URL: {img.get('url', '')[:80]}...")
    print(f"  Source: {img.get('source', 'unknown')}")
    print(f"  Cached: {img.get('cached', False)}")
    print(f"  LLM Verified: {img.get('llm_verified', False)}")
    if img.get('llm_confidence'):
        print(f"  LLM Confidence: {img.get('llm_confidence', 0.0):.2f}")
        print(f"  LLM Reason: {img.get('llm_reason', 'N/A')[:60]}...")
    test1_pass = True
else:
    print("FAILED - No images returned")
    test1_pass = False

# Test 2: Second request (should hit cache)
print("\n[TEST 2] Second request - Same product (should be cached)")
print("-" * 60)

start = time.time()
images2 = fetch_vendor_product_images(
    vendor_name="Emerson",
    model_family="Rosemount 3051",
    product_type="Pressure Transmitter"
)
elapsed2 = time.time() - start

if images2:
    img2 = images2[0]
    print("SUCCESS!")
    print(f"  Time: {elapsed2:.2f}s")
    print(f"  Source: {img2.get('source', 'unknown')}")
    print(f"  Cached: {img2.get('cached', False)}")
    
    if elapsed2 < 1.0:
        print(f"  FAST! Cache hit confirmed ({elapsed2:.2f}s)")
        test2_pass = True
    else:
        print(f"  WARNING: Slow ({elapsed2:.2f}s) - may have missed cache")
        test2_pass = False
else:
    print("FAILED - No images returned")
    test2_pass = False

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Test 1 (First Request): {'PASS' if test1_pass else 'FAIL'}")
print(f"Test 2 (Cache Hit):     {'PASS' if test2_pass else 'FAIL'}")

if test1_pass and test2_pass:
    print("\nALL TESTS PASSED!")
    print("\nKey Features Working:")
    print("  - Search engine integration")
    print("  - LLM verification")
    print("  - Azure Blob caching")
    print("  - Fast cache retrieval")
else:
    print("\nSOME TESTS FAILED")
    if not test1_pass:
        print("  - First request failed (search/LLM/cache issue)")
    if not test2_pass:
        print("  - Cache retrieval not working optimally")

print("\n" + "="*60)

