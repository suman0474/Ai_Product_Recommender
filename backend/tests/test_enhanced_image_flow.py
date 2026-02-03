"""
Test Script for Enhanced Product Image Flow with LLM Verification

This script tests the complete flow:
1. Cache check
2. Search engine fetching
3. LLM verification
4. Azure Blob caching
5. Cache retrieval

Usage:
    python backend/tests/test_enhanced_image_flow.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import logging
from agentic.vendor_image_utils import (
    fetch_vendor_product_images,
    fetch_images_for_vendor_matches
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def test_basic_image_fetch():
    """Test basic image fetching with LLM verification."""
    print("\n" + "="*80)
    print("TEST 1: Basic Image Fetch with LLM Verification")
    print("="*80)
    
    vendor = "Emerson"
    model_family = "Rosemount 3051"
    product_type = "Pressure Transmitter"
    
    logger.info(f"Fetching images for: {vendor} {model_family} ({product_type})")
    
    images = fetch_vendor_product_images(
        vendor_name=vendor,
        model_family=model_family,
        product_type=product_type
    )
    
    if images:
        logger.info(f"✓ Successfully fetched {len(images)} image(s)")
        for i, img in enumerate(images):
            print(f"\nImage {i+1}:")
            print(f"  URL: {img.get('url', '')[:100]}...")
            print(f"  Source: {img.get('source', 'unknown')}")
            print(f"  Cached: {img.get('cached', False)}")
            print(f"  LLM Verified: {img.get('llm_verified', False)}")
            if img.get('llm_confidence'):
                print(f"  LLM Confidence: {img.get('llm_confidence', 0.0):.2f}")
    else:
        logger.warning("✗ No images returned")
    
    return images


def test_cache_retrieval():
    """Test cache retrieval (should hit cache if test 1 ran)."""
    print("\n" + "="*80)
    print("TEST 2: Cache Retrieval (Should be instant)")
    print("="*80)
    
    vendor = "Emerson"
    model_family = "Rosemount 3051"
    product_type = "Pressure Transmitter"
    
    logger.info(f"Fetching images for: {vendor} {model_family} (should hit cache)")
    
    import time
    start = time.time()
    
    images = fetch_vendor_product_images(
        vendor_name=vendor,
        model_family=model_family,
        product_type=product_type
    )
    
    elapsed = time.time() - start
    
    if images:
        logger.info(f"✓ Cache retrieval completed in {elapsed:.2f}s")
        if elapsed < 1.0:
            logger.info("✓ FAST! Cache hit confirmed!")
        else:
            logger.warning("⚠ Slow response - may have missed cache")
        
        img = images[0]
        print(f"\nCached Image:")
        print(f"  Cached: {img.get('cached', False)}")
        print(f"  Source: {img.get('source', 'unknown')}")
    else:
        logger.warning("✗ No images returned from cache")
    
    return images


def test_vendor_match_enrichment():
    """Test enriching vendor matches with images."""
    print("\n" + "="*80)
    print("TEST 3: Vendor Match Enrichment")
    print("="*80)
    
    vendor = "Siemens"
    matches = [
        {
            'vendor_name': 'Siemens',
            'model_family': 'SITRANS P',
            'product_type': 'Pressure Transmitter',
            'product_name': 'SITRANS P DS III',
            'score': 95
        },
        {
            'vendor_name': 'Siemens',
            'model_family': 'SITRANS P',
            'product_type': 'Pressure Transmitter',
            'product_name': 'SITRANS P 320',
            'score': 88
        }
    ]
    
    logger.info(f"Enriching {len(matches)} matches for {vendor}")
    
    enriched = fetch_images_for_vendor_matches(
        vendor_name=vendor,
        matches=matches
    )
    
    if enriched:
        logger.info(f"✓ Successfully enriched {len(enriched)} matches")
        
        for i, match in enumerate(enriched):
            print(f"\nMatch {i+1}:")
            print(f"  Product: {match.get('product_name', 'Unknown')}")
            print(f"  Score: {match.get('score', 0)}")
            print(f"  Images: {len(match.get('product_images', []))}")
            print(f"  Image Source: {match.get('image_source', 'none')}")
            print(f"  LLM Verified: {match.get('llm_verified_image', False)}")
            if match.get('llm_confidence'):
                print(f"  LLM Confidence: {match.get('llm_confidence', 0.0):.2f}")
    else:
        logger.warning("✗ No enriched matches returned")
    
    return enriched


def test_different_vendor():
    """Test with a different vendor to verify it's not cached."""
    print("\n" + "="*80)
    print("TEST 4: Different Vendor (Cache Miss Expected)")
    print("="*80)
    
    vendor = "ABB"
    model_family = "266"
    product_type = "Pressure Transmitter"
    
    logger.info(f"Fetching images for: {vendor} {model_family}")
    
    images = fetch_vendor_product_images(
        vendor_name=vendor,
        model_family=model_family,
        product_type=product_type
    )
    
    if images:
        logger.info(f"✓ Successfully fetched {len(images)} image(s)")
        img = images[0]
        print(f"\nImage Details:")
        print(f"  LLM Verified: {img.get('llm_verified', False)}")
        if img.get('llm_confidence'):
            print(f"  LLM Confidence: {img.get('llm_confidence', 0.0):.2f}")
            print(f"  LLM Reason: {img.get('llm_reason', 'N/A')}")
    else:
        logger.warning("✗ No images returned")
    
    return images


def run_all_tests():
    """Run all test cases."""
    print("\n" + "#"*80)
    print("# ENHANCED IMAGE FLOW TEST SUITE")
    print("#"*80)
    
    results = {}
    
    try:
        # Test 1: Basic fetch with LLM verification
        results['test1'] = test_basic_image_fetch()
    except Exception as e:
        logger.error(f"Test 1 failed: {e}", exc_info=True)
        results['test1'] = None
    
    try:
        # Test 2: Cache retrieval
        results['test2'] = test_cache_retrieval()
    except Exception as e:
        logger.error(f"Test 2 failed: {e}", exc_info=True)
        results['test2'] = None
    
    try:
        # Test 3: Vendor match enrichment
        results['test3'] = test_vendor_match_enrichment()
    except Exception as e:
        logger.error(f"Test 3 failed: {e}", exc_info=True)
        results['test3'] = None
    
    try:
        # Test 4: Different vendor
        results['test4'] = test_different_vendor()
    except Exception as e:
        logger.error(f"Test 4 failed: {e}", exc_info=True)
        results['test4'] = None
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for r in results.values() if r is not None and len(r) > 0)
    total = len(results)
    
    print(f"\nTests Passed: {passed}/{total}")
    print("\nTest Details:")
    print(f"  Test 1 (Basic Fetch): {'✓ PASS' if results['test1'] else '✗ FAIL'}")
    print(f"  Test 2 (Cache Hit):   {'✓ PASS' if results['test2'] else '✗ FAIL'}")
    print(f"  Test 3 (Enrichment):  {'✓ PASS' if results['test3'] else '✗ FAIL'}")
    print(f"  Test 4 (New Vendor):  {'✓ PASS' if results['test4'] else '✗ FAIL'}")
    
    print("\n" + "#"*80)
    print(f"# TEST SUITE {'COMPLETED' if passed == total else 'COMPLETED WITH ERRORS'}")
    print("#"*80 + "\n")


if __name__ == "__main__":
    run_all_tests()
