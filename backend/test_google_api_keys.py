"""
Test script to verify Google API keys functionality
Tests both GOOGLE_API_KEY and GOOGLE_API_KEY1
"""
import os
import sys
from dotenv import load_dotenv
import google.generativeai as genai
from datetime import datetime

# Load environment variables
load_dotenv()

def test_api_key(key_name, api_key):
    """
    Test a single Google API key
    
    Args:
        key_name: Name of the key (for logging)
        api_key: The actual API key value
    
    Returns:
        dict: Test results
    """
    print(f"\n{'='*60}")
    print(f"Testing {key_name}")
    print(f"{'='*60}")
    
    result = {
        "key_name": key_name,
        "key_present": bool(api_key),
        "key_format_valid": False,
        "api_accessible": False,
        "can_generate_content": False,
        "model_info": None,
        "error": None,
        "timestamp": datetime.now().isoformat()
    }
    
    if not api_key:
        print(f"❌ {key_name} is not set or empty")
        result["error"] = "API key not set"
        return result
    
    print(f"✓ {key_name} is present")
    print(f"   Key format: {api_key[:10]}...{api_key[-10:] if len(api_key) > 20 else ''}")
    
    # Check key format (Google API keys typically start with "AIza")
    if api_key.startswith("AIza"):
        print(f"✓ Key format appears valid (starts with 'AIza')")
        result["key_format_valid"] = True
    else:
        print(f"⚠ Warning: Key format unusual (doesn't start with 'AIza')")
    
    try:
        # Configure the API with the key
        genai.configure(api_key=api_key)
        print(f"✓ API configured successfully")
        result["api_accessible"] = True
        
        # List available models
        print(f"\nAttempting to list available models...")
        models = list(genai.list_models())
        if models:
            print(f"✓ Successfully retrieved {len(models)} models")
            print(f"   Available models:")
            for model in models[:5]:  # Show first 5 models
                print(f"   - {model.name}")
            result["model_info"] = {
                "total_models": len(models),
                "sample_models": [model.name for model in models[:5]]
            }
        
        # Test content generation with a simple prompt
        print(f"\nAttempting to generate content...")
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content("Say 'API test successful' in a creative way.")
        
        if response and response.text:
            print(f"✓ Content generation successful!")
            print(f"   Response: {response.text[:100]}...")
            result["can_generate_content"] = True
            result["sample_response"] = response.text[:200]
        else:
            print(f"⚠ Response received but no text generated")
            
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Error: {error_msg}")
        result["error"] = error_msg
        
        # Provide more specific error analysis
        if "API key not valid" in error_msg or "invalid" in error_msg.lower():
            print(f"   → This key appears to be INVALID or EXPIRED")
        elif "quota" in error_msg.lower():
            print(f"   → This key has exceeded its quota")
        elif "permission" in error_msg.lower():
            print(f"   → This key lacks necessary permissions")
        elif "network" in error_msg.lower() or "connection" in error_msg.lower():
            print(f"   → Network connectivity issue")
        else:
            print(f"   → Unknown error type")
    
    return result

def main():
    """Main test function"""
    print("\n" + "="*60)
    print("Google API Key Verification Test")
    print("="*60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Get API keys from environment
    api_key1 = os.getenv("GOOGLE_API_KEY")
    api_key2 = os.getenv("GOOGLE_API_KEY1")
    
    # Test both keys
    results = []
    results.append(test_api_key("GOOGLE_API_KEY", api_key1))
    results.append(test_api_key("GOOGLE_API_KEY1", api_key2))
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}\n")
    
    for result in results:
        status = "✓ WORKING" if result["can_generate_content"] else "❌ NOT WORKING"
        print(f"{result['key_name']}: {status}")
        
        if result["key_present"]:
            print(f"  - Key Present: ✓")
            print(f"  - Format Valid: {'✓' if result['key_format_valid'] else '❌'}")
            print(f"  - API Accessible: {'✓' if result['api_accessible'] else '❌'}")
            print(f"  - Can Generate Content: {'✓' if result['can_generate_content'] else '❌'}")
            if result["error"]:
                print(f"  - Error: {result['error']}")
        else:
            print(f"  - Key Present: ❌")
        print()
    
    # Recommendations
    print(f"{'='*60}")
    print("RECOMMENDATIONS")
    print(f"{'='*60}\n")
    
    working_keys = [r for r in results if r["can_generate_content"]]
    
    if len(working_keys) == 2:
        print("✓ Both API keys are functioning properly!")
        print("  You can use either key for your application.")
    elif len(working_keys) == 1:
        print(f"⚠ Only {working_keys[0]['key_name']} is working.")
        print(f"  Consider fixing or replacing the non-working key.")
    else:
        print("❌ Neither API key is working!")
        print("  Actions needed:")
        print("  1. Verify keys are correct in .env file")
        print("  2. Check if keys are enabled in Google Cloud Console")
        print("  3. Verify billing is enabled for the project")
        print("  4. Check if Generative AI API is enabled")
        print("  5. Consider generating new API keys")
    
    print()
    return results

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        sys.exit(1)
