"""
DEFINITIVE Google API Key Verification Script
Tests both GOOGLE_API_KEY and GOOGLE_API_KEY1 using correct model names
"""
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from datetime import datetime

# Load environment variables
load_dotenv()

output_file = "test_output_final.txt"

def log(message):
    """Log to both console and file"""
    try:
        print(message)
    except UnicodeEncodeError:
        print(message.encode('ascii', 'ignore').decode('ascii'))
    
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(message + "\n")

def test_api_key(key_name, model_name="gemini-2.5-flash"):
    """
    Test a single Google API key using LangChain
    
    Args:
        key_name: Environment variable name
        model_name: Gemini model to use
    
    Returns:
        dict: Test results
    """
    log(f"\n{'='*60}")
    log(f"Testing: {key_name}")
    log(f"Model: {model_name}")
    log(f"{'='*60}")
    
    result = {
        "key_name": key_name,
        "model": model_name,
        "key_present": False,
        "can_generate": False,
        "error": None,
        "status": "UNKNOWN"
    }
    
    api_key = os.getenv(key_name)
    
    if not api_key:
        log(f"  [FAIL] {key_name} not found in environment variables.")
        result["error"] = "Key not found"
        result["status"] = "NOT_FOUND"
        return result
    
    result["key_present"] = True
    log(f"  [OK] Key found: {api_key[:10]}...{api_key[-6:]}")
    
    try:
        log(f"  [..] Initializing {model_name}...")
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0,
            timeout=30
        )
        
        test_prompt = "Reply with exactly: WORKING"
        log(f"  [..] Sending test prompt...")
        
        response = llm.invoke(test_prompt)
        response_text = response.content.strip()
        
        log(f"  [OK] Response: '{response_text}'")
        
        if "WORKING" in response_text.upper():
            log(f"  [SUCCESS] {key_name} is FULLY FUNCTIONAL!")
            result["can_generate"] = True
            result["status"] = "WORKING"
        else:
            log(f"  [WARNING] Response received but unexpected content")
            result["can_generate"] = True
            result["status"] = "WORKING (unexpected response)"
            
    except Exception as e:
        error_str = str(e)
        log(f"  [FAIL] Error: {error_str[:200]}")
        result["error"] = error_str
        
        # Determine specific failure reason
        if "expired" in error_str.lower() or "API_KEY_INVALID" in error_str:
            result["status"] = "EXPIRED"
            log(f"  [REASON] API key has EXPIRED")
        elif "quota" in error_str.lower():
            result["status"] = "QUOTA_EXCEEDED"
            log(f"  [REASON] API quota exceeded")
        elif "not found" in error_str.lower():
            result["status"] = "MODEL_NOT_FOUND"
            log(f"  [REASON] Model not found (try different model)")
        elif "permission" in error_str.lower():
            result["status"] = "NO_PERMISSION"
            log(f"  [REASON] Insufficient permissions")
        else:
            result["status"] = "ERROR"
            log(f"  [REASON] Unknown error")
    
    return result

def main():
    """Main test function"""
    # Clear previous output
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"Google API Key Verification - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60 + "\n")
    
    log("\n" + "="*60)
    log("      GOOGLE API KEY VERIFICATION TEST")
    log("="*60)
    log(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Test Method: LangChain ChatGoogleGenerativeAI")
    
    # Test configuration - use same model for fair comparison
    test_model = "gemini-2.5-flash"
    
    results = []
    
    # Test GOOGLE_API_KEY
    results.append(test_api_key("GOOGLE_API_KEY", test_model))
    
    # Test GOOGLE_API_KEY1
    results.append(test_api_key("GOOGLE_API_KEY1", test_model))
    
    # Print Summary
    log("\n" + "="*60)
    log("                    FINAL SUMMARY")
    log("="*60 + "\n")
    
    for r in results:
        if r["status"] == "WORKING":
            icon = "[OK]"
            status_text = "WORKING"
        elif r["status"] == "EXPIRED":
            icon = "[XX]"
            status_text = "EXPIRED - Needs replacement"
        else:
            icon = "[??]"
            status_text = r["status"]
        
        log(f"{icon} {r['key_name']}: {status_text}")
    
    log("\n" + "-"*60)
    
    # Detailed recommendations
    working_keys = [r for r in results if r["status"] == "WORKING"]
    expired_keys = [r for r in results if r["status"] == "EXPIRED"]
    
    if len(working_keys) == 2:
        log("\n[EXCELLENT] Both API keys are working perfectly!")
    elif len(working_keys) == 1:
        log(f"\n[PARTIAL] Only {working_keys[0]['key_name']} is working.")
        log(f"[ACTION] Replace expired key: {expired_keys[0]['key_name']}")
    else:
        log("\n[CRITICAL] No working API keys!")
        log("[ACTION] Generate new keys from Google Cloud Console")
    
    log("\n" + "="*60)
    log("                    TEST COMPLETE")
    log("="*60 + "\n")
    
    return results

if __name__ == "__main__":
    main()
