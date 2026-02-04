import os
import sys
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
 
# Load environment variables
load_dotenv()
 
output_file = "test_output.txt"
 
def log(message):
    # Print to console avoiding encoding errors
    try:
        print(message)
    except UnicodeEncodeError:
        print(message.encode('ascii', 'ignore').decode('ascii'))
       
    # Write to file with explicit utf-8 encoding
    timestamp = ""
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(message + "\n")
 
def test_api_key(key_name, model_name="gemini-2.5-flash"):
    log(f"\n--- Testing Key: {key_name} ---")
       
    api_key = os.getenv(key_name)
    if not api_key:
        log(f"FAIL: {key_name} not found in environment variables.")
        return
 
    log(f"Initializing model {model_name}...")
    try:
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0
        )
       
        test_input = "Hello, can you hear me? Answer with 'YES'."
        log(f"Sending input: '{test_input}'")
       
        response = llm.invoke(test_input)
        response_content = response.content
       
        log(f"Response Received: '{response_content}'")
       
        if "YES" in response_content:
            log(f"SUCCESS: {key_name} is working correctly.")
        else:
            log(f"WARNING: Response received but unexpected content. Content: {response_content}")
           
    except Exception as e:
        log(f"FAIL: {key_name} failed. Error: {str(e)}")
 
if __name__ == "__main__":
    # Clear previous output
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("--- Google API Key Tester CLI Output ---\n")
       
    log("Starting API Key Test Session...")
   
    # Test text model
    test_api_key("GOOGLE_API_KEY", "gemini-2.5-flash")
   
    # Check for the secondary key seen in main.py
    if os.getenv("GOOGLE_API_KEY1"):
        test_api_key("GOOGLE_API_KEY1", "gemini-2.5-pro")
    else:
        log("\nGOOGLE_API_KEY1 not found in env (optional).")
