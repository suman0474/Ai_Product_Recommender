
import sys
import os
import json
import logging
from unittest.mock import MagicMock, patch

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Mock config to avoid loading real keys if not strictly needed
with patch('config.AgenticConfig') as MockConfig:
    MockConfig.PRO_MODEL = "gemini-pro"
    
    # Import tool to test
    from tools.instrument_tools import modify_instruments_tool

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_modify_instruments():
    print("Testing modify_instruments_tool...")
    
    # Mock data
    current_instruments = [{"category": "Pressure Transmitter", "product_name": "PT-101", "quantity": 1}]
    current_accessories = []
    modification_request = "Add a thermowell for the transmitter."
    
    # Mock Azure Blob Connection
    mock_docs_collection = MagicMock()
    mock_docs_collection.find.return_value = [
        {"filename": "Project_Specs.pdf", "document_type": "Specification", "description": "General piping specs", "session_id": "test_session"}
    ]
    mock_blob_conn = {'collections': {'documents': mock_docs_collection}}
    
    # Mock LLM Chain
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = {
        "instruments": current_instruments,
        "accessories": [{"category": "Thermowell", "accessory_name": "TW-101", "quantity": 1, "specifications": {"material": "316SS"}}],
        "changes_made": ["Added Thermowell TW-101"],
        "summary": "Added thermowell."
    }
    
    # Mock create_llm_with_fallback to return a mock that produces the chain
    # Actually modify_instruments_tool constructs the chain internally: chain = prompt | llm | parser
    # We need to mock the chain execution or the LLM.
    # It's easier to mock the `chain.invoke` but since chain is local variable, we must patch `create_llm_with_fallback` 
    # and the subsequent chain construction is hard to mock essentially without patching the whole chain pipeline.
    # Alternatively, we can patch `ChatGoogleGenerativeAI` or just `instrument_tools.create_llm_with_fallback`
    
    with patch('tools.instrument_tools.get_azure_blob_connection', return_value=mock_blob_conn) as mock_get_conn, \
         patch('tools.instrument_tools.create_llm_with_fallback') as mock_create_llm, \
         patch('tools.instrument_tools.ChatPromptTemplate') as mock_prompt, \
         patch('tools.instrument_tools.JsonOutputParser') as mock_parser:

        # Setup Mock Chain
        mock_llm_instance = MagicMock()
        mock_create_llm.return_value = mock_llm_instance
        
        # We need the chain.invoke to return our result
        # chain = prompt | llm | parser
        # In langchain this creates a RunnableSequence. 
        # We can just mock the whole chain construction line? No, it's inside the function.
        # We can make the mock_parser (last element) return a format that implies the chain result?
        # A simpler way is to `patch('tools.instrument_tools.chain.invoke')` but `chain` is local.
        
        # Let's rely on `mock_chain` object returned by `prompt | llm | parser`
        # created by `__or__` calls.
        
        # prompt | llm -> intermediate
        # intermediate | parser -> chain
        mock_intermediate = MagicMock()
        mock_final_chain = MagicMock()
        
        mock_prompt.from_template.return_value = MagicMock()
        # Mocking the pipe operators which are `__or__`
        mock_prompt.from_template.return_value.__or__.return_value = mock_intermediate
        mock_intermediate.__or__.return_value = mock_final_chain
        
        mock_final_chain.invoke.return_value = {
            "instruments": current_instruments,
            "accessories": [{"category": "Thermowell", "accessory_name": "TW-101", "quantity": 1, "specifications": {"material": "316SS"}}],
            "changes_made": ["Added Thermowell TW-101"],
            "summary": "Added thermowell."
        }
        
        # Mock message generation chain separately? 
        # The tool creates a second chain for message.
        # We can let that fail or mock it too. The tool handles failure gracefully.
        
        result = modify_instruments_tool.func(
            modification_request=modification_request,
            current_instruments=current_instruments,
            current_accessories=current_accessories,
            search_session_id="test_session"
        )
        
        # Verification
        print(f"Result Success: {result.get('success')}")
        print(f"Accessories: {len(result.get('accessories', []))}")
        print(f"Message: {result.get('message')}")
        
        # Check if user docs were fetched
        mock_docs_collection.find.assert_called_once()
        print("User documents fetched: YES")
        
        # Check if LLM was invoked with user docs
        # We can check the arguments to invoke
        # Since there are two chains (modification and message), check all calls
        found_context = False
        for call in mock_final_chain.invoke.call_args_list:
            args = call[0][0]
            if "user_documents_context" in args and "Project_Specs.pdf" in args["user_documents_context"]:
                 found_context = True
                 break
        
        if found_context:
             print("User documents passed to LLM: YES")
        else:
             print("User documents passed to LLM: NO")
             print(f"Call args list: {[c[0][0].keys() for c in mock_final_chain.invoke.call_args_list]}")

if __name__ == "__main__":
    test_modify_instruments()
