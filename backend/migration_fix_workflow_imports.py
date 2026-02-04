"""
Comprehensive import fix for restructured files
Fixes imports in workflow files that reference old module locations
"""
import os
import re

WORKFLOW_IMPORT_FIXES = {
    # Checkpointing moved to infrastructure
    "from .checkpointing import": "from agentic.infrastructure.state.checkpointing.local import",
    "from ..checkpointing import": "from agentic.infrastructure.state.checkpointing.local import",
    
    # LLM manager moved to utils
    "from .llm_manager import": "from agentic.utils.llm_manager import",
    "from ..llm_manager import": "from agentic.utils.llm_manager import",
    
    # Orchestrator utils
    "from .orchestrator_utils import": "from agentic.utils.orchestrator_utils import",
    "from ..orchestrator_utils import": "from agentic.utils.orchestrator_utils import",
    
    # State management
    "from .workflow_state_manager import": "from agentic.infrastructure.state.workflow_state import",
    "from ..workflow_state_manager import": "from agentic.infrastructure.state.workflow_state import",
    
    # Deep agent
    "from .deep_agent import": "from agentic.deep_agent import",
    "from ..deep_agent import": "from agentic.deep_agent import",
    
    # RAG components
    "from .rag_components import": "from agentic.rag.components import",
    "from ..rag_components import": "from agentic.rag.components import",
    
    # Internal API
    "from .internal_api import": "from agentic.infrastructure.api.internal import",
    "from ..internal_api import": "from agentic.infrastructure.api.internal import",
}


def fix_file_imports(filepath):
    """Fix imports in a single file."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return False
    
    original_content = content
    changes = 0
    
    # Apply fixes
    for old_import, new_import in WORKFLOW_IMPORT_FIXES.items():
        if old_import in content:
            content = content.replace(old_import, new_import)
            changes += 1
            print(f"  - Fixed: {old_import} → {new_import}")
    
    if content != original_content:
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✓ Fixed {filepath} ({changes} changes)")
            return True
        except Exception as e:
            print(f"Error writing {filepath}: {e}")
            return False
    
    return False


def main():
    print("=" * 70)
    print("Comprehensive Import Fix for Workflows")
    print("=" * 70)
    
    # Directories to process
    directories = [
        r"d:\AI PR\AIPR\backend\agentic\workflows",
        r"d:\AI PR\AIPR\backend\agentic\infrastructure\api",
    ]
    
    total_files = 0
    fixed_files = 0
    
    for directory in directories:
        print(f"\nProcessing: {directory}")
        print("-" * 70)
        
        for root, dirs, files in os.walk(directory):
            # Skip __pycache__
            dirs[:] = [d for d in dirs if d != '__pycache__']
            
            for filename in files:
                if filename.endswith('.py'):
                    filepath = os.path.join(root, filename)
                    total_files += 1
                    if fix_file_imports(filepath):
                        fixed_files += 1
    
    print("=" * 70)
    print(f"Summary: Fixed {fixed_files}/{total_files} files")
    print("=" * 70)


if __name__ == "__main__":
    main()
