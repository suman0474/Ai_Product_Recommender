"""
Final Import Fix - Workflow Sibling Imports
Fixes incorrect relative imports in workflow files after restructuring
"""
import os
import re

# Mapping of incorrect imports to correct ones
SIBLING_IMPORT_FIXES = {
    # In solution/ and instrument/ - standards_rag is a sibling, not child
    "from .standards_rag.": "from ..standards_rag.",
    "from .strategy_rag.": "from ..strategy_rag.",
    "from .index_rag.": "from ..index_rag.",
    "from .engenie_chat.": "from ..engenie_chat.",
    
    # Standards detector moved to agents
    "from .standards_detector import": "from ...agents.standards_detector import",
    "from ..standards_detector import": "from ...agents.standards_detector import",
}


def fix_workflow_imports(filepath):
    """Fix sibling imports in workflow files."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        print(f"❌ Error reading {filepath}: {e}")
        return False
    
    original_content = content
    changes_made = []
    
    # Apply each fix
    for old_import, new_import in SIBLING_IMPORT_FIXES.items():
        if old_import in content:
            content = content.replace(old_import, new_import)
            changes_made.append(f"{old_import} → {new_import}")
    
    if content != original_content:
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Fixed {os.path.basename(filepath)}")
            for change in changes_made:
                print(f"   - {change}")
            return True
        except Exception as e:
            print(f"❌ Error writing {filepath}: {e}")
            return False
    
    return False


def main():
    print("=" * 80)
    print("FINAL FIX: Workflow Sibling Imports")
    print("=" * 80)
    print()
    
    # Target files that need fixing
    target_files = [
        r"d:\AI PR\AIPR\backend\agentic\workflows\solution\solution_workflow.py",
        r"d:\AI PR\AIPR\backend\agentic\workflows\instrument\identifier.py",
    ]
    
    fixed_count = 0
    
    for filepath in target_files:
        if os.path.exists(filepath):
            print(f"Processing: {os.path.basename(filepath)}")
            if fix_workflow_imports(filepath):
                fixed_count += 1
            print()
        else:
            print(f"⚠️  File not found: {filepath}\n")
    
    print("=" * 80)
    print(f"Summary: Fixed {fixed_count}/{len(target_files)} files")
    print("=" * 80)


if __name__ == "__main__":
    main()
