"""
Automated Import Update Script for Backend Restructuring

This script updates all import statements to reflect the new directory structure.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

# Import mapping: old_import -> new_import
IMPORT_MAPPINGS = {
    # Core modules
    r'from core.models import': 'from core.models import',
    r'import core.models as models': 'import core.models as models',
    r'from core.auth.auth_models import': 'from core.auth.auth_models import',
    r'from core.auth.auth_utils import': 'from core.auth.auth_utils import',
    r'from core.extraction_engine import': 'from core.extraction_engine import',
    r'from core.loading import': 'from core.loading import',
    r'from core.chaining import': 'from core.chaining import',
    
    # Services - Azure
    r'from services.azure.blob_utils import': 'from services.azure.blob_utils import',
    r'from services.azure.image_utils import': 'from services.azure.image_utils import',
    r'from services.azure.cosmos_manager import': 'from services.azure.cosmos_manager import',
    
    # Services - Products
    r'from services.products.catalog_builder import': 'from services.products.catalog_builder import',
    r'from services.products.standardization import': 'from services.products.standardization import',
    
    # Services - LLM
    r'from services.llm.fallback import': 'from services.llm.fallback import',
    r'from services.llm.standardization import': 'from services.llm.standardization import',
}


def update_file_imports(file_path: Path) -> Tuple[bool, int]:
    """
    Update imports in a single Python file.
    
    Returns:
        (changed, num_replacements) - Whether file changed and number of replacements
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")
        return False, 0
    
    original_content = content
    total_replacements = 0
    
    # Apply each import mapping
    for old_pattern, new_import in IMPORT_MAPPINGS.items():
        # Count replacements for this pattern
        matches = len(re.findall(old_pattern, content))
        if matches > 0:
            content = re.sub(old_pattern, new_import, content)
            total_replacements += matches
            print(f"  ✓ {file_path.name}: {matches}x {old_pattern} -> {new_import}")
    
    # Only write if content changed
    if content != original_content:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True, total_replacements
        except Exception as e:
            print(f"❌ Error writing {file_path}: {e}")
            return False, 0
    
    return False, 0


def update_all_imports(root_dir: str = ".") -> Dict[str, int]:
    """
    Update imports in all Python files in the backend.
    
    Returns:
        Statistics dictionary
    """
    root_path = Path(root_dir)
    stats = {
        "files_scanned": 0,
        "files_updated": 0,
        "total_replacements": 0,
        "files_with_errors": 0
    }
    
    # Find all Python files
    python_files = list(root_path.rglob("*.py"))
    
    # Exclude virtual environment and pycache
    python_files = [
        f for f in python_files 
        if not any(part in f.parts for part in ['venv', '__pycache__', '.venv', 'env'])
    ]
    
    print(f"\n{'='*70}")
    print(f"  IMPORT UPDATE SCRIPT - Backend Restructuring")
    print(f"{'='*70}\n")
    print(f"Found {len(python_files)} Python files to process...\n")
    
    for py_file in python_files:
        stats["files_scanned"] += 1
        changed, replacements = update_file_imports(py_file)
        
        if changed:
            stats["files_updated"] += 1
            stats["total_replacements"] += replacements
    
    return stats


if __name__ == "__main__":
    import sys
    
    # Get backend directory
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    
    print(f"Working directory: {backend_dir}\n")
    
    # Run the update
    stats = update_all_imports(backend_dir)
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"  UPDATE SUMMARY")
    print(f"{'='*70}")
    print(f"  Files scanned:        {stats['files_scanned']}")
    print(f"  Files updated:        {stats['files_updated']}")
    print(f"  Total replacements:   {stats['total_replacements']}")
    print(f"  Files with errors:    {stats['files_with_errors']}")
    print(f"{'='*70}\n")
    
    if stats['files_updated'] > 0:
        print("✅ Import updates completed successfully!")
        print("⚠️  Please review the changes with: git diff")
        sys.exit(0)
    else:
        print("ℹ️  No imports needed updating.")
        sys.exit(0)
