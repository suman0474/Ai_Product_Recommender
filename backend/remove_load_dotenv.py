"""
Script to remove load_dotenv() calls from all Python files in backend.
Part of Phase 1: Override Fix Implementation
"""

import os
import re
from pathlib import Path

# Files to process (exclude initialization.py which should keep it)
EXCLUDE_FILES = ['initialization.py', 'remove_load_dotenv.py']

def remove_load_dotenv_from_file(file_path):
    """Remove load_dotenv() call and import from a Python file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content
        modified = False

        # Pattern 1: Remove "from dotenv import load_dotenv" line
        pattern1 = r'from dotenv import load_dotenv\s*\n'
        if re.search(pattern1, content):
            content = re.sub(pattern1, '', content)
            modified = True

        # Pattern 2: Remove standalone "load_dotenv()" calls
        pattern2 = r'^\s*load_dotenv\(\)\s*\n'
        if re.search(pattern2, content, re.MULTILINE):
            content = re.sub(pattern2, '', content, flags=re.MULTILINE)
            modified = True

        # Pattern 3: Remove "# Load environment variables" comment before load_dotenv()
        pattern3 = r'^\s*#\s*Load environment variables\s*\n\s*load_dotenv\(\)\s*\n'
        if re.search(pattern3, content, re.MULTILINE):
            content = re.sub(pattern3, '', content, flags=re.MULTILINE)
            modified = True

        # Pattern 4: Remove lines with conditional load_dotenv
        pattern4 = r'^\s*if.*load_dotenv\(\).*\n'
        if re.search(pattern4, content, re.MULTILINE):
            content = re.sub(pattern4, '', content, flags=re.MULTILINE)
            modified = True

        # Write back if modified
        if modified and content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True, "Modified"
        elif modified:
            return False, "No changes needed"
        else:
            return False, "No load_dotenv found"

    except Exception as e:
        return False, f"Error: {e}"


def main():
    """Process all Python files in backend directory"""
    backend_dir = Path(__file__).parent
    processed = 0
    modified = 0
    errors = []

    print("="*70)
    print("Removing load_dotenv() calls from Python files")
    print("="*70)

    for py_file in backend_dir.rglob("*.py"):
        # Skip excluded files
        if py_file.name in EXCLUDE_FILES:
            print(f"SKIP: {py_file.relative_to(backend_dir)} (excluded)")
            continue

        # Skip __pycache__ and virtual environments
        if '__pycache__' in str(py_file) or 'venv' in str(py_file) or '.venv' in str(py_file):
            continue

        processed += 1
        success, message = remove_load_dotenv_from_file(py_file)

        if success:
            modified += 1
            print(f"[OK] {py_file.relative_to(backend_dir)}: {message}")
        elif "Error" in message:
            errors.append((py_file, message))
            print(f"[ERR] {py_file.relative_to(backend_dir)}: {message}")
        else:
            print(f"  {py_file.relative_to(backend_dir)}: {message}")

    print("="*70)
    print(f"Summary:")
    print(f"  Files processed: {processed}")
    print(f"  Files modified: {modified}")
    print(f"  Errors: {len(errors)}")

    if errors:
        print("\nErrors encountered:")
        for file_path, error in errors:
            print(f"  - {file_path}: {error}")

    print("="*70)
    print("\n[DONE] PHASE 1 Step 3 Complete: load_dotenv() calls removed")
    print("  Environment now loaded ONCE in initialization.py")


if __name__ == "__main__":
    main()
