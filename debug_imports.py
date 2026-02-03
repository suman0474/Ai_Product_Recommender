import sys
import os

print(f"Executable: {sys.executable}")
print(f"Path: {sys.path}")

try:
    import flask
    print(f"Flask version: {flask.__version__}")
except ImportError as e:
    print(f"Flask import failed: {e}")

# Add backend to path for internal modules
backend_path = os.path.join(os.getcwd(), 'backend')
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

try:
    from agentic.deep_agent import parallel_specs_enrichment
    print("✅ parallel_specs_enrichment imported successfully")
except ImportError as e:
    print(f"❌ parallel_specs_enrichment import failed: {e}")

try:
    # Try importing the module that failed for the user
    from agentic.deep_agent import parallel_processing
    print("✅ parallel_processing imported successfully")
except ImportError as e:
    print(f"❌ parallel_processing import failed: {e}")
