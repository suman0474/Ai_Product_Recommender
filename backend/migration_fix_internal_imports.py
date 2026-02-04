"""
Import Update Script for Moved Files (Internal Imports)
Updates internal imports within files that were moved during restructuring
"""
import os
import re

# Mapping of old relative import paths to new paths for files in infrastructure/api/
INFRA_API_MAPPINGS = {
    # Auth & Utils
    "from .auth_decorators import": "from ..utils.auth_decorators import",
    "from .api_utils import": "from .utils import",
    
    # Workflows
    "from .workflow import": "from ...workflows.base.workflow import",
    "from .solution_workflow import": "from ...workflows.solution.solution_workflow import",
    "from .instrument_identifier_workflow import": "from ...workflows.instrument.identifier import",
    "from .potential_product_index import": "from ...rag.product_index import",
    "from .grounded_chat_workflow import": "from ...workflows.chat.grounded_chat import",
    "from .schema_workflow import": "from ...workflows.schema.schema_workflow import",
    
    # State Management
    "from .workflow_state_manager import": "from ..state.workflow_state import",
    "from .session_orchestrator import": "from ..state.session.orchestrator import",
    "from .checkpointing import": "from ..state.checkpointing.local import",
    "from .azure_checkpointing import": "from ..state.checkpointing.azure import",
    "from .cosmos_session_manager import": "from ..state.session.cosmos_manager import",
    "from .session_cleanup_manager import": "from ..state.session.cleanup import",
    "from .context_managers import": "from ..state.context.managers import",
    "from .instance_manager import": "from ..state.execution.instance_manager import",
    "from .global_executor_manager import": "from ..state.execution.executor_manager import",
    "from .thread_manager import": "from ..state.execution.thread_manager import",
    "from .state_storage import": "from ..state.storage import",
    
    # Caching
    "from .caching.bounded_cache_manager import": "from ..caching.bounded_cache import",
    "from .rag_cache import": "from ..caching.rag_cache import",
    "from .embedding_cache_manager import": "from ..caching.embedding_cache import",
    "from .llm_response_cache_manager import": "from ..caching.llm_response_cache import",
    
    # Infrastructure Utils
    "from .fast_fail import": "from ..utils.fast_fail import",
    "from .circuit_breaker import": "from ..utils.circuit_breaker import",
    "from .rate_limiter import": "from ..utils.rate_limiter import",
    "from .rate_limits import": "from ..utils.rate_limits import",
    "from .compression_manager import": "from ..utils.compression import",
    "from .metrics_generator import": "from ..utils.metrics import",
    
    # Agents
    "from .router_agent import": "from ...agents.routing.router_agent import",
    "from .intent_classification_routing_agent import": "from ...agents.routing.intent_classifier import",
    "from .semantic_intent_classifier import": "from ...agents.routing.semantic_classifier import",
    "from .shared_agents import": "from ...agents.shared.shared_agents import",
    "from .standards_detector import": "from ...agents.standards_detector import",
    
    # RAG
    "from .rag_components import": "from ...rag.components import",
    "from .rag_logger import": "from ...rag.logger import",
    "from .vector_store import": "from ...rag.vector_store import",
    
    # Utils
    "from .debug_utils import": "from ...utils.debug import",
    "from .exceptions import": "from ...utils.exceptions import",
    "from .input_sanitizer import": "from ...utils.input_sanitizer import",
    "from .llm_manager import": "from ...utils.llm_manager import",
    "from .streaming_utils import": "from ...utils.streaming import",
    "from .vendor_image_utils import": "from ...utils.vendor_images import",
    "from .orchestrator_utils import": "from ...utils.orchestrator_utils import",
    
    # Internal API (same directory)
    "from .internal_api import": "from .internal import",
    "from .session_api import": "from .session import",
    "from .api_streaming import": "from .streaming import",
    
    # Models
    "from .models import": "from ...models import",
    
    # Core
    "from .base_agent import": "from ...core.agents.base_agent import",
    "from .base_memory import": "from ...core.agents.base_memory import",
    "from .base_state import": "from ...core.state.base_state import",
    "from .base_cache import": "from ...core.state.base_cache import",
    "from .common_nodes import": "from ...core.common_nodes import",
    
    # Engenie Chat -> Workflows
    "from .engenie_chat.": "from ...workflows.engenie_chat.",
    "from .standards_rag.": "from ...workflows.standards_rag.",
    "from .strategy_rag.": "from ...workflows.strategy_rag.",
    "from .index_rag.": "from ...workflows.index_rag.",
    
    # Deep Agent
    "from .deep_agent.": "from ...deep_agent.",
}


def update_imports_in_file(filepath, mappings):
    """Update imports in a single file."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return False, 0
    
    original_content = content
    changes = 0
    
    # Sort mappings by length (longer first) to avoid partial replacements
    sorted_mappings = sorted(mappings.items(), key=lambda x: -len(x[0]))
    
    for old_import, new_import in sorted_mappings:
        if old_import in content:
            content = content.replace(old_import, new_import)
            changes += 1
    
    if content != original_content:
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True, changes
        except Exception as e:
            print(f"Error writing {filepath}: {e}")
            return False, 0
    
    return False, 0


def process_directory(directory, mappings):
    """Process all Python files in a directory."""
    total_files = 0
    modified_files = 0
    total_changes = 0
    
    for root, dirs, files in os.walk(directory):
        # Skip pycache
        dirs[:] = [d for d in dirs if d != '__pycache__']
        
        for filename in files:
            if filename.endswith('.py'):
                filepath = os.path.join(root, filename)
                total_files += 1
                modified, changes = update_imports_in_file(filepath, mappings)
                if modified:
                    modified_files += 1
                    total_changes += changes
                    print(f"Modified: {filepath} ({changes} changes)")
    
    return total_files, modified_files, total_changes


def main():
    print("=" * 60)
    print("Internal Import Update Script for Moved Files")
    print("=" * 60)
    
    # Update infrastructure/api/ files
    api_dir = r"d:\AI PR\AIPR\backend\agentic\infrastructure\api"
    print(f"\nProcessing: {api_dir}")
    print("-" * 60)
    
    total_files, modified_files, total_changes = process_directory(api_dir, INFRA_API_MAPPINGS)
    
    print("-" * 60)
    print(f"\nSummary:")
    print(f"  Total Python files scanned: {total_files}")
    print(f"  Files modified: {modified_files}")
    print(f"  Total import changes: {total_changes}")
    print("=" * 60)


if __name__ == "__main__":
    main()
