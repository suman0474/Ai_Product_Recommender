"""
Import Update Script for Agentic Restructuring
Run this script to update all imports after folder restructuring
"""
import os
import re

# Mapping of old import paths to new import paths
IMPORT_MAPPINGS = {
    # Core - Base Classes
    "from agentic.core.agents.base_agent": "from agentic.core.agents.base_agent",
    "from agentic.core.agents.base_memory": "from agentic.core.agents.base_memory",
    "from agentic.core.state.base_state": "from agentic.core.state.base_state",
    "from agentic.core.state.base_cache": "from agentic.core.state.base_cache",
    "from agentic.core.common_nodes": "from agentic.core.common_nodes",
    
    # Infrastructure - API
    "from agentic.infrastructure.api.main_api import": "from agentic.infrastructure.api.main_api import",
    "from agentic.infrastructure.api.streaming": "from agentic.infrastructure.api.streaming",
    "from agentic.infrastructure.api.utils": "from agentic.infrastructure.api.utils",
    "from agentic.infrastructure.api.internal": "from agentic.infrastructure.api.internal",
    "from agentic.infrastructure.api.session": "from agentic.infrastructure.api.session",
    "from agentic.infrastructure.api.monitoring": "from agentic.infrastructure.api.monitoring",
    
    # Infrastructure - State - Checkpointing
    "from agentic.infrastructure.state.checkpointing.azure": "from agentic.infrastructure.state.checkpointing.azure",
    "from agentic.infrastructure.state.checkpointing.local": "from agentic.infrastructure.state.checkpointing.local",
    
    # Infrastructure - State - Session
    "from agentic.infrastructure.state.session.cosmos_manager": "from agentic.infrastructure.state.session.cosmos_manager",
    "from agentic.infrastructure.state.session.orchestrator": "from agentic.infrastructure.state.session.orchestrator",
    "from agentic.infrastructure.state.session.cleanup": "from agentic.infrastructure.state.session.cleanup",
    
    # Infrastructure - State - Context
    "from agentic.infrastructure.state.context.managers": "from agentic.infrastructure.state.context.managers",
    "from agentic.infrastructure.state.context.chromadb": "from agentic.infrastructure.state.context.chromadb",
    "from agentic.infrastructure.state.context.langchain": "from agentic.infrastructure.state.context.langchain",
    "from agentic.infrastructure.state.context.lock_monitor": "from agentic.infrastructure.state.context.lock_monitor",
    
    # Infrastructure - State - Execution
    "from agentic.infrastructure.state.execution.instance_manager": "from agentic.infrastructure.state.execution.instance_manager",
    "from agentic.infrastructure.state.execution.thread_manager": "from agentic.infrastructure.state.execution.thread_manager",
    "from agentic.infrastructure.state.execution.executor_manager": "from agentic.infrastructure.state.execution.executor_manager",
    "from agentic.infrastructure.state.storage": "from agentic.infrastructure.state.storage",
    "from agentic.infrastructure.state.workflow_state": "from agentic.infrastructure.state.workflow_state",
    
    # Infrastructure - Caching
    "from agentic.infrastructure.caching.bounded_cache": "from agentic.infrastructure.caching.bounded_cache",
    "from agentic.infrastructure.caching.validation_cache": "from agentic.infrastructure.caching.validation_cache",
    "from agentic.infrastructure.caching.rag_cache": "from agentic.infrastructure.caching.rag_cache",
    "from agentic.infrastructure.caching.embedding_cache": "from agentic.infrastructure.caching.embedding_cache",
    "from agentic.infrastructure.caching.llm_response_cache": "from agentic.infrastructure.caching.llm_response_cache",
    "from agentic.infrastructure.caching.embedding_processor": "from agentic.infrastructure.caching.embedding_processor",
    
    # Infrastructure - Utils
    "from agentic.infrastructure.utils.auth_decorators": "from agentic.infrastructure.utils.auth_decorators",
    "from agentic.infrastructure.utils.compression": "from agentic.infrastructure.utils.compression",
    "from agentic.infrastructure.utils.circuit_breaker": "from agentic.infrastructure.utils.circuit_breaker",
    "from agentic.infrastructure.utils.rate_limiter": "from agentic.infrastructure.utils.rate_limiter",
    "from agentic.infrastructure.utils.rate_limits": "from agentic.infrastructure.utils.rate_limits",
    "from agentic.infrastructure.utils.fast_fail": "from agentic.infrastructure.utils.fast_fail",
    "from agentic.infrastructure.utils.metrics": "from agentic.infrastructure.utils.metrics",
    
    # Agents
    "from agentic.agents.routing.router_agent": "from agentic.agents.routing.router_agent",
    "from agentic.agents.routing.intent_classifier": "from agentic.agents.routing.intent_classifier",
    "from agentic.agents.routing.semantic_classifier": "from agentic.agents.routing.semantic_classifier",
    "from agentic.agents.shared.shared_agents": "from agentic.agents.shared.shared_agents",
    "from agentic.agents.standards_detector": "from agentic.agents.standards_detector",
    
    # Utils
    "from agentic.utils.debug": "from agentic.utils.debug",
    "from agentic.utils.exceptions": "from agentic.utils.exceptions",
    "from agentic.utils.input_sanitizer": "from agentic.utils.input_sanitizer",
    "from agentic.utils.llm_manager": "from agentic.utils.llm_manager",
    "from agentic.utils.streaming": "from agentic.utils.streaming",
    "from agentic.utils.vendor_images": "from agentic.utils.vendor_images",
    "from agentic.utils.orchestrator_utils": "from agentic.utils.orchestrator_utils",
    
    # RAG
    "from agentic.rag.components": "from agentic.rag.components",
    "from agentic.rag.logger": "from agentic.rag.logger",
    "from agentic.rag.vector_store": "from agentic.rag.vector_store",
    "from agentic.rag.product_index": "from agentic.rag.product_index",
    "from agentic.rag.enrichment": "from agentic.rag.enrichment",
    "from agentic.rag.strategy_enrichment": "from agentic.rag.strategy_enrichment",
    
    # Workflows - Base
    "from agentic.workflows.base.workflow import": "from agentic.workflows.base.workflow import",
    "from agentic.workflows.base.orchestrator": "from agentic.workflows.base.orchestrator",
    "from agentic.workflows.base.registry": "from agentic.workflows.base.registry",
    "from agentic.workflows.base.streaming": "from agentic.workflows.base.streaming",
    
    # Workflows - Individual
    "from agentic.workflows.chat.grounded_chat": "from agentic.workflows.chat.grounded_chat",
    "from agentic.workflows.instrument.identifier": "from agentic.workflows.instrument.identifier",
    "from agentic.workflows.schema.schema_workflow": "from agentic.workflows.schema.schema_workflow",
    "from agentic.workflows.solution.solution_workflow": "from agentic.workflows.solution.solution_workflow",
    
    # Workflows - RAG subdirectories (now in workflows/)
    "from agentic.workflows.standards_rag.": "from agentic.workflows.standards_rag.",
    "from agentic.workflows.strategy_rag.": "from agentic.workflows.strategy_rag.",
    "from agentic.workflows.index_rag.": "from agentic.workflows.index_rag.",
    "from agentic.workflows.engenie_chat.": "from agentic.workflows.engenie_chat.",
    
    # Deep Agent - reorganized
    "from agentic.deep_agent.workflows.workflows.deep_agentic_workflow": "from agentic.deep_agent.workflows.workflows.deep_agentic_workflow",
    "from agentic.deep_agent.workflows.workflow": "from agentic.deep_agent.workflows.workflows.workflow",
    
    # Deep Agent - Schema Generation
    "from agentic.deep_agent.schema.generation.deep_agent": "from agentic.deep_agent.schema.generation.deep_agent",
    "from agentic.deep_agent.schema.generation.field_extractor": "from agentic.deep_agent.schema.generation.field_extractor",
    "from agentic.deep_agent.schema.generation.async_generator": "from agentic.deep_agent.schema.generation.async_generator",
    "from agentic.deep_agent.schema.generation.parallel_generator": "from agentic.deep_agent.schema.generation.parallel_generator",
    "from agentic.deep_agent.schema.populator": "from agentic.deep_agent.schema.populator",
    "from agentic.deep_agent.schema.failure_memory": "from agentic.deep_agent.schema.failure_memory",
    "from agentic.deep_agent.schema.populator_legacy": "from agentic.deep_agent.schema.populator_legacy",
    
    # Deep Agent - Specifications
    "from agentic.deep_agent.specifications.generation.llm_generator": "from agentic.deep_agent.specifications.generation.llm_generator",
    "from agentic.deep_agent.specifications.generation.orchestrator": "from agentic.deep_agent.specifications.generation.orchestrator",
    "from agentic.deep_agent.specifications.generation.parallel_enrichment": "from agentic.deep_agent.specifications.generation.parallel_enrichment",
    "from agentic.deep_agent.specifications.verification.verifier": "from agentic.deep_agent.specifications.verification.verifier",
    "from agentic.deep_agent.specifications.verification.normalizer": "from agentic.deep_agent.specifications.verification.normalizer",
    "from agentic.deep_agent.specifications.aggregator": "from agentic.deep_agent.specifications.aggregator",
    "from agentic.deep_agent.specifications.templates.templates": "from agentic.deep_agent.specifications.templates.templates",
    "from agentic.deep_agent.specifications.templates.augmented": "from agentic.deep_agent.specifications.templates.augmented",
    
    # Deep Agent - Standards
    "from agentic.deep_agent.standards.deep_agent": "from agentic.deep_agent.standards.deep_agent",
    "from agentic.deep_agent.standards.integration": "from agentic.deep_agent.standards.integration",
    
    # Deep Agent - Orchestration
    "from agentic.deep_agent.orchestration.batch_orchestrator": "from agentic.deep_agent.orchestration.batch_orchestrator",
    "from agentic.deep_agent.orchestration.stateless_orchestrator": "from agentic.deep_agent.orchestration.stateless_orchestrator",
    "from agentic.deep_agent.orchestration.enrichment_worker": "from agentic.deep_agent.orchestration.enrichment_worker",
    "from agentic.deep_agent.orchestration.task_queue": "from agentic.deep_agent.orchestration.task_queue",
    "from agentic.deep_agent.orchestration.worker_runner": "from agentic.deep_agent.orchestration.worker_runner",
    
    # Deep Agent - Processing
    "from agentic.deep_agent.processing.parallel.optimized_agent": "from agentic.deep_agent.processing.parallel.optimized_agent",
    "from agentic.deep_agent.processing.parallel.enrichment_engine": "from agentic.deep_agent.processing.parallel.enrichment_engine",
    "from agentic.deep_agent.processing.parallel.processor": "from agentic.deep_agent.processing.parallel.processor",
    "from agentic.deep_agent.processing.normalizers": "from agentic.deep_agent.processing.normalizers",
    "from agentic.deep_agent.processing.value_normalizer": "from agentic.deep_agent.processing.value_normalizer",
    
    # Deep Agent - Memory
    "from agentic.deep_agent.memory.memory": "from agentic.deep_agent.memory.memory.memory",
    "from agentic.deep_agent.memory.memory.state_manager": "from agentic.deep_agent.memory.memory.state_manager",
    
    # Deep Agent - Agents
    "from agentic.deep_agent.agents.sub_agents": "from agentic.deep_agent.agents.sub_agents",
    "from agentic.deep_agent.agents.adaptive_prompt_engine": "from agentic.deep_agent.agents.adaptive_prompt_engine",
    
    # Deep Agent - Documents
    "from agentic.deep_agent.documents.loader": "from agentic.deep_agent.documents.loader",
    
    # Deep Agent - Utils
    "from agentic.deep_agent.utils.compatibility_wrapper": "from agentic.deep_agent.utils.compatibility_wrapper",
    "from agentic.deep_agent.utils.integration": "from agentic.deep_agent.utils.integration",
    "from agentic.deep_agent.utils.learning_engine": "from agentic.deep_agent.utils.learning_engine",
    "from agentic.deep_agent.utils.orchestrators": "from agentic.deep_agent.utils.orchestrators",
}

# Directories to process
BACKEND_DIR = r"d:\AI PR\AIPR\backend"
EXCLUDE_DIRS = {".git", "__pycache__", "venv", ".pytest_cache", "node_modules", "chroma_data"}


def update_imports_in_file(filepath):
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
    sorted_mappings = sorted(IMPORT_MAPPINGS.items(), key=lambda x: -len(x[0]))
    
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


def process_directory(directory):
    """Process all Python files in a directory recursively."""
    total_files = 0
    modified_files = 0
    total_changes = 0
    
    for root, dirs, files in os.walk(directory):
        # Exclude certain directories
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        
        for filename in files:
            if filename.endswith('.py'):
                filepath = os.path.join(root, filename)
                total_files += 1
                modified, changes = update_imports_in_file(filepath)
                if modified:
                    modified_files += 1
                    total_changes += changes
                    print(f"Modified: {filepath} ({changes} changes)")
    
    return total_files, modified_files, total_changes


def main():
    print("=" * 60)
    print("Agentic Import Update Script")
    print("=" * 60)
    print(f"\nProcessing directory: {BACKEND_DIR}")
    print("-" * 60)
    
    total_files, modified_files, total_changes = process_directory(BACKEND_DIR)
    
    print("-" * 60)
    print(f"\nSummary:")
    print(f"  Total Python files scanned: {total_files}")
    print(f"  Files modified: {modified_files}")
    print(f"  Total import changes: {total_changes}")
    print("=" * 60)


if __name__ == "__main__":
    main()
