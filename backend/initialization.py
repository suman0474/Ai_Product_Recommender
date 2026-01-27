"""
Application Initialization Module
==================================

Single point of initialization for the entire application.
Ensures environment variables are loaded once, configurations are validated,
and all singletons are properly initialized.

This module MUST be called FIRST before any other imports in main.py.
"""

import os
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def initialize_application() -> bool:
    """
    Single point of initialization for the entire application.
    Called ONCE at startup in main.py before any other imports.

    Returns:
        bool: True if initialization successful

    Raises:
        RuntimeError: If critical configuration is missing or invalid
    """
    logger.info("="*70)
    logger.info("[INIT] Starting application initialization")
    logger.info("="*70)

    # Step 1: Load environment variables ONCE
    logger.info("[INIT] Step 1: Loading environment variables...")
    load_dotenv()
    logger.info("[INIT] ✓ Environment variables loaded from .env")

    # Step 2: Validate all configurations
    logger.info("[INIT] Step 2: Validating configurations...")
    try:
        validate_all_configs()
        logger.info("[INIT] ✓ All configurations validated")
    except RuntimeError as e:
        logger.error(f"[INIT] ✗ Configuration validation failed: {e}")
        raise

    # Step 3: Initialize singletons
    logger.info("[INIT] Step 3: Initializing singletons...")
    try:
        initialize_singletons()
        logger.info("[INIT] ✓ All singletons initialized")
    except Exception as e:
        logger.error(f"[INIT] ✗ Singleton initialization failed: {e}")
        raise RuntimeError(f"Singleton initialization failed: {e}")

    logger.info("="*70)
    logger.info("[INIT] Application initialization complete")
    logger.info("="*70)

    return True


def validate_all_configs() -> None:
    """
    Validate all configuration after environment loading.
    Raises RuntimeError if any critical config is missing/invalid.

    This runs AFTER load_dotenv() to ensure all environment variables
    are available for validation.
    """
    errors = []

    # Validate Pinecone Configuration
    try:
        from config import PineconeConfig
        PineconeConfig.validate()
        logger.info("[INIT]   ✓ PineconeConfig validated")
    except ValueError as e:
        errors.append(f"PineconeConfig: {e}")
    except ImportError:
        logger.warning("[INIT]   ⚠ PineconeConfig not available (optional)")

    # Validate Agentic Configuration
    try:
        from config import AgenticConfig
        AgenticConfig.validate()
        logger.info("[INIT]   ✓ AgenticConfig validated")
    except ValueError as e:
        errors.append(f"AgenticConfig: {e}")
    except ImportError:
        logger.warning("[INIT]   ⚠ AgenticConfig not available (optional)")

    # Validate Azure Configuration (optional)
    try:
        azure_conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        if azure_conn_str:
            logger.info("[INIT]   ✓ Azure Storage configured")
        else:
            logger.warning("[INIT]   ⚠ Azure Storage not configured (optional)")
    except Exception as e:
        logger.warning(f"[INIT]   ⚠ Azure validation warning: {e}")

    # Validate Google API Keys
    google_key = os.getenv("GOOGLE_API_KEY")
    if google_key:
        logger.info("[INIT]   ✓ GOOGLE_API_KEY configured")
    else:
        logger.warning("[INIT]   ⚠ GOOGLE_API_KEY not set - some features may be unavailable")

    # Validate OpenAI API Key (fallback)
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        logger.info("[INIT]   ✓ OPENAI_API_KEY configured (fallback available)")
    else:
        logger.warning("[INIT]   ⚠ OPENAI_API_KEY not set - no LLM fallback available")

    # Validate Google Custom Search (optional)
    google_cx = os.getenv("GOOGLE_CX")
    if google_cx:
        logger.info("[INIT]   ✓ GOOGLE_CX configured (image search available)")
    else:
        logger.warning("[INIT]   ⚠ GOOGLE_CX not set - image search may be limited")

    # If there are critical errors, raise exception
    if errors:
        error_msg = "Critical configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        logger.error(error_msg)
        raise RuntimeError(error_msg)


def initialize_singletons() -> None:
    """
    Initialize all singleton instances at startup.
    Ensures consistent state across application.

    This should be called AFTER validate_all_configs() to ensure
    all required environment variables are present.
    """
    # Initialize API Key Manager
    try:
        from config.api_key_manager import api_key_manager
        google_count = api_key_manager.get_google_key_count()
        logger.info(f"[INIT]   ✓ API Key Manager initialized: {google_count} Google key(s) available")

        if google_count == 0:
            logger.warning("[INIT]   ⚠ No Google API keys configured")
        elif google_count > 1:
            logger.info(f"[INIT]   ✓ API key rotation enabled with {google_count} keys")

    except ImportError as e:
        logger.warning(f"[INIT]   ⚠ API Key Manager not available: {e}")
    except Exception as e:
        logger.error(f"[INIT]   ✗ API Key Manager initialization failed: {e}")
        raise

    # Initialize Azure Blob Manager (if configured)
    try:
        from azure_blob_config import azure_blob_manager
        if azure_blob_manager.is_configured():
            azure_blob_manager.get_container_client()
            logger.info("[INIT]   ✓ Azure Blob Manager connected")
        else:
            logger.warning("[INIT]   ⚠ Azure Blob Manager not configured (optional)")
    except ImportError:
        logger.warning("[INIT]   ⚠ Azure Blob Manager not available (optional)")
    except Exception as e:
        logger.warning(f"[INIT]   ⚠ Azure Blob Manager initialization failed: {e}")
        # Don't raise - Azure is optional

    # Initialize Protected Caches (PHASE 2 FIX)
    try:
        from advanced_parameters import IN_MEMORY_ADVANCED_SPEC_CACHE, SCHEMA_PARAM_CACHE

        # Set admin token for protected caches
        cache_admin_token = os.getenv("CACHE_ADMIN_TOKEN")
        if cache_admin_token:
            IN_MEMORY_ADVANCED_SPEC_CACHE.set_admin_token(cache_admin_token)
            SCHEMA_PARAM_CACHE.set_admin_token(cache_admin_token)
            logger.info("[INIT]   ✓ Protected caches configured with admin token")
        else:
            logger.warning("[INIT]   ⚠ CACHE_ADMIN_TOKEN not set - caches will require token to clear")

    except ImportError:
        logger.warning("[INIT]   ⚠ Cache modules not available")
    except Exception as e:
        logger.error(f"[INIT]   ✗ Cache initialization failed: {e}")

    # Log initialization summary
    logger.info("[INIT]   ✓ Singleton initialization complete")


def check_initialization_status() -> dict:
    """
    Check the status of all initialized components.
    Useful for health checks and debugging.

    Returns:
        dict: Status of each component
    """
    status = {
        "environment_loaded": True,  # If we got here, env is loaded
        "google_api_keys": 0,
        "openai_configured": False,
        "azure_configured": False,
        "pinecone_configured": False,
    }

    # Check API keys
    try:
        from config.api_key_manager import api_key_manager
        status["google_api_keys"] = api_key_manager.get_google_key_count()
    except:
        pass

    status["openai_configured"] = bool(os.getenv("OPENAI_API_KEY"))
    status["azure_configured"] = bool(os.getenv("AZURE_STORAGE_CONNECTION_STRING"))
    status["pinecone_configured"] = bool(os.getenv("PINECONE_API_KEY"))

    return status


if __name__ == "__main__":
    # Test initialization
    try:
        initialize_application()
        print("\n✓ Initialization successful")
        print("\nStatus:")
        status = check_initialization_status()
        for key, value in status.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"\n✗ Initialization failed: {e}")
