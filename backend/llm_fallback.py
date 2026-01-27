"""
LLM Fallback Utility
Provides automatic fallback from Google Gemini to OpenAI when Gemini fails
Includes timeout support for LLM calls

PHASE 1 FIX: Uses centralized API Key Manager instead of global state
"""
import os
import logging
import threading
from typing import Optional, Any
from functools import wraps

logger = logging.getLogger(__name__)

# PHASE 1 FIX: Use centralized API Key Manager
# Replaces global state with thread-safe singleton
try:
    from config.api_key_manager import api_key_manager
    GOOGLE_API_KEY = api_key_manager.get_current_google_key()
    OPENAI_API_KEY = api_key_manager.get_openai_key()
    GOOGLE_API_KEYS = [api_key_manager.get_google_key_by_index(i)
                       for i in range(api_key_manager.get_google_key_count())]
except ImportError:
    # Fallback if api_key_manager not available (initialization phase)
    logger.warning("[LLM_FALLBACK] API Key Manager not available, using environment variables directly")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GOOGLE_API_KEYS = [GOOGLE_API_KEY] if GOOGLE_API_KEY else []

def get_current_google_api_key() -> str:
    """
    Get the current Google API key.
    PHASE 1 FIX: Uses API Key Manager instead of global state.
    """
    try:
        from config.api_key_manager import api_key_manager
        return api_key_manager.get_current_google_key() or ""
    except ImportError:
        return GOOGLE_API_KEY or ""

def rotate_google_api_key() -> bool:
    """
    Rotate to the next available Google API key.
    PHASE 1 FIX: Uses API Key Manager (thread-safe) instead of global state.

    Returns:
        True if rotation succeeded, False if only one key available
    """
    try:
        from config.api_key_manager import api_key_manager
        return api_key_manager.rotate_google_key()
    except ImportError:
        logger.warning("[LLM_FALLBACK] API Key Manager not available, cannot rotate")
        return False

# Log available keys at startup
if len(GOOGLE_API_KEYS) > 1:
    logger.info(f"[LLM_FALLBACK] 🔑 Loaded {len(GOOGLE_API_KEYS)} Google API keys for rotation")

# Model mappings: Gemini -> OpenAI equivalent
MODEL_MAPPINGS = {
    "gemini-2.5-flash": "gpt-4o-mini",
    "gemini-2.5-pro": "gpt-4o",
    "gemini-1.5-pro": "gpt-4o",
}


class LLMTimeoutError(Exception):
    """Exception raised when LLM call exceeds timeout"""
    pass


class LLMWithTimeout:
    """
    Wrapper for LLM that adds timeout functionality
    Uses threading for cross-platform compatibility (Windows-safe)
    """

    def __init__(self, base_llm: Any, timeout_seconds: int = 60):
        """
        Initialize LLM with timeout wrapper

        Args:
            base_llm: The underlying LLM instance
            timeout_seconds: Maximum seconds to wait for LLM response (default: 60)
        """
        self.base_llm = base_llm
        self.timeout_seconds = timeout_seconds
        self.model_name = getattr(base_llm, 'model_name', getattr(base_llm, 'model', 'unknown'))

    def _invoke_with_timeout(self, method_name: str, *args, **kwargs):
        """
        Execute LLM method with timeout using threading

        Args:
            method_name: Name of the method to call ('invoke', 'ainvoke', etc.)
            *args: Positional arguments to pass to the method
            **kwargs: Keyword arguments to pass to the method

        Returns:
            Result from the LLM method

        Raises:
            LLMTimeoutError: If the call exceeds timeout_seconds
        """
        result = [None]
        error = [None]

        def target():
            try:
                method = getattr(self.base_llm, method_name)
                result[0] = method(*args, **kwargs)
            except Exception as e:
                error[0] = e

        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(timeout=self.timeout_seconds)

        if thread.is_alive():
            logger.error(f"[LLM_TIMEOUT] {method_name} call exceeded {self.timeout_seconds}s timeout (model: {self.model_name})")
            raise LLMTimeoutError(
                f"LLM call exceeded {self.timeout_seconds}s timeout (model: {self.model_name})"
            )

        if error[0]:
            raise error[0]

        return result[0]

    def invoke(self, *args, **kwargs):
        """
        Invoke the LLM with timeout protection

        Args:
            *args: Positional arguments to pass to LLM.invoke()
            **kwargs: Keyword arguments to pass to LLM.invoke()

        Returns:
            LLM response

        Raises:
            LLMTimeoutError: If the call exceeds timeout
        """
        return self._invoke_with_timeout('invoke', *args, **kwargs)

    def batch(self, *args, **kwargs):
        """Batch invoke with timeout"""
        return self._invoke_with_timeout('batch', *args, **kwargs)

    def stream(self, *args, **kwargs):
        """Stream invoke (no timeout applied for streaming)"""
        return self.base_llm.stream(*args, **kwargs)

    def __getattr__(self, name):
        """Proxy all other attributes to the base LLM"""
        return getattr(self.base_llm, name)


def get_openai_equivalent(gemini_model: str) -> str:
    """
    Get the OpenAI model equivalent for a Gemini model

    Args:
        gemini_model: Gemini model name

    Returns:
        OpenAI model name
    """
    return MODEL_MAPPINGS.get(gemini_model, "gpt-4o-mini")


def create_llm_with_fallback(
    model: str = "gemini-2.5-flash",
    temperature: float = 0.1,
    google_api_key: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    max_tokens: Optional[int] = None,
    timeout: Optional[int] = None,
    skip_test: bool = True,
    max_afc_calls: Optional[int] = None,
    **kwargs
) -> Any:
    """
    Create an LLM instance with automatic fallback from Gemini to OpenAI and timeout support.
    
    🔄 MULTI-KEY ROTATION: If multiple GOOGLE_API_KEY, GOOGLE_API_KEY2, etc. are configured,
    the function will try each key in sequence when quota is exhausted (429 errors).
    
    🔀 OPENAI FALLBACK: After all Google keys are exhausted, falls back to OpenAI.

    Args:
        model: Gemini model name (will be mapped to OpenAI if fallback is needed)
        temperature: Temperature for generation
        google_api_key: Google API key (optional, uses env var if not provided)
        openai_api_key: OpenAI API key (optional, uses env var if not provided)
        max_tokens: Maximum tokens for generation
        timeout: Timeout in seconds for LLM calls (set to None to disable)
        skip_test: If True, skip the test LLM call during initialization (saves ~1.5s)
        max_afc_calls: (FUTURE) Max concurrent AFC calls when API supports it
        **kwargs: Additional arguments to pass to the LLM

    Returns:
        LLM instance (ChatGoogleGenerativeAI or ChatOpenAI) - always a Runnable!
    """
    openai_key = openai_api_key or OPENAI_API_KEY
    
    # Determine which keys to try
    keys_to_try = GOOGLE_API_KEYS if GOOGLE_API_KEYS else ([google_api_key] if google_api_key else [GOOGLE_API_KEY] if GOOGLE_API_KEY else [])
    keys_to_try = [k for k in keys_to_try if k]  # Filter out None/empty
    
    last_error = None
    
    # Try each Google API key
    for key_idx, google_key in enumerate(keys_to_try):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            logger.info(f"[LLM_FALLBACK] Creating Google Gemini: {model} (key #{key_idx + 1}/{len(keys_to_try)}, skip_test={skip_test})")

            # Prepare AFC configuration for when API supports it
            local_kwargs = kwargs.copy()
            if max_afc_calls:
                logger.info(f"[LLM_FALLBACK] AFC concurrent limit set to {max_afc_calls}")
                local_kwargs['max_afc_calls'] = max_afc_calls

            llm = ChatGoogleGenerativeAI(
                model=model,
                temperature=temperature,
                google_api_key=google_key,
                max_output_tokens=max_tokens,
                **local_kwargs
            )

            # Skip test call if specified (saves ~1.5s per initialization)
            if skip_test:
                logger.info(f"[LLM_FALLBACK] Skipping test call, returning LLM directly: {model}")
                # Wrap with timeout if specified
                if timeout is not None:
                    logger.info(f"[LLM_FALLBACK] Wrapping LLM with {timeout}s timeout")
                    return LLMWithTimeout(llm, timeout_seconds=timeout)
                return llm
            else:
                # Test the model with a simple call to verify it works
                try:
                    _ = llm.invoke("test")
                    logger.info(f"[LLM_FALLBACK] ✅ Successfully initialized Google Gemini: {model}")
                    # Wrap with timeout if specified
                    if timeout is not None:
                        logger.info(f"[LLM_FALLBACK] Wrapping LLM with {timeout}s timeout")
                        return LLMWithTimeout(llm, timeout_seconds=timeout)
                    return llm
                except Exception as test_error:
                    error_str = str(test_error)
                    # Check if this is a quota error - try next key
                    if '429' in error_str or 'RESOURCE_EXHAUSTED' in error_str or 'quota' in error_str.lower():
                        logger.warning(f"[LLM_FALLBACK] ⚠️ Key #{key_idx + 1} quota exhausted, trying next key...")
                        last_error = test_error
                        continue
                    else:
                        logger.warning(f"[LLM_FALLBACK] Gemini model test failed: {test_error}")
                        raise test_error

        except ImportError as ie:
            logger.error(f"[LLM_FALLBACK] Failed to import langchain_google_genai: {ie}")
            last_error = ie
            break  # No point trying other keys if import fails
        except Exception as e:
            error_str = str(e)
            # Check if this is a quota error - try next key
            if '429' in error_str or 'RESOURCE_EXHAUSTED' in error_str or 'quota' in error_str.lower():
                logger.warning(f"[LLM_FALLBACK] ⚠️ Key #{key_idx + 1} quota exhausted, trying next key...")
                last_error = e
                continue
            else:
                logger.warning(f"[LLM_FALLBACK] Failed to initialize Gemini ({model}): {e}")
                last_error = e
                # Try next key for any error
                continue

    # ==========================================================================
    # OPENAI FALLBACK - ENABLED
    # ==========================================================================
    # If we reach here, all Google keys failed or none were available
    
    openai_model = get_openai_equivalent(model)
    
    if openai_key:
        try:
            from langchain_openai import ChatOpenAI

            logger.info(f"[LLM_FALLBACK] 🔀 Using OpenAI fallback: {openai_model} (equivalent to {model})")

            llm = ChatOpenAI(
                model=openai_model,
                temperature=temperature,
                openai_api_key=openai_key,
                max_tokens=max_tokens,
                **kwargs
            )

            logger.info(f"[LLM_FALLBACK] ✅ Successfully initialized OpenAI: {openai_model}")

            if timeout is not None:
                logger.info(f"[LLM_FALLBACK] Wrapping LLM with {timeout}s timeout")
                return LLMWithTimeout(llm, timeout_seconds=timeout)
            return llm

        except ImportError as ie:
            logger.error(f"[LLM_FALLBACK] Failed to import langchain_openai: {ie}")
            raise RuntimeError(f"Both Gemini and OpenAI failed. OpenAI import error: {ie}. Install with: pip install langchain-openai")
        except Exception as e:
            logger.error(f"[LLM_FALLBACK] Failed to initialize OpenAI ({openai_model}): {e}")
            raise RuntimeError(f"Both Gemini and OpenAI initialization failed. Last error: {e}")
    else:
        # No OpenAI key available
        if last_error:
            error_msg = f"All {len(keys_to_try)} Google API key(s) failed. Last error: {last_error}"
            if '429' in str(last_error) or 'RESOURCE_EXHAUSTED' in str(last_error):
                error_msg += "\n💡 TIP: Add OPENAI_API_KEY to .env for automatic fallback, or add more GOOGLE_API_KEY2, GOOGLE_API_KEY3, etc."
            raise RuntimeError(error_msg)
        else:
            raise RuntimeError("No API keys available. Set GOOGLE_API_KEY or OPENAI_API_KEY in .env")


def create_llm_langchain(
    model: str = "gemini-2.5-flash",
    temperature: float = 0.1,
    google_api_key: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    **kwargs
) -> Any:
    """
    Alias for create_llm_with_fallback for LangChain compatibility

    This is the recommended function to use across the codebase.
    """
    return create_llm_with_fallback(
        model=model,
        temperature=temperature,
        google_api_key=google_api_key,
        openai_api_key=openai_api_key,
        **kwargs
    )


from agentic.context_managers import LLMResourceManager, GlobalResourceRegistry

class FallbackLLMClient:
    """
    Wrapper class for non-LangChain LLM usage with fallback support.
    Supports multi-key rotation for Google API keys.
    Similar to GeminiClient but with OpenAI fallback.
    """

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        temperature: float = 0.1,
        api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.openai_api_key = openai_api_key or OPENAI_API_KEY
        self.client = None
        self.client_type = None  # 'gemini' or 'openai'
        self._current_key_idx = 0
        
        # Determine which Google API keys to use
        if api_key:
            self._google_api_keys = [api_key]
        elif GOOGLE_API_KEYS:
            self._google_api_keys = GOOGLE_API_KEYS.copy()
        elif GOOGLE_API_KEY:
            self._google_api_keys = [GOOGLE_API_KEY]
        else:
            self._google_api_keys = []
        
        # Context manager support
        self._resource_manager: Optional[LLMResourceManager] = None
        self._registry = GlobalResourceRegistry()

        self._initialize_client()
        
    def __enter__(self):
        """Enable context manager usage"""
        self._resource_manager = LLMResourceManager(
            "llm_client",
            f"fallback_{self.model_name}_{id(self)}",
            timeout_seconds=150
        )
        self._resource_manager.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup LLM resources"""
        if self._resource_manager:
            self._resource_manager.__exit__(exc_type, exc_val, exc_tb)
        return False

    def _initialize_client(self):
        """Initialize the LLM client with multi-key rotation and fallback logic."""
        # Try each Google API key
        for key_idx, google_key in enumerate(self._google_api_keys):
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    import google.generativeai as genai
                genai.configure(api_key=google_key)
                self.client = genai.GenerativeModel(self.model_name)
                self.client_type = 'gemini'
                self._current_key_idx = key_idx
                logger.info(f"[FallbackLLMClient] Using Gemini: {self.model_name} (key #{key_idx + 1}/{len(self._google_api_keys)})")
                return
            except Exception as e:
                logger.warning(f"[FallbackLLMClient] Gemini key #{key_idx + 1} failed: {e}")
                continue

        # Fallback to OpenAI
        if self.openai_api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.openai_api_key)
                self.client_type = 'openai'
                self.openai_model = get_openai_equivalent(self.model_name)
                logger.info(f"[FallbackLLMClient] 🔀 Using OpenAI fallback: {self.openai_model}")
                return
            except Exception as e:
                logger.error(f"[FallbackLLMClient] OpenAI initialization failed: {e}")
                raise RuntimeError(f"Both Gemini and OpenAI initialization failed")

        raise RuntimeError("No valid API keys available for LLM initialization")
    
    def _rotate_to_next_key(self) -> bool:
        """Try rotating to the next available Google API key. Returns True if successful."""
        if len(self._google_api_keys) <= 1:
            return False
        
        start_idx = self._current_key_idx
        for _ in range(len(self._google_api_keys) - 1):
            next_idx = (self._current_key_idx + 1) % len(self._google_api_keys)
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    import google.generativeai as genai
                genai.configure(api_key=self._google_api_keys[next_idx])
                self.client = genai.GenerativeModel(self.model_name)
                self.client_type = 'gemini'
                self._current_key_idx = next_idx
                logger.info(f"[FallbackLLMClient] 🔄 Rotated to key #{next_idx + 1}/{len(self._google_api_keys)}")
                return True
            except Exception as e:
                self._current_key_idx = next_idx
                logger.warning(f"[FallbackLLMClient] Key #{next_idx + 1} rotation failed: {e}")
                continue
        
        return False

    def invoke(self, prompt: str, **kwargs) -> str:
        """
        Invoke the LLM with a prompt. Uses key rotation on quota errors.

        Args:
            prompt: The input prompt
            **kwargs: Additional arguments

        Returns:
            Generated text response
        """
        import time
        
        max_retries = 5
        base_retry_delay = 5  # Base delay in seconds (exponential backoff)
        last_exception = None
        keys_tried = 0  # Track how many keys we've rotated through
        
        for attempt in range(max_retries):
            try:
                if self.client_type == 'gemini':
                    response = self.client.generate_content(
                        prompt,
                        generation_config={
                            'temperature': self.temperature,
                            **kwargs
                        }
                    )
                    return response.text

                elif self.client_type == 'openai':
                    response = self.client.chat.completions.create(
                        model=self.openai_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.temperature,
                        **kwargs
                    )
                    return response.choices[0].message.content

            except Exception as e:
                error_msg = str(e)
                is_rate_limit = any(x in error_msg for x in ['429', 'Resource exhausted', 'RESOURCE_EXHAUSTED', 'quota', '503', 'overloaded'])
                
                if is_rate_limit and self.client_type == 'gemini':
                    # First, try rotating to another Google API key
                    if keys_tried < len(self._google_api_keys) - 1:
                        logger.warning(f"[FallbackLLMClient] ⚠️ Quota exhausted on key #{self._current_key_idx + 1}, trying next key...")
                        if self._rotate_to_next_key():
                            keys_tried += 1
                            continue  # Retry immediately with new key
                    
                    # If no more keys or rotation failed, wait and retry
                    if attempt < max_retries - 1:
                        wait_time = base_retry_delay * (2 ** attempt)
                        logger.warning(f"[FallbackLLMClient] Rate limit hit, retry {attempt + 1}/{max_retries} after {wait_time}s")
                        time.sleep(wait_time)
                        last_exception = e
                        continue
                elif is_rate_limit and attempt < max_retries - 1:
                    # OpenAI rate limits - just do exponential backoff
                    wait_time = base_retry_delay * (2 ** attempt)
                    logger.warning(f"[FallbackLLMClient] Rate limit hit, retry {attempt + 1}/{max_retries} after {wait_time}s")
                    time.sleep(wait_time)
                    last_exception = e
                    continue
                
                logger.error(f"[FallbackLLMClient] Error invoking LLM: {e}")
                last_exception = e

                # Try to failover to OpenAI if using Gemini
                if self.client_type == 'gemini' and self.openai_api_key:
                    logger.info("[FallbackLLMClient] 🔀 Attempting runtime failover to OpenAI...")
                    try:
                        from openai import OpenAI
                        client = OpenAI(api_key=self.openai_api_key)
                        openai_model = get_openai_equivalent(self.model_name)

                        response = client.chat.completions.create(
                            model=openai_model,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=self.temperature,
                            **kwargs
                        )

                        # Update client for future calls
                        self.client = client
                        self.client_type = 'openai'
                        self.openai_model = openai_model
                        logger.info(f"[FallbackLLMClient] ✅ Runtime failover successful to {openai_model}")

                        return response.choices[0].message.content
                    except Exception as fallback_error:
                        logger.error(f"[FallbackLLMClient] ❌ Runtime failover failed: {fallback_error}")

                raise e
        
        # If we exhausted all retries without success
        if last_exception:
            raise last_exception
        raise RuntimeError("LLM invocation failed after retries")


# Convenience functions
def get_default_llm(temperature: float = 0.1, model: str = "gemini-2.5-flash") -> Any:
    """Get a default LLM instance with fallback"""
    return create_llm_with_fallback(model=model, temperature=temperature)


def get_llm_for_task(task_type: str = "general", temperature: float = 0.1) -> Any:
    """
    Get an LLM optimized for a specific task type

    Args:
        task_type: Type of task ('general', 'fast', 'precise', 'creative')
        temperature: Temperature setting

    Returns:
        LLM instance with fallback
    """
    task_configs = {
        "general": {"model": "gemini-2.5-flash", "temp": 0.1},
        "fast": {"model": "gemini-2.5-flash", "temp": 0.0},
        "precise": {"model": "gemini-2.5-flash", "temp": 0.0},
        "creative": {"model": "gemini-2.5-pro", "temp": 0.7},
    }

    config = task_configs.get(task_type, task_configs["general"])
    return create_llm_with_fallback(
        model=config["model"],
        temperature=temperature if temperature != 0.1 else config["temp"]
    )