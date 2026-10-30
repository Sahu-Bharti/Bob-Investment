"""
IBM Granite LLM client for the Investment Agentic AI.
Includes retry logic, timeout handling, and graceful fallbacks.
"""

from __future__ import annotations
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="ibm_watsonx_ai")
warnings.filterwarnings("ignore", message=".*deprecated.*", category=Warning)

from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.schema import TextGenParameters

from config import (
    IBM_WATSONX_API_KEY,
    IBM_WATSONX_PROJECT_ID,
    IBM_WATSONX_URL,
    GRANITE_MODEL_ID,
    GENERATION_PARAMS,
    LLM_MAX_RETRIES,
    LLM_RETRY_DELAY_S,
)


def _build_model() -> ModelInference:
    credentials = Credentials(
        url=IBM_WATSONX_URL,
        api_key=IBM_WATSONX_API_KEY,
    )
    params = TextGenParameters(
        max_new_tokens=GENERATION_PARAMS["max_new_tokens"],
        min_new_tokens=GENERATION_PARAMS["min_new_tokens"],
        temperature=GENERATION_PARAMS["temperature"],
        top_k=GENERATION_PARAMS["top_k"],
        top_p=GENERATION_PARAMS["top_p"],
        repetition_penalty=GENERATION_PARAMS["repetition_penalty"],
    )
    return ModelInference(
        model_id=GRANITE_MODEL_ID,
        credentials=credentials,
        project_id=IBM_WATSONX_PROJECT_ID,
        params=params,
    )


# Lazy singleton — avoids re-authentication on every call
_model: ModelInference | None = None


def get_model() -> ModelInference:
    global _model
    if _model is None:
        _model = _build_model()
    return _model


def generate(prompt: str, max_retries: int = LLM_MAX_RETRIES) -> str:
    """
    Send *prompt* to IBM Granite and return the generated text string.
    Retries up to *max_retries* times on transient failures.
    """
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            model = get_model()
            response = model.generate_text(prompt=prompt)
            text = response.strip() if isinstance(response, str) else str(response).strip()
            if text:
                return text
            # Empty response — retry
            raise ValueError("Empty response received from IBM Granite.")
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(LLM_RETRY_DELAY_S * attempt)
                # Force model re-init on auth errors
                if "auth" in str(exc).lower() or "401" in str(exc) or "403" in str(exc):
                    reset_model()
    raise RuntimeError(
        f"IBM Granite call failed after {max_retries} attempts. "
        f"Last error: {last_error}"
    )


def generate_safe(prompt: str, fallback: str = "") -> str:
    """
    Same as generate() but returns *fallback* instead of raising on failure.
    Useful for non-critical LLM calls where we want a graceful degradation.
    """
    try:
        return generate(prompt)
    except Exception as exc:
        return fallback or f"[IBM Granite unavailable: {exc}]"


def reset_model() -> None:
    """Force re-initialisation of the model (e.g., after credential change)."""
    global _model
    _model = None


def is_model_available() -> bool:
    """Return True if IBM Granite can be reached (lightweight connectivity check)."""
    try:
        get_model()
        return True
    except Exception:
        return False
