import hashlib
import json
import os
import time
from typing import Any, Callable

from diskcache import Cache
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

CACHE_DIR = os.getenv("LLM_CACHE_DIR", ".llm_cache")
cache = Cache(CACHE_DIR)

# Initialize a single Groq client
api_key = os.getenv("GROQ_API_KEY")
if not api_key or not api_key.strip():
    raise ValueError("Missing GROQ_API_KEY in environment variables.")

client = Groq(api_key=api_key.strip())


def get_cache_key(model: str, prompt: str, **kwargs: Any) -> str:
    """Generate a deterministic SHA-256 key from model name, prompt, and kwargs."""
    payload = json.dumps({"model": model, "prompt": prompt, "kwargs": kwargs}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cached_call(
    model: str,
    prompt: str,
    call_fn: Callable[..., str],
    max_retries: int = 5,
    initial_delay: float = 2.0,
    **kwargs: Any,
) -> str:
    """Cached LLM wrapper with exponential backoff on retryable errors."""
    key = get_cache_key(model, prompt, **kwargs)

    # 1. Check cache first
    if key in cache:
        return cache[key]

    # 2. Cache miss -> retry loop
    delay = initial_delay
    last_exception = None

    for attempt in range(max_retries):
        try:
            response_text = call_fn(model=model, prompt=prompt, **kwargs)
            cache[key] = response_text
            return response_text
        except Exception as e:
            last_exception = e
            print(f"API Error (Attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay}s...")
            time.sleep(delay)
            delay *= 2.0

    raise RuntimeError(f"API call failed after {max_retries} attempts: {last_exception}")


def _call_groq(model: str, prompt: str, **kwargs: Any) -> str:
    """Execute Chat Completion using the single Groq client."""
    temperature = kwargs.pop("temperature", 0.7)

    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        **kwargs,  # Forwards extra arguments (e.g. response_format, max_tokens)
    )
    return completion.choices[0].message.content or ""


def cached_generate(
    prompt: str = "",
    model: str = "openai/gpt-oss-20b",
    **kwargs: Any,
) -> str:
    """Cached generation entry point using a single client."""
    return cached_call(model=model, prompt=prompt, call_fn=_call_groq, **kwargs)