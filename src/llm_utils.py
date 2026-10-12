import hashlib
import itertools
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

# 1. Initialize a pool of clients from all available keys
_clients = []
for i in range(1, 10):  # Check GROQ_API_KEY_1 through GROQ_API_KEY_9
    key = os.getenv(f"GROQ_API_KEY_{i}")
    if key and key.strip():
        _clients.append(Groq(api_key=key.strip()))

if not _clients:
    # Fallback to single GROQ_API_KEY if present
    single_key = os.getenv("GROQ_API_KEY")
    if single_key and single_key.strip():
        _clients.append(Groq(api_key=single_key.strip()))
    else:
        raise ValueError("No GROQ_API_KEY or GROQ_API_KEY_X environment variables found.")

# 2. Infinite round-robin iterator across API keys
_client_cycle = itertools.cycle(_clients)


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

    # Cache hit
    if key in cache:
        return cache[key]

    delay = initial_delay
    last_exception = None

    # Cache miss -> retry loop
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
    """Fetch next client in rotation and invoke Groq Chat Completions."""
    client = next(_client_cycle)

    temperature = kwargs.pop("temperature", 0.7)

    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        **kwargs,  # Forwards remaining kwargs like response_format, max_tokens, etc.
    )
    return completion.choices[0].message.content or ""


def cached_generate(
    prompt: str = "",
    model: str = "openai/gpt-oss-20b",
    **kwargs: Any,
) -> str:
    """Cached generation entry point defaulting to openai/gpt-oss-20b."""
    return cached_call(model=model, prompt=prompt, call_fn=_call_groq, **kwargs)