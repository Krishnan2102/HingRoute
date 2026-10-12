import hashlib
import json
import os
import time
from typing import Any, Callable
from diskcache import Cache
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# Persistent disk cache in .llm_cache/
CACHE_DIR = os.getenv("LLM_CACHE_DIR", ".llm_cache")
cache = Cache(CACHE_DIR)

_groq_client = None


def get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set.")
        _groq_client = Groq(api_key=api_key)
    return _groq_client


def get_cache_key(model: str, prompt: str, **kwargs: Any) -> str:
    payload = json.dumps({"model": model, "prompt": prompt, "kwargs": kwargs}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cached_call(
    model: str,
    prompt: str,
    call_fn: Callable[..., str],
    max_retries: int = 5,
    initial_delay: float = 1.0,
    **kwargs: Any,
) -> str:
    key = get_cache_key(model, prompt, **kwargs)

    if key in cache:
        return cache[key]

    delay = initial_delay
    last_exception = None

    for attempt in range(max_retries):
        try:
            response_text = call_fn(model=model, prompt=prompt, **kwargs)
            cache[key] = response_text
            return response_text
        except Exception as e:
            last_exception = e
            time.sleep(delay)
            delay *= 2.0

    raise RuntimeError(f"Groq API call failed after {max_retries} attempts: {last_exception}")


def _call_groq(model: str, prompt: str, **kwargs: Any) -> str:
    client = get_groq_client()
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=kwargs.get("temperature", 0.7),
    )
    return completion.choices[0].message.content


def cached_generate(model: str = "llama-3.1-8b-instant", prompt: str = "", **kwargs: Any) -> str:
    """Convenience wrapper for cached generation with Groq."""
    return cached_call(model=model, prompt=prompt, call_fn=_call_groq, **kwargs)