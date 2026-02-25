"""
Ollama LLM wrapper — supports both streaming and non-streaming calls.
"""
from __future__ import annotations

from typing import Generator

import ollama

from mcu_rag.config import LLM_MODEL, OLLAMA_BASE_URL


def _get_client() -> ollama.Client:
    return ollama.Client(host=OLLAMA_BASE_URL)


def generate(
    prompt: str,
    system: str = "",
    model: str = LLM_MODEL,
    temperature: float = 0.1,
) -> str:
    """Non-streaming generation. Returns the full response text."""
    client = _get_client()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        response = client.chat(
            model=model,
            messages=messages,
            options={"temperature": temperature},
        )
        return response["message"]["content"]
    except Exception as exc:
        raise RuntimeError(
            f"Ollama generation failed — is Ollama running? (model: {model})\n{exc}"
        ) from exc


def stream_generate(
    prompt: str,
    system: str = "",
    model: str = LLM_MODEL,
    temperature: float = 0.1,
) -> Generator[str, None, None]:
    """
    Streaming generation — yields text tokens one at a time.
    Use with Streamlit's st.write_stream().
    Raises RuntimeError if Ollama is unreachable.
    """
    client = _get_client()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        stream = client.chat(
            model=model,
            messages=messages,
            stream=True,
            options={"temperature": temperature},
        )
        for chunk in stream:
            token = chunk.get("message", {}).get("content", "")
            if token:
                yield token
    except Exception as exc:
        raise RuntimeError(
            f"Ollama streaming failed — is Ollama running? (model: {model})\n{exc}"
        ) from exc


def is_ollama_running(model: str = LLM_MODEL) -> bool:
    """Check if Ollama is reachable and the model is available."""
    try:
        client = _get_client()
        models = client.list()
        available = [m["model"] for m in models.get("models", [])]
        # Check if model name (without tag) matches
        model_base = model.split(":")[0]
        return any(model_base in m for m in available)
    except Exception:
        return False


def list_available_models() -> list[str]:
    """Return list of available Ollama model names."""
    try:
        client = _get_client()
        models = client.list()
        return [m["model"] for m in models.get("models", [])]
    except Exception:
        return []
