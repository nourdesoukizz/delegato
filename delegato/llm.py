"""Thin LiteLLM wrapper with retry logic and JSON parsing."""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

import litellm


class LLMError(Exception):
    """Raised when LLM calls fail after exhausting retries."""


_RETRYABLE_EXCEPTIONS = (
    litellm.RateLimitError,
    litellm.ServiceUnavailableError,
    litellm.InternalServerError,
)

_FATAL_EXCEPTIONS = (
    litellm.AuthenticationError,
    litellm.BadRequestError,
)


async def complete(
    *,
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.0,
    max_tokens: int = 4096,
    response_format: dict | None = None,
    timeout: float = 60.0,
    max_retries: int = 2,
    retry_delay: float = 1.0,
) -> str:
    """Call litellm.acompletion and return the response content string.

    Retries with exponential backoff on transient errors.
    Raises LLMError on exhausted retries or fatal errors.
    """
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "timeout": timeout,
            }
            if response_format is not None:
                kwargs["response_format"] = response_format

            response = await litellm.acompletion(**kwargs)
            return response.choices[0].message.content

        except _FATAL_EXCEPTIONS as exc:
            raise LLMError(str(exc)) from exc

        except _RETRYABLE_EXCEPTIONS as exc:
            last_error = exc
            if attempt < max_retries:
                delay = retry_delay * (2**attempt)
                await asyncio.sleep(delay)

        except Exception as exc:
            raise LLMError(str(exc)) from exc

    raise LLMError(f"Exhausted {max_retries + 1} attempts: {last_error}") from last_error


def _extract_json(text: str) -> Any:
    """Parse JSON from text, handling code-fence wrapped responses."""
    text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from ```json ... ``` code fences
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    raise LLMError(f"Could not parse JSON from response: {text[:200]}")


async def complete_json(
    *,
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.0,
    max_tokens: int = 4096,
    timeout: float = 60.0,
    max_retries: int = 2,
    retry_delay: float = 1.0,
) -> Any:
    """Call complete() with JSON response format and parse the result."""
    raw = await complete(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
        timeout=timeout,
        max_retries=max_retries,
        retry_delay=retry_delay,
    )
    return _extract_json(raw)
