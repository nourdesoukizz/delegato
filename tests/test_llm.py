"""Tests for delegato LLM wrapper — all LLM calls are mocked."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

import litellm
from delegato.llm import LLMError, _extract_json, complete, complete_json


# ── Helpers ──────────────────────────────────────────────────────────────────


def _mock_response(content: str):
    """Build a mock litellm response object."""
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


# ── complete() Tests ─────────────────────────────────────────────────────────


class TestComplete:
    @patch("delegato.llm.litellm.acompletion", new_callable=AsyncMock)
    async def test_success(self, mock_acompletion):
        mock_acompletion.return_value = _mock_response("Hello world")
        result = await complete(
            model="openai/gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert result == "Hello world"
        mock_acompletion.assert_awaited_once()

    @patch("delegato.llm.litellm.acompletion", new_callable=AsyncMock)
    async def test_passes_response_format(self, mock_acompletion):
        mock_acompletion.return_value = _mock_response('{"key": "val"}')
        await complete(
            model="openai/gpt-4o",
            messages=[{"role": "user", "content": "json"}],
            response_format={"type": "json_object"},
        )
        call_kwargs = mock_acompletion.call_args.kwargs
        assert call_kwargs["response_format"] == {"type": "json_object"}

    @patch("delegato.llm.litellm.acompletion", new_callable=AsyncMock)
    async def test_retry_on_rate_limit(self, mock_acompletion):
        mock_acompletion.side_effect = [
            litellm.RateLimitError(
                message="rate limited", llm_provider="openai", model="gpt-4o"
            ),
            _mock_response("ok after retry"),
        ]
        result = await complete(
            model="openai/gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
            max_retries=2,
            retry_delay=0.01,
        )
        assert result == "ok after retry"
        assert mock_acompletion.await_count == 2

    @patch("delegato.llm.litellm.acompletion", new_callable=AsyncMock)
    async def test_retry_on_service_unavailable(self, mock_acompletion):
        mock_acompletion.side_effect = [
            litellm.ServiceUnavailableError(
                message="unavailable", llm_provider="openai", model="gpt-4o"
            ),
            _mock_response("recovered"),
        ]
        result = await complete(
            model="openai/gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
            max_retries=1,
            retry_delay=0.01,
        )
        assert result == "recovered"

    @patch("delegato.llm.litellm.acompletion", new_callable=AsyncMock)
    async def test_retry_on_internal_server_error(self, mock_acompletion):
        mock_acompletion.side_effect = [
            litellm.InternalServerError(
                message="server error", llm_provider="openai", model="gpt-4o"
            ),
            _mock_response("ok"),
        ]
        result = await complete(
            model="openai/gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
            max_retries=1,
            retry_delay=0.01,
        )
        assert result == "ok"

    @patch("delegato.llm.litellm.acompletion", new_callable=AsyncMock)
    async def test_exhausted_retries_raises(self, mock_acompletion):
        mock_acompletion.side_effect = litellm.RateLimitError(
            message="rate limited", llm_provider="openai", model="gpt-4o"
        )
        with pytest.raises(LLMError, match="Exhausted"):
            await complete(
                model="openai/gpt-4o",
                messages=[{"role": "user", "content": "Hi"}],
                max_retries=1,
                retry_delay=0.01,
            )
        assert mock_acompletion.await_count == 2  # initial + 1 retry

    @patch("delegato.llm.litellm.acompletion", new_callable=AsyncMock)
    async def test_auth_error_immediate_failure(self, mock_acompletion):
        mock_acompletion.side_effect = litellm.AuthenticationError(
            message="bad key", llm_provider="openai", model="gpt-4o"
        )
        with pytest.raises(LLMError, match="bad key"):
            await complete(
                model="openai/gpt-4o",
                messages=[{"role": "user", "content": "Hi"}],
                max_retries=3,
                retry_delay=0.01,
            )
        # Should NOT retry on auth errors
        assert mock_acompletion.await_count == 1

    @patch("delegato.llm.litellm.acompletion", new_callable=AsyncMock)
    async def test_bad_request_immediate_failure(self, mock_acompletion):
        mock_acompletion.side_effect = litellm.BadRequestError(
            message="invalid model", model="gpt-4o", llm_provider="openai"
        )
        with pytest.raises(LLMError, match="invalid model"):
            await complete(
                model="openai/gpt-4o",
                messages=[{"role": "user", "content": "Hi"}],
                max_retries=3,
                retry_delay=0.01,
            )
        assert mock_acompletion.await_count == 1

    @patch("delegato.llm.litellm.acompletion", new_callable=AsyncMock)
    async def test_unexpected_exception_raises(self, mock_acompletion):
        mock_acompletion.side_effect = RuntimeError("unexpected")
        with pytest.raises(LLMError, match="unexpected"):
            await complete(
                model="openai/gpt-4o",
                messages=[{"role": "user", "content": "Hi"}],
            )


# ── _extract_json() Tests ───────────────────────────────────────────────────


class TestExtractJson:
    def test_plain_json(self):
        result = _extract_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_json_with_whitespace(self):
        result = _extract_json('  \n  {"a": 1}  \n  ')
        assert result == {"a": 1}

    def test_json_in_code_fence(self):
        text = '```json\n{"key": "value"}\n```'
        result = _extract_json(text)
        assert result == {"key": "value"}

    def test_json_in_plain_code_fence(self):
        text = '```\n{"key": "value"}\n```'
        result = _extract_json(text)
        assert result == {"key": "value"}

    def test_json_array(self):
        result = _extract_json('[1, 2, 3]')
        assert result == [1, 2, 3]

    def test_invalid_json_raises(self):
        with pytest.raises(LLMError, match="Could not parse JSON"):
            _extract_json("not json at all")


# ── complete_json() Tests ────────────────────────────────────────────────────


class TestCompleteJson:
    @patch("delegato.llm.litellm.acompletion", new_callable=AsyncMock)
    async def test_returns_parsed_json(self, mock_acompletion):
        mock_acompletion.return_value = _mock_response('{"subtasks": []}')
        result = await complete_json(
            model="openai/gpt-4o",
            messages=[{"role": "user", "content": "decompose"}],
        )
        assert result == {"subtasks": []}

    @patch("delegato.llm.litellm.acompletion", new_callable=AsyncMock)
    async def test_handles_code_fence_response(self, mock_acompletion):
        mock_acompletion.return_value = _mock_response(
            '```json\n{"result": true}\n```'
        )
        result = await complete_json(
            model="openai/gpt-4o",
            messages=[{"role": "user", "content": "test"}],
        )
        assert result == {"result": True}

    @patch("delegato.llm.litellm.acompletion", new_callable=AsyncMock)
    async def test_invalid_json_raises(self, mock_acompletion):
        mock_acompletion.return_value = _mock_response("not json")
        with pytest.raises(LLMError, match="Could not parse JSON"):
            await complete_json(
                model="openai/gpt-4o",
                messages=[{"role": "user", "content": "test"}],
            )
