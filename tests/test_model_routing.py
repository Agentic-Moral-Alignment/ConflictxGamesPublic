"""Tests for provider-prefix routing and model-name helpers."""

from types import SimpleNamespace

import pytest

from conflictxgame.common.model_wrappers import (
    AnthropicClient,
    ModelWrapper,
    OpenAIClient,
    split_provider_prefix,
    short_model_name,
)
from conflictxgame.common import model_wrappers


class TestSplitProviderPrefix:
    def test_recognized_prefixes(self):
        assert split_provider_prefix("openai:anthropic.claude-sonnet-5") == (
            "openai", "anthropic.claude-sonnet-5",
        )
        assert split_provider_prefix("paid-openai:gpt-5.6-luna") == (
            "paid-openai", "gpt-5.6-luna",
        )
        assert split_provider_prefix("anthropic:claude-sonnet-5") == (
            "anthropic", "claude-sonnet-5",
        )
        assert split_provider_prefix("vllm:meta-llama/Llama-3-8b") == (
            "vllm", "meta-llama/Llama-3-8b",
        )

    def test_no_prefix_passthrough(self):
        assert split_provider_prefix("gpt-4o-mini") == (None, "gpt-4o-mini")
        assert split_provider_prefix("claude-3-5-sonnet-latest") == (
            None, "claude-3-5-sonnet-latest",
        )

    def test_unrecognized_colons_untouched(self):
        # Windows paths and arbitrary colons are not routing prefixes.
        assert split_provider_prefix(r"C:\models\llama-7b") == (None, r"C:\models\llama-7b")
        assert split_provider_prefix("weird:name") == (None, "weird:name")


class TestShortModelName:
    def test_strips_prefix_and_path(self):
        assert short_model_name("openai:anthropic.claude-sonnet-5") == "anthropic.claude-sonnet-5"
        assert short_model_name("meta-llama/Llama-3-8b") == "Llama-3-8b"
        assert short_model_name("gpt-4o-mini") == "gpt-4o-mini"
        # Filesystem-safe: no colons survive for recognized prefixes.
        assert ":" not in short_model_name("openai:openai.gpt-5.6-luna")


class TestCreateRouting:
    def test_openai_prefix_forces_openai_client_for_claude_id(self, monkeypatch):
        """Gateway-served Claude ids route through the OpenAI-compatible client."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example.edu/api")
        client = ModelWrapper.create("openai:anthropic.claude-sonnet-5", max_tokens=8)
        assert isinstance(client, OpenAIClient)
        # The routing prefix is stripped from the id sent to the API.
        assert client.model_name == "anthropic.claude-sonnet-5"

    def test_paid_openai_prefix_uses_a_client_specific_key_and_endpoint(self, monkeypatch):
        captured = {}

        class FakeOpenAI:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setattr(model_wrappers, "OpenAI", FakeOpenAI)
        monkeypatch.setenv("OPENAI_API_KEY", "gateway-key")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example.edu/api")
        monkeypatch.setenv("PAID_OPENAI_API_KEY", "paid-key")
        client = ModelWrapper.create("paid-openai:gpt-5.6-luna", max_tokens=8)

        assert isinstance(client, OpenAIClient)
        assert client.model_name == "gpt-5.6-luna"
        assert captured == {
            "api_key": "paid-key",
            "base_url": "https://api.openai.com/v1",
        }

    def test_unprefixed_claude_still_routes_to_anthropic(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        # The invariant is the routing itself: an unprefixed claude id must
        # never silently become an OpenAI client (which would send it to the
        # wrong endpoint with the wrong key). Whether construction *raises*
        # depends on the environment — ImportError without the SDK, and some
        # SDK versions defer the missing-key error to request time — so assert
        # the client type when it does construct.
        try:
            client = ModelWrapper.create("claude-3-5-sonnet-latest")
        except Exception:
            return  # SDK missing or key required at construction: also correct
        assert isinstance(client, AnthropicClient)
        assert not isinstance(client, OpenAIClient)


def _thinking(text=None):
    """A thinking block as anthropic 0.45 surfaces it: ``.text`` is None."""
    return SimpleNamespace(type="thinking", thinking="...", text=text)


def _text(text):
    return SimpleNamespace(type="text", text=text)


class TestAnthropicContentBlocks:
    """The reply is the first ``text`` block, wherever it sits.

    Sonnet 5 runs adaptive thinking by default, so its messages arrive as
    ``[thinking, text]``. Reading ``content[0].text`` there yields ``None``,
    which the pipeline's JSON parser cannot take -- every generation call
    would fail before the first outline.
    """

    def _client(self, monkeypatch, content):
        pytest.importorskip("anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        client = ModelWrapper.create("claude-sonnet-5", max_retries=1, max_tokens=8)
        assert isinstance(client, AnthropicClient)
        response = SimpleNamespace(
            content=content,
            usage=SimpleNamespace(input_tokens=1, output_tokens=2),
        )
        client.client = SimpleNamespace(
            messages=SimpleNamespace(create=lambda **kwargs: response)
        )
        return client

    def test_skips_leading_thinking_block(self, monkeypatch):
        client = self._client(monkeypatch, [_thinking(), _text('{"ok": true}')])
        assert client.generate([{"role": "user", "content": "hi"}]) == '{"ok": true}'
        assert client.consecutive_failures == 0

    def test_plain_text_reply_unchanged(self, monkeypatch):
        client = self._client(monkeypatch, [_text("plain")])
        assert client.generate([{"role": "user", "content": "hi"}]) == "plain"

    def test_no_text_block_is_a_failure_not_none(self, monkeypatch):
        # A reply that is all thinking (budget spent before the answer) must
        # take the retry/circuit-breaker path and end as the '' sentinel,
        # never leak None into the callers.
        client = self._client(monkeypatch, [_thinking()])
        assert client.generate([{"role": "user", "content": "hi"}]) == ""
        assert client.consecutive_failures == 1
