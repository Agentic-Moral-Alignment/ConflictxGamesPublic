"""Regression tests for OpenAI-compatible gateway quirks.

Both cases below were hit live against the institutional gateway (LiteLLM in front of
Vertex-served Claude) while smoke-testing the game + solo pipeline.
"""

import sys
import types

import pytest

from conflictxgame.common.model_wrappers import short_model_name


class FakeAPIError(Exception):
    """Stands in for openai.APIError, which may not be installed offline."""


@pytest.fixture
def openai_client_cls(monkeypatch):
    """OpenAIClient with its SDK dependencies stubbed out."""
    from conflictxgame.common import model_wrappers as mw

    monkeypatch.setattr(mw, "OpenAI", lambda *a, **k: object(), raising=False)
    monkeypatch.setattr(mw, "APIError", FakeAPIError, raising=False)
    return mw.OpenAIClient


class TestMalformedResponses:
    """The gateway sometimes answers with an unparsed body instead of an object.

    Hit live: ``AttributeError: 'str' object has no attribute 'choices'``
    escaped the retry loop and aborted the run. Any unexpected shape must
    become a retryable :class:`EmptyResponseError` instead.
    """

    def test_bare_string_is_retryable(self):
        from conflictxgame.common.model_wrappers import EmptyResponseError, completion_text

        with pytest.raises(EmptyResponseError, match="bare string"):
            completion_text("upstream connect error or disconnect/reset")

    @pytest.mark.parametrize("response", [None, 42, object()])
    def test_missing_fields_are_retryable(self, response):
        from conflictxgame.common.model_wrappers import EmptyResponseError, completion_text

        with pytest.raises(EmptyResponseError):
            completion_text(response)

    def test_empty_choices_list_is_retryable(self):
        from conflictxgame.common.model_wrappers import EmptyResponseError, completion_text

        with pytest.raises(EmptyResponseError):
            completion_text(types.SimpleNamespace(choices=[]))

    def test_null_content_is_retryable(self):
        from conflictxgame.common.model_wrappers import EmptyResponseError, completion_text

        response = types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=None))]
        )
        with pytest.raises(EmptyResponseError, match="No content"):
            completion_text(response)

    def test_well_formed_response_passes_through(self):
        from conflictxgame.common.model_wrappers import completion_text

        response = types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="PONG"))]
        )
        assert completion_text(response) == "PONG"

    def test_generate_retries_a_bare_string_then_succeeds(self, openai_client_cls, monkeypatch):
        """A transient bare string must not abort the call."""
        client = openai_client_cls("anthropic.claude-sonnet-5", temperature=0.0)
        client._send_temperature = False
        monkeypatch.setattr(client, "_exponential_backoff", lambda attempt: None)

        good = types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="PONG"))]
        )
        replies = ["gateway error text", good]

        def fake_create(**kwargs):
            return replies.pop(0)

        client.client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=fake_create))
        )
        assert client.generate([{"role": "user", "content": "hi"}]) == "PONG"
        assert replies == [], "the retry never happened"

    def test_generate_gives_up_cleanly_when_every_attempt_is_malformed(
        self, openai_client_cls, monkeypatch
    ):
        client = openai_client_cls("anthropic.claude-sonnet-5", temperature=0.0)
        client._send_temperature = False
        monkeypatch.setattr(client, "_exponential_backoff", lambda attempt: None)
        client.client = types.SimpleNamespace(
            chat=types.SimpleNamespace(
                completions=types.SimpleNamespace(create=lambda **kw: "still broken")
            )
        )
        # Returns the empty-string sentinel rather than raising.
        assert client.generate([{"role": "user", "content": "hi"}]) == ""


class TestUnsupportedTemperatureDetection:
    def test_matches_gateway_wording(self, openai_client_cls):
        detect = openai_client_cls._is_unsupported_temperature_error
        # The exact message Vertex-served claude-sonnet-5 returns via LiteLLM.
        assert detect(FakeAPIError(
            "Error code: 400 - {'detail': 'litellm.BadRequestError: "
            "Vertex_aiException BadRequestError - {\"type\":\"error\",\"error\":"
            "{\"type\":\"invalid_request_error\",\"message\":\"`temperature` is "
            "deprecated for this model.\"}}'}"
        ))
        assert detect(FakeAPIError("temperature is not supported for this model"))
        assert detect(FakeAPIError("Unsupported parameter: 'temperature'"))

    def test_ignores_unrelated_errors(self, openai_client_cls):
        detect = openai_client_cls._is_unsupported_temperature_error
        assert not detect(FakeAPIError("rate limit exceeded"))
        assert not detect(FakeAPIError("temperature must be between 0 and 2"))
        assert not detect(FakeAPIError("model is deprecated"))


class TestRequestKwargs:
    def test_temperature_dropped_once_rejected(self, openai_client_cls):
        client = openai_client_cls("anthropic.claude-sonnet-5", temperature=0.0)
        assert client._request_kwargs([])["temperature"] == 0.0
        client._send_temperature = False
        assert "temperature" not in client._request_kwargs([])

    def test_default_top_p_still_omitted(self, openai_client_cls):
        client = openai_client_cls("anthropic.claude-sonnet-5")
        assert "top_p" not in client._request_kwargs([])
        client.top_p = 0.9
        assert client._request_kwargs([])["top_p"] == 0.9

    def test_gpt5_family_still_pinned_to_default_temperature(
        self, openai_client_cls, caplog
    ):
        client = openai_client_cls("openai.gpt-5.6-luna", temperature=0.0, max_tokens=256)
        assert client.temperature == 1.0
        assert client.requested_temperature == 0.0
        assert client.effective_temperature == 1.0
        assert client.max_tokens == 1024
        assert 'pins the effective temperature to 1.0' in caplog.text

    def test_gpt5_default_temperature_needs_no_override_warning(
        self, openai_client_cls, caplog
    ):
        client = openai_client_cls('openai.gpt-5.6-luna', temperature=1.0)
        assert client.requested_temperature == client.effective_temperature == 1.0
        assert 'pins the effective temperature' not in caplog.text


class TestOutputFilenameSafety:
    """A `provider:` prefix must never reach a filename.

    On Windows, writing to ``openai:model.csv`` silently redirects the content
    into an NTFS alternate data stream on a file named ``openai`` — the results
    vanish with no error.
    """

    @pytest.mark.parametrize("model", [
        "openai:anthropic.claude-sonnet-5",
        "openai:openai.gpt-5.6-luna",
        "anthropic:claude-3-5-sonnet-latest",
        "vllm:meta-llama/Llama-3-8b",
    ])
    def test_no_colon_or_separator_survives(self, model):
        short = short_model_name(model)
        assert ":" not in short
        assert "/" not in short and "\\" not in short

    def test_evaluate_models_uses_short_name(self):
        """The dyadic evaluator must name its CSV with short_model_name."""
        import pathlib

        source = (
            pathlib.Path(__file__).parent.parent
            / "src" / "conflictxgame" / "solo" / "evaluate_models.py"
        )
        text = source.read_text(encoding="utf-8")
        assert 'f"{short_model_name(args.model)}.csv"' in text
        assert "args.model.split('/')[-1]}.csv" not in text
