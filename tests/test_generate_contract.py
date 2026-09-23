from types import SimpleNamespace

import pytest

from conflictxgame.common import model_wrappers as wrappers


class FakeAPIError(Exception):
    pass


def test_openai_final_temperature_fallback_returns_empty(monkeypatch):
    monkeypatch.setattr(wrappers, 'OpenAI', lambda *args, **kwargs: object())
    monkeypatch.setattr(wrappers, 'APIError', FakeAPIError)
    client = wrappers.OpenAIClient('gateway-model', max_retries=1)

    def reject(**_kwargs):
        raise FakeAPIError('Unsupported parameter: temperature')

    client.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=reject))
    )
    assert client.generate([{'role': 'user', 'content': 'hello'}]) == ''
    assert client.consecutive_failures == 1


def test_zero_retries_is_clamped_to_one(monkeypatch, caplog):
    monkeypatch.setattr(wrappers, 'OpenAI', lambda *args, **kwargs: object())
    client = wrappers.OpenAIClient('gateway-model', max_retries=0)
    assert client.max_retries == 1
    assert 'clamping to one attempt' in caplog.text


def test_anthropic_request_parameter_error_is_not_masked(monkeypatch):
    monkeypatch.setattr(wrappers, 'Anthropic', lambda: object())
    monkeypatch.setattr(wrappers, 'APIStatusError', FakeAPIError)
    client = wrappers.AnthropicClient('claude-test', max_retries=1)
    error = FakeAPIError('bad request parameters')

    def fail(_messages):
        raise error

    monkeypatch.setattr(client, '_anthropic_request_params', fail)
    with pytest.raises(FakeAPIError, match='bad request parameters'):
        client.generate([{'role': 'user', 'content': 'hello'}])
